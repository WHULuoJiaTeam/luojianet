# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""LearningRateScheduler Callback class."""

import math
import numpy as np

from luojianet_ms import log as logger
import luojianet_ms.common.dtype as mstype
from luojianet_ms.common.tensor import Tensor
from luojianet_ms.train.callback._callback import Callback
from luojianet_ms.train.serialization import save_checkpoint
from attrdict import AttrDict
import yaml
import os

# from pytorch_metric_learning import losses, miners, distances, reducers, samplers
# from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator
# from pytorch_metric_learning.utils import common_functions as c_f


class MLTrain(Callback):

    def __init__(self, ml_dir, cfg_path, total_epochs, num_classes, save_dir):
        super(MLTrain, self).__init__()
        self.ml_dir = ml_dir
        self.cfg_path = cfg_path
        self.total_epochs = total_epochs
        self.begin_ml = True
        self.num_cls = num_classes
        self.save_dir = save_dir
        
    def epoch_begin(self, run_context):
        cb_params = run_context.original_args()

        ### frozen param to crop img
        if cb_params.cur_epoch_num == self.total_epochs:
            self.begin_ml = True
            for param in list(cb_params.train_network.cells_and_names()):
                param[1].requires_grad = False
                # if "ml_cls" in param[0]:
                #     param[1].requires_grad = False
        
    def epoch_end(self, run_context):
        if self.begin_ml:
            import sys
            sys.path.append(self.ml_dir)
            from model import NET
            from utils import train,StepLR
            from dataset import TrainDataset

            cfg_path = self.cfg_path
            with open(cfg_path, 'r') as f:
                cfg = AttrDict(yaml.load(f, Loader=yaml.FullLoader))

            # train setting
            num_epochs = cfg.TRAIN.epoch
            base_lr = cfg.TRAIN.lr
            alpha = cfg.TRAIN.alpha
            beta = cfg.TRAIN.beta
            base = cfg.TRAIN.base
            m = cfg.TRAIN.mPerCls

            # set train and test dataloader
            trainset = TrainDataset(cfg, test_mode=False)
            sampler = samplers.MPerClassSampler(trainset.targets, m=m, length_before_new_iter=len(trainset))

            trainloader = c_f.get_train_dataloader(trainset, batch_size=m * self.num_cls, sampler=sampler,
                                                   num_workers=cfg.TRAIN.num_workers,
                                                   collate_fn=None)

            # set model
            print(len(trainset))
            device = torch.device("cuda")
            writer = None
            runner.model.module.ml_head = NET(cfg, writer).to(device)

            # optimizer
            optimizer = optim.Adam(runner.model.module.ml_head.parameters(), lr=base_lr)

            # pytorch-metric-learning stuff
            distance = distances.CosineSimilarity()

            reducer = reducers.ThresholdReducer(low=0)

            loss_func = losses.MultiSimilarityLoss(alpha=alpha, beta=beta, base=base)
            mining_func = miners.MultiSimilarityMiner(epsilon=0.1)

            # training

            for epoch in range(1, num_epochs + 1):
                print('---------- epoch:{} ----------'.format(epoch))
                loss = train(runner.model.module.ml_head, loss_func, mining_func, device, trainloader, optimizer, epoch,
                             cfg)

                cur_lr = StepLR(optimizer, base_lr, num_epochs, epoch)

                if epoch > 90 and epoch % 40 == 0:
                    cb_params = run_context.original_args()
                    network = cb_params.train_network
                    save_name = "epoch_" + str(cb_params.cur_epoch_num) + "ml_" + str(epoch) +".ckpt"
                    save_path = os.path.join(self.save_dir, save_name)
                    save_checkpoint(network, save_path)
                    print('----- save ml model -----')
