# -*- coding:utf-8 -*-
import os
from attrdict import AttrDict
import yaml
import argparse

import luojianet_ms
import luojianet_ms.nn as nn
from luojianet_ms import context
from luojianet_ms.common import set_seed
from luojianet_ms.common.initializer import Normal
from luojianet_ms.train.callback import LossMonitor, ModelCheckpoint, CheckpointConfig, TimeMonitor
from luojianet_ms import Model, load_checkpoint, load_param_into_net

from src.luojianet_ml_tool.network_define import TrainOneStepCell, LossCallBack
from src.luojianet_ml_tool.dataset import *
from src.luojianet_ml_tool.utils import *
from src.luojianet_ml_tool.lr_schedule import dynamic_lr
from src.luojianet_ml_tool.model import ResNet152_ml, LeNet5
import luojianet_ms.ops as ops
from luojianet_ms import ms_function


class LossNet(nn.Module):
    def __init__(self, net, cfg, testset=None, trainset=None):
        super(LossNet, self).__init__(auto_prefix=False)
        self.net = net
        self.trainset = trainset
        self.testset = testset
        self.iter = 0
        self.loss_record_list = []
        self.distance = distances.CosineSimilarity()
        self.reducer = reducers.ThresholdReducer(low=0)
        self.loss_func = losses.MultiSimilarityLoss(alpha=1.5, beta=70, base=0.5) #
        self.mining_func = miners.MultiSimilarityMiner(epsilon=0.1)
        # self.accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)

    def forward(self, x, labels):
        # cb_param = run_context.original_args()
        # cur_epoch = cb_param.cur_epoch_num
        self.iter += 1
        embeddings = self.net(x)
        indices_tuple = self.mining_func(embeddings, labels)
        # print('-----', indices_tuple)
        loss = self.loss_func(embeddings, labels, indices_tuple)


        # self.loss_record_list.append(loss.asnumpy())  # GRAPH-MODE not support
        # print('----loss', loss)

        # if not self.testset is None:
        #     cur_acc = test(self.trainset, self.testset, self.net, self.accuracy_calculator)
        #     # print('----cur_acc', cur_acc)
        return loss


if __name__ == "__main__":


    """set seed"""
    # set_seed(1)

    parser = argparse.ArgumentParser(description="default name", add_help=False)
    parser.add_argument("--config_path", required=True, type=str, default='./configs/ml_standard.yaml', help="Config file path")

    """get config file"""
    cfg_path = parser.parse_args().config_path
    with open(cfg_path, 'r') as f:
        cfg = AttrDict(yaml.load(f, Loader=yaml.FullLoader))

    """set gpu devices"""
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=int(cfg.Device_id))

    """train setting"""
    num_epochs = cfg.TRAIN.epoch

    """set train and test dataloader"""
    trainset = DatasetGenerator(cfg, test_mode=False)
    trainloader = create_ml_dataset(trainset, cfg.TRAIN.batch_size, is_training=True, shuffle=True, is_aug=True, num_parallel_workers=1)
    print('dataset size: ', trainloader.get_dataset_size())

    """set model"""
    # net = LeNet5(num_class=16, num_channel=3)
    net = ResNet152_ml(cfg, num_class=128)
    net.set_train(True)
    """loss function"""
    loss_net = LossNet(net, cfg)

    """learning rate"""
    cur_lr = luojianet_ms.Tensor(dynamic_lr(cfg, trainloader.get_dataset_size()), luojianet_ms.float32)

    """optimizer"""
    optimizer = nn.Adam(net.trainable_params(), learning_rate=cur_lr)

    """forward back functions"""
    cb = [LossMonitor(1), TimeMonitor(data_size=trainloader.get_dataset_size()), LossCallBack(cfg, rank_id=0)]
    # add forward back functions
    ckptconfig = CheckpointConfig(save_checkpoint_steps=cfg.save_checkpoint_epochs * trainloader.get_dataset_size(),
                                  keep_checkpoint_max=cfg.keep_checkpoint_max)
    save_checkpoint_path = os.path.join(cfg.save_checkpoint_path)
    ckpoint_cb = ModelCheckpoint(prefix='gt_det_12', directory=save_checkpoint_path, config=ckptconfig)
    cb += [ckpoint_cb]

    """set model"""
    model = TrainOneStepCell(loss_net, optimizer, degree=1)
    model = Model(model, amp_level="O0")

    """train model"""
    model.train(num_epochs, trainloader, callbacks=cb, dataset_sink_mode=False)



