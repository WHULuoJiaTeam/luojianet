# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""train FasterRcnn and get checkpoint files."""

import os
import time
import numpy as np

import luojianet_ms.common.dtype as mstype
from luojianet_ms import context, Tensor, Parameter
from luojianet_ms.communication.management import init, get_rank, get_group_size
from luojianet_ms.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor, LossMonitor
from luojianet_ms.train import Model
from luojianet_ms.context import ParallelMode
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from luojianet_ms.nn import SGD
from luojianet_ms.common import set_seed

from src.FasterRcnn.faster_rcnn import Faster_Rcnn
from src.network_define import LossCallBack, WithLossCell, TrainOneStepCell, LossNet
from src.dataset import data_to_mindrecord_byte_image, create_fasterrcnn_dataset
from src.dataset_generator import create_my_fasterrcnn_dataset
from src.lr_schedule import dynamic_lr
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num, get_rank_id
from src.ml_train_cb import MLTrain


def train_fasterrcnn_():
    """ train_fasterrcnn_ """
    print("Start create dataset!")

    # When create MindDataset, using the fitst mindrecord file, such as FasterRcnn.mindrecord0.
    # dataset = create_fasterrcnn_dataset(config, mindrecord_file, batch_size=config.batch_size,
    #                                     device_num=device_num, rank_id=rank,
    #                                     num_parallel_workers=config.num_parallel_workers,
    #                                     python_multiprocessing=config.python_multiprocessing)

    dataset = create_my_fasterrcnn_dataset(config, batch_size=config.batch_size,
                                            device_num=device_num, rank_id=rank,
                                            num_parallel_workers=config.num_parallel_workers,
                                            python_multiprocessing=config.python_multiprocessing)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done! -- dataset size: {}".format(dataset_size))

    return dataset_size, dataset


def modelarts_pre_process():
    config.save_checkpoint_path = config.output_path


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_fasterrcnn():
    """ train_fasterrcnn """
    dataset_size, dataset = train_fasterrcnn_()

    net = Faster_Rcnn(config=config)
    net = net.set_train()

    load_path = config.pre_trained
    if load_path != "":
        param_dict = load_checkpoint(load_path)

        key_mapping = {'down_sample_layer.1.beta': 'bn_down_sample.beta',
                       'down_sample_layer.1.gamma': 'bn_down_sample.gamma',
                       'down_sample_layer.0.weight': 'conv_down_sample.weight',
                       'down_sample_layer.1.moving_mean': 'bn_down_sample.moving_mean',
                       'down_sample_layer.1.moving_variance': 'bn_down_sample.moving_variance',
                       }

        ### del cls & bbox_reg
        if "fasterrcnnresnetv1152" in load_path:
            print('pop keys of resnet152')
            param_dict.pop("accum.rcnn.cls_scores.weight")
            param_dict.pop("accum.rcnn.cls_scores.bias")
            param_dict.pop("accum.rcnn.reg_scores.weight")
            param_dict.pop("accum.rcnn.reg_scores.bias")
            param_dict.pop("rcnn.cls_scores.weight")
            param_dict.pop("rcnn.cls_scores.bias")
            param_dict.pop("rcnn.reg_scores.weight")
            param_dict.pop("rcnn.reg_scores.bias")
            param_dict.pop("stat.rcnn.cls_scores.weight")
            param_dict.pop("stat.rcnn.cls_scores.bias")
            param_dict.pop("stat.rcnn.reg_scores.weight")
            param_dict.pop("stat.rcnn.reg_scores.bias")

        for oldkey in list(param_dict.keys()):
            if not oldkey.startswith(('backbone', 'end_point', 'global_step', 'learning_rate', 'moments', 'momentum')):
                data = param_dict.pop(oldkey)
                newkey = 'backbone.' + oldkey
                param_dict[newkey] = data
                oldkey = newkey
            for k, v in key_mapping.items():
                if k in oldkey:
                    newkey = oldkey.replace(k, v)
                    param_dict[newkey] = param_dict.pop(oldkey)
                    break
        for item in list(param_dict.keys()):
            if not item.startswith('backbone'):
                param_dict.pop(item)

        for key, value in param_dict.items():
            tensor = value.asnumpy().astype(np.float32)
            param_dict[key] = Parameter(tensor, key)
        load_param_into_net(net, param_dict)

    device_type = "Ascend" if context.get_context("device_target") == "Ascend" else "Others"
    if device_type == "Ascend":
        net.to_float(mstype.float16)

    loss = LossNet()
    lr = Tensor(dynamic_lr(config, dataset_size), mstype.float32)

    opt = SGD(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)
    net_with_loss = WithLossCell(net, loss)
    if config.run_distribute:
        net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale, reduce_flag=True,  # net, _  changed by xwj
                               mean=True, degree=device_num)
    else:
        net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale)

    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossCallBack(rank_id=rank)
    loss_monitor_cb = LossMonitor(per_print_times=1)

    cb = [time_cb, loss_cb, loss_monitor_cb]
    if config.save_checkpoint:
        ckptconfig = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * dataset_size,
                                      keep_checkpoint_max=config.keep_checkpoint_max)
        save_checkpoint_path = os.path.join(config.save_checkpoint_path, "ckpt_" + str(rank) + "/")
        ckpoint_cb = ModelCheckpoint(prefix='faster_rcnn', directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]

    if config.run_eval:
        from src.eval_callback import EvalCallBack
        from src.eval_utils import create_eval_mindrecord, apply_eval
        config.prefix = "FasterRcnn_eval.mindrecord"
        anno_json = os.path.join(config.coco_root, "annotations/instances_val2017.json")
        mindrecord_path = os.path.join(config.coco_root, "FASTERRCNN_MINDRECORD", config.prefix)
        config.instance_set = "annotations/instances_val2017.json"

        if not os.path.exists(mindrecord_path):
            config.mindrecord_file = mindrecord_path
            create_eval_mindrecord(config)
        eval_net = Faster_Rcnn(config)
        eval_cb = EvalCallBack(config, eval_net, apply_eval, dataset_size, mindrecord_path, anno_json,
                               save_checkpoint_path)
        cb += [eval_cb]

    model = Model(net, amp_level="O0")
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=False)


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id

    # set random seed
    set_seed(1)
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())  # get_device_id()
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=config.device_target, device_id=1)

    if config.device_target == "GPU":
        # context.set_context(enable_graph_kernel=True)  # by xwj
        pass
    if config.run_distribute:
        if config.device_target == "Ascend":
            rank = get_rank_id()
            device_num = get_device_num()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
        else:
            init("nccl")
            context.reset_auto_parallel_context()
            rank = get_rank()
            device_num = get_group_size()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    else:
        rank = 0
        device_num = 1

    train_fasterrcnn()


"""
mpirun -n 2 --output-filename log_output \
    python train.py  \
    --config_path=$CONFIG_FILE \
    --run_distribute=True \
    --device_target="GPU"
"""