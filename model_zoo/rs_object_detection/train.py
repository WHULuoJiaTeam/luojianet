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
"""train model and get checkpoint files."""
import os
import time

from src.luojia_detection.configuration.config import config
from src.luojia_detection.env.moxing_adapter import moxing_wrapper
from src.luojia_detection.env.device_adapter import get_device_id, get_device_num
from src.luojia_detection.detectors import Mask_Rcnn_Resnet,Faster_Rcnn_Resnet
from src.luojia_detection.callback_functions import LossCallBack, WithLossCell, TrainOneStepCell, LossNet
from src.luojia_detection.datasets import create_maskrcnn_dataset, create_fasterrcnn_dataset
from src.luojia_detection.solver import dynamic_lr

import luojianet_ms.common.dtype as mstype
from luojianet_ms import context, Tensor, Parameter
from luojianet_ms.communication.management import init, get_rank, get_group_size
from luojianet_ms.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor, LossMonitor
from luojianet_ms.train import Model
from luojianet_ms.context import ParallelMode
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from luojianet_ms.nn import Momentum, SGD
from luojianet_ms.common import set_seed


def modelarts_pre_process():
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),\
                    int(int(time.time() - s_time) % 60)))
                print("Extract Done")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))
        print("#" * 200, os.listdir(save_dir_1))
        print("#" * 200, os.listdir(os.path.join(config.data_path, config.modelarts_dataset_unzip_name)))

        config.coco_root = os.path.join(config.data_path, config.modelarts_dataset_unzip_name)
    config.pre_trained = os.path.join(config.coco_root, config.pre_trained)
    config.save_checkpoint_path = config.output_path


def load_pretrained_ckpt(net, load_path, device_target):
    param_dict = load_checkpoint(load_path)

    if config.pre_trained != '':
        key_mapping = {'down_sample_layer.1.beta': 'bn_down_sample.beta',
                       'down_sample_layer.1.gamma': 'bn_down_sample.gamma',
                       'down_sample_layer.0.weight': 'conv_down_sample.weight',
                       'down_sample_layer.1.moving_mean': 'bn_down_sample.moving_mean',
                       'down_sample_layer.1.moving_variance': 'bn_down_sample.moving_variance',
                       }
        for oldkey in list(param_dict.keys()):
            if not oldkey.startswith(('backbone', 'end_point', 'global_step',
                                      'learning_rate', 'moments', 'momentum')):
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
            if config.mask_on:
                if not (item.startswith('backbone') or item.startswith('rcnn_mask')):
                    param_dict.pop(item)
            else:
                if not item.startswith('backbone'):
                    param_dict.pop(item)

    if device_target == 'GPU':
        for key, value in param_dict.items():
            tensor = Tensor(value, mstype.float32)
            param_dict[key] = Parameter(tensor, key)

    load_param_into_net(net, param_dict)
    return net


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_model():
    print("Start train!")

    dataset_sink_mode_flag = False
    if config.run_distribute:
        if config.device_target == "Ascend":
            init()
            rank = get_rank()
            dataset_sink_mode_flag = device_target == 'Ascend'
            device_num = get_group_size()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        else:
            init("nccl")
            context.reset_auto_parallel_context()
            rank = get_rank()
            device_num = get_group_size()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True, parameter_broadcast=True)
    else:
        rank = 0
        device_num = 1

    print("Start create dataset!")

    if config.mask_on:
        prefix = 'mask_rcnn'
        dataset = create_maskrcnn_dataset(batch_size=config.batch_size,
                                           device_num=device_num, rank_id=rank)
    else:
        prefix = 'faster_rcnn'
        dataset = create_fasterrcnn_dataset(batch_size=config.batch_size,
                                           device_num=device_num, rank_id=rank)

    dataset_size = dataset.get_dataset_size()
    print("total images num: ", dataset_size)
    print("Create dataset done!")
    if config.mask_on:
        net = Mask_Rcnn_Resnet(config=config)
    else:
        net = Faster_Rcnn_Resnet(config=config)
    net = net.set_train()

    load_path = config.pre_trained
    if load_path != "":
        print("Loading pretrained resnet checkpoint")
        net = load_pretrained_ckpt(net=net, load_path=load_path, device_target=device_target)

    loss = LossNet()
    lr = Tensor(dynamic_lr(config, dataset_size), mstype.float32)

    opt = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=config.momentum,
              weight_decay=config.weight_decay, loss_scale=config.loss_scale)

    net_with_loss = WithLossCell(net, loss)
    if config.run_distribute:
        net = TrainOneStepCell(net_with_loss, opt, sens=config.loss_scale, reduce_flag=True,
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
        ckpoint_cb = ModelCheckpoint(prefix=prefix, directory=save_checkpoint_path, config=ckptconfig)
        cb += [ckpoint_cb]

    model = Model(net)
    model.train(config.epoch_size, dataset, callbacks=cb, dataset_sink_mode=dataset_sink_mode_flag)


if __name__ == '__main__':
    import os

    # set environment for training
    device_target = config.device_target

    # PYNATIVE_MODE for debug, GRAPH_MODE for training
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device_id)
    context.set_context(mode=context.PYNATIVE_MODE, device_target=device_target, device_id=0)

    # set random seed, if necessary
    # set_seed(1)

    # run training
    train_model()
