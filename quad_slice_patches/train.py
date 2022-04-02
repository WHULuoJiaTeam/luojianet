##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: ZhangZhan
## Wuhan University
## zhangzhanstep@whu.edu.cn
## Copyright (c) 2022
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
from luojianet_ms.common import set_seed
import luojianet_ms.nn as nn
from luojianet_ms import context
from luojianet_ms.communication.management import init, get_rank
from luojianet_ms.context import ParallelMode
from luojianet_ms.train.serialization import load_checkpoint, load_param_into_net
from luojianet_ms.train.callback import LossMonitor, TimeMonitor
from luojianet_ms.train.loss_scale_manager import FixedLossScaleManager
from luojianet_ms.train.model import Model
from luojianet_ms.train.callback import ModelCheckpoint, CheckpointConfig

import settings
from dataset import get_dataset
import luojianet_ms.dataset as ds
from network import DeepLabV3Plus, SoftmaxCrossEntropyLoss

set_seed(1)


def cosine_lr(base_lr, decay_steps, total_steps):
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        yield base_lr * 0.5 * (1 + np.cos(np.pi * step_ / decay_steps))


def poly_lr(base_lr, decay_steps, total_steps, end_lr=0.0001, power=0.9):
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        yield (base_lr - end_lr) * ((1.0 - step_ / decay_steps) ** power) + end_lr


def exponential_lr(base_lr, decay_steps, decay_rate, total_steps, staircase=False):
    for i in range(total_steps):
        if staircase:
            power_ = i // decay_steps
        else:
            power_ = float(i) / decay_steps
        yield base_lr * (decay_rate ** power_)


class BuildTrainNetwork(nn.Module):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def call(self, image, label):
        pred = self.network(image)
        loss = self.criterion(pred, label)
        return loss


def train(dataset):
    args = settings.get_args()

    # init multicards training
    if args.device_target == "CPU":
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="CPU")
    else:
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="GPU")
    if args.is_distributed:
        init("nccl")
        args.rank = get_rank()
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode,
                                          gradients_mean=True,
                                          device_number=args.device_number)

    # network
    if args.model_name == 'DeepLabV3plus_s16':
        net = DeepLabV3Plus(phase='train', num_classes=args.num_classes,
                            output_stride=16, freeze_bn=args.freeze_bn)
    elif args.model_name == 'DeepLabV3plus_s8':
        net = DeepLabV3Plus(phase='train', num_classes=args.num_classes,
                            output_stride=8, freeze_bn=args.freeze_bn)
    else:
        raise NotImplementedError

    # loss
    loss = SoftmaxCrossEntropyLoss(args.num_classes, args.ignore_label)
    train_net = BuildTrainNetwork(net, loss)

    # load pretrained model
    if args.ckpt_file:
        param_dict = load_checkpoint(args.ckpt_file)
        load_param_into_net(train_net, param_dict)

    # optimizer
    iters_per_epoch = dataset.get_dataset_size()
    total_train_steps = iters_per_epoch * args.train_epochs
    if args.lr_type == 'cos':
        lr_iter = cosine_lr(args.base_lr, total_train_steps, total_train_steps)
    elif args.lr_type == 'poly':
        lr_iter = poly_lr(args.base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    elif args.lr_type == 'exp':
        lr_iter = exponential_lr(args.base_lr, args.lr_decay_step, args.lr_decay_rate,
                                 total_train_steps, staircase=True)
    else:
        raise NotImplementedError
    opt = nn.Momentum(params=train_net.trainable_params(), learning_rate=lr_iter,
                      momentum=0.9, weight_decay=0.0001, loss_scale=args.loss_scale)

    # loss scale
    manager_loss_scale = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)

    # Callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()
    cbs = [time_cb, loss_cb]
    if args.rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=args.save_steps,
                                     keep_checkpoint_max=args.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix=args.model_name, directory=args.model_dir, config=config_ck)
        cbs.append(ckpoint_cb)

    # define model and train
    model = Model(train_net, optimizer=opt, loss_scale_manager=manager_loss_scale)
    model.train(epoch=args.train_epochs, train_dataset=dataset, callbacks=cbs)


def fun_target(image_queue, label_queue):
    class ObjectDataset:
        def __init__(self):
            self.image_queue = image_queue
            self.label_queue = label_queue

        def __getitem__(self, index):
            image_sample = self.image_queue[index]
            label_sample = self.label_queue[index]
            return image_sample, label_sample

        def __len__(self):
            return len(self.image_queue)

    dataset = ds.GeneratorDataset(source=ObjectDataset(), column_names=["image_sample", "label_sample"], shuffle=False)
    dataset = dataset.repeat(count=1)
    train(dataset=dataset)


def main_train():
    # Create main dataset
    dataset = get_dataset(split='train', repeat=1)
    dataset_size = dataset.get_dataset_size()
    print("Main dataset create success, dataset_size=%d", dataset_size)

    # Train for each image_list and label_list
    image_queue = []
    label_queue = []
    for data in dataset.create_dict_iterator():
        image_list = data["image_list"]
        label_list = data["label_list"]
        _, list_size, _, _, _ = image_list.shape
        # Remove first batch dimension
        image_list = image_list[0, :, :, :, :]
        label_list = label_list[0, :, :, :, :]
        for i in range(list_size):
            image_queue.append(image_list[i])
            label_queue.append(label_list[i])
        fun_target(image_queue, label_queue)


if __name__ == '__main__':
    main_train()