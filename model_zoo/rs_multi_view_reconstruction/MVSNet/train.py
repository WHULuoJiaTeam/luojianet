
import argparse
import os
import numpy as np
import time
import sys
import datetime
import matplotlib.pyplot as plt
import luojianet_ms.dataset as ds
from src.dataset import MVSDatasetGenerator
from src.mvsnet import MVSNet
from src.loss import MVSNetWithLoss
from luojianet_ms import nn, Model
from luojianet_ms.train.callback import Callback, ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, LearningRateScheduler
from luojianet_ms import context

from luojianet_ms.nn import learning_rate_schedule
from luojianet_ms import load_checkpoint, load_param_into_net


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='A LuoJiaNET Implementation of MVSNet')
parser.add_argument('--dataset', default='whu', help='select dataset')

parser.add_argument('--data_root', default='/mnt/gj/stereo', help='train datapath')
parser.add_argument('--logdir', default='./checkpoints_mvsnet/', help='the directory to save checkpoints/logs')
parser.add_argument('--normalize', type=str, default='mean', help='methods of center_image, mean[mean var] or standard[0-1].')

# input parameters
parser.add_argument('--view_num', type=int, default=3, help='Number of images (1 ref image and view_num - 1 view images).')
parser.add_argument('--ndepths', type=int, default=200, help='the number of depth values')
parser.add_argument('--max_w', type=int, default=768, help='Maximum image width')
parser.add_argument('--max_h', type=int, default=384, help='Maximum image height')
parser.add_argument('--resize_scale', type=float, default=1, help='output scale for depth and image (W and H)')
parser.add_argument('--sample_scale', type=float, default=0.25, help='Downsample scale for building cost volume (W and H)')
parser.add_argument('--interval_scale', type=float, default=1, help='the number of depth values')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--adaptive_scaling', type=bool, default=True, help='Let image size to fit the network, including scaling and cropping')

# network architecture
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--decay_rate', type=float, default=0.9, help='decay rate for lr adjustment')
parser.add_argument('--decay_step', type=int, default=5000, help='decay step for lr adjustment')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

# parse arguments and check
args = parser.parse_args()


def create_dataset(mode, args):
    ds.config.set_seed(args.seed)
    dataset_generator = MVSDatasetGenerator(args.data_root, mode, args.view_num, args.normalize, args)

    input_data = ds.GeneratorDataset(dataset_generator,
                                     column_names=["image", "camera", "target", "values", "mask"],
                                     shuffle=True)
    input_data = input_data.batch(batch_size=args.batch_size)

    return input_data


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    train_dataset = create_dataset("train", args)
    val_dataset = create_dataset("test", args)

    train_data_size = train_dataset.get_dataset_size()
    print(args)
    print(train_data_size)

    # create network
    net = MVSNet(args.max_h, args.max_w, False)
    net_with_loss = MVSNetWithLoss(net)

    # learning rate and optimizer
    learning_rate = learning_rate_schedule.ExponentialDecayLR(args.lr, args.decay_rate, args.decay_step)
    net_opt = nn.RMSProp(net.trainable_params(), learning_rate=learning_rate)

    model = Model(net_with_loss, loss_fn=None, optimizer=net_opt, amp_level='O0')
    config_ck = CheckpointConfig(save_checkpoint_steps=train_data_size, keep_checkpoint_max=100)
    ckpoint_cb = ModelCheckpoint(prefix="checkpoint_mvsnet_whu", directory=args.logdir, config=config_ck)

    time_cb = TimeMonitor()

    output = model.train(args.epochs, train_dataset, callbacks=[ckpoint_cb, LossMonitor(1), time_cb],
                         dataset_sink_mode=False)

