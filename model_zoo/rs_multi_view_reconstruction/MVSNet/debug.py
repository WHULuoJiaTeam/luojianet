
import argparse
import os
import numpy as np
import time
import sys
import datetime
import matplotlib.pyplot as plt
from preprocess import write_cam, save_pfm
import luojianet_ms.dataset as ds
from dataset_copy import MVSDatasetGenerator
from mvsnet_debug import MVSNet
from luojianet_ms import Tensor, ops
from loss import MVSNetWithLoss
from luojianet_ms import nn, Model
from luojianet_ms.train.callback import Callback, ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from luojianet_ms import context
from luojianet_ms import load_checkpoint, load_param_into_net
import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser(description='A PyTorch Implementation of MVSNet')
parser.add_argument('--mode', default='test', help='train or test', choices=['train', 'test', 'profile'])
parser.add_argument('--model', default='rednet', help='select model', choices=['mvsnet', 'rmvsnet', 'rednet'])
parser.add_argument('--dataset', default='whu', help='select dataset')

parser.add_argument('--data_root', default='/mnt/gj/stereo', help='train datapath')
parser.add_argument('--loadckpt', default='./checkpoints/whu_rednet/model_000008_0.1282.ckpt', help='load a specific checkpoint')
# parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/mvsnet/', help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')
parser.add_argument('--normalize', type=str, default='standard', help='methods of center_image, mean[mean var] or standard[0-1].')

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
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

# parse arguments and check
args = parser.parse_args()


def create_dataset(mode, args):
    ds.config.set_seed(1)
    dataset_generator = MVSDatasetGenerator(args.data_root, mode, args.view_num, args.normalize, args)

    input_data = ds.GeneratorDataset(dataset_generator,
                                     column_names=["image", "camera", "target", "values", "mask"],
                                     shuffle=True)
    input_data = input_data.batch(batch_size=args.batch_size)

    return input_data


# model_path = "checkpoint/checkpoint_mvsnet_whu_7-500_1.ckpt"
net = MVSNet(384, 768, False)
model = Model(net)

dataset_generator = MVSDatasetGenerator(args.data_root, "train", args.view_num, args.normalize, args)
ds_train = create_dataset("train", args)

i = 1
total_time = 0
for data in ds_train.create_dict_iterator():
    start = time.time()

    imgs = Tensor(data['image'])
    cams = Tensor(data['camera'])
    values = Tensor(data['values'])

    output = model.predict(imgs, cams, values)

    print(output.shape)

    end = time.time()

    print("{} cost {}".format(i, end - start))
    total_time += end - start
    i += 1

    if i > 100:
        break


print("cost {}".format(total_time / i))

# data, label = dataset_generator.__getitem__(0)
#
# expand_dims = ops.ExpandDims()
#
# data = expand_dims(Tensor(data), 0)
# label = expand_dims(Tensor(label), 0)
#
# # param_dict = load_checkpoint(model_path)
# # load_param_into_net(net, param_dict)
#
# output = net(data)
#
# # print(depth_values[0, :, 0, 0])
# # print(output)
#
# print(output.shape)
# #
# # for i, idx in zip(range(4), [50, 100, 150, 200]):
# #     plt.subplot(2, 2, i+1)
# #     plt.imshow(output[0, 0, idx-1].asnumpy())
# #
# # plt.show()
#
# # cv2.imwrite("test.tif", output.asnumpy()[0])
# plt.subplot(1, 2, 1)
# plt.imshow(output.asnumpy()[0])
# plt.subplot(1, 2, 2)
# plt.imshow(label.asnumpy()[0])
# plt.show()
#
# # idx = 0
# # for i in [0, 49, 99, 149, 199]:
# #     idx += 1
# #     plt.subplot(2, 3, idx)
# #     plt.imshow(output.asnumpy()[0, 0, i])
# # plt.show()
