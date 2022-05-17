import argparse
import numpy as np
from easydict import EasyDict as ed
# Configuration for model definition
hrnetw48_config = ed({
    "extra": {
        "FINAL_CONV_KERNEL": 1,
        "STAGE1": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 1,
            "BLOCK": "BOTTLENECK",
            "NUM_BLOCKS": [4],
            "NUM_CHANNELS": [64],
            "FUSE_METHOD": "SUM"
        },
        "STAGE2": {
            "NUM_MODULES": 1,
            "NUM_BRANCHES": 2,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4],
            "NUM_CHANNELS": [48, 96],
            "FUSE_METHOD": "SUM"
        },
        "STAGE3": {
            "NUM_MODULES": 4,
            "NUM_BRANCHES": 3,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4],
            "NUM_CHANNELS": [48, 96, 192],
            "FUSE_METHOD": "SUM"
        },
        "STAGE4": {
            "NUM_MODULES": 3,
            "NUM_BRANCHES": 4,
            "BLOCK": "BASIC",
            "NUM_BLOCKS": [4, 4, 4, 4],
            "NUM_CHANNELS": [48, 96, 192, 384],
            "FUSE_METHOD": "SUM"
        }
    },
})
def obtain_search_args():
    parser = argparse.ArgumentParser(description="search args")

    # checking point
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='search/first', help='set the checkpoint name')
    parser.add_argument('--model_encode_path', type=str, default='/media/dell/DATA/wy/Seg_NAS/model/model_encode/GID-5/14layers_mixedcell1_3operation/first_connect_4.npy')
    parser.add_argument('--search_stage', type=str, default='first', choices=['first', 'second', 'third', 'hrnet'], help='witch search stage')

    parser.add_argument('--batch-size', type=int, default=12, metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--dataset', type=str, default='uadataset', choices=['uadataset', 'cityscapes'], help='dataset name (default: pascal)')
    parser.add_argument('--data_path', type=str, default='/media/dell/DATA/wy/data', help='dataset root path')

    parser.add_argument('--workers', type=int, default=0,metavar='N', help='dataloader threads')
    parser.add_argument('--crop_size', type=int, default=512, help='crop image size')
    parser.add_argument('--resize', type=int, default=512, help='resize image size')

    parser.add_argument('--nclass', type=int, default=12, help='number of class')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=60, metavar='N', help='number of epochs to train')
    parser.add_argument('--alpha_epochs', type=int, default=20, metavar='N', help='number of alpha epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--eval_start', type=int, default=20, metavar='N', help='start eval epochs (default:0)')
    parser.add_argument('--filter_multiplier', type=int, default=8)
    parser.add_argument('--block_multiplier', type=int, default=5)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--alpha_epoch', type=int, default=20, metavar='N', help='epoch to start training alphas')

    # optimizer params
    parser.add_argument('--lr', type=float, default=0.025, metavar='LR',help='learning rate (default: auto)')
    parser.add_argument('--min_lr', type=float, default=0.001)
    parser.add_argument('--arch-lr', type=float, default=3e-3, metavar='LR', help='learning rate for alpha and beta in architect searching process')

    parser.add_argument('--lr-scheduler', type=str, default='cos', choices=['poly', 'step', 'cos'], help='lr scheduler mode')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=3e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--arch-weight-decay', type=float, default=1e-3, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov (default: False)')

    # cuda, seed and logging
    parser.add_argument('--gpu-ids', type=str, default='0',help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluuation interval (default: 1)')
    parser.add_argument('--val', action='store_true', default=True, help='skip validation during training')
    parser.add_argument('--affine', default=False, type=bool, help='whether use affine in BN')
    parser.add_argument('--multi_scale', default=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0), type=bool, help='whether use multi_scale in train')
    args = parser.parse_args()

    return args

def obtain_retrain_args():
    parser = argparse.ArgumentParser(description="search args")

    # checking point
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='retrain', help='set the checkpoint name')
    parser.add_argument('--model_name', type=str, default='flexinet')

    parser.add_argument('--batch-size', type=int, default=10, metavar='N', help='input batch size for training (default: auto)')
    parser.add_argument('--dataset', type=str, default='uadataset', choices=['uadataset', 'cityscapes'], help='dataset name (default: pascal)')
    parser.add_argument('--data_path', type=str, default='/media/dell/DATA/wy/data', help='dataset root path')
    parser.add_argument('--local_train_url', type=str, default='./run/save', help='model saved path')

    parser.add_argument('--workers', type=int, default=8, metavar='N', help='dataloader threads')
    parser.add_argument('--crop_size', type=int, default=512, help='crop image size')
    parser.add_argument('--resize', type=int, default=512, help='resize image size')

    parser.add_argument('--nclass', type=int, default=12, help='number of class')

    # training hyper params
    parser.add_argument('--epochs', type=int, default=60, metavar='N', help='number of epochs to train')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N', help='start epochs (default:0)')
    parser.add_argument('--eval_start', type=int, default=20, metavar='N', help='start eval epochs (default:0)')
    parser.add_argument('--filter_multiplier', type=int, default=8)
    parser.add_argument('--block_multiplier', type=int, default=5)

    # optimizer params
    parser.add_argument('--lr', type=float, default=0.025, metavar='LR',help='learning rate (default: auto)')
    parser.add_argument('--min_lr', type=float, default=0.001)
    parser.add_argument('--arch-lr', type=float, default=3e-3, metavar='LR', help='learning rate for alpha and beta in architect searching process')

    parser.add_argument('--lr-scheduler', type=str, default='cos', choices=['poly', 'step', 'cos'], help='lr scheduler mode')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=3e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--arch-weight-decay', type=float, default=1e-3, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov (default: False)')

    # cuda, seed and logging
    parser.add_argument('--gpu-ids', type=str, default='0',help='use which gpu to train, must be a comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluuation interval (default: 1)')
    parser.add_argument('--val', action='store_true', default=True, help='skip validation during training')
    parser.add_argument('--affine', default=False, type=bool, help='whether use affine in BN')
    parser.add_argument('--multi_scale', default=(0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0), type=bool, help='whether use multi_scale in train')
    args = parser.parse_args()

    return args
