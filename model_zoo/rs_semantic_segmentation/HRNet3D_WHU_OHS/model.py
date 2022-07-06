# HRNet-3D网络模型

import os
import logging

import numpy as np

import luojianet_ms.context as context
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
from luojianet_ms import Tensor

context.set_context(device_target="GPU")

BN_MOMENTUM = 0.99

logger = logging.getLogger(__name__)

stage1_18_cfg = {'NUM_CHANNELS': [64], 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4], 'NUM_MODULES': 1, 'NUM_BRANCHES': 1,
                 'FUSE_METHOD': 'SUM'}
stage2_18_cfg = {'NUM_CHANNELS': [18, 36], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4], 'NUM_MODULES': 1, 'NUM_BRANCHES': 2,
                 'FUSE_METHOD': 'SUM'}
stage3_18_cfg = {'NUM_CHANNELS': [18, 36, 72], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4], 'NUM_MODULES': 4,
                 'NUM_BRANCHES': 3, 'FUSE_METHOD': 'SUM'}
stage4_18_cfg = {'NUM_CHANNELS': [18, 36, 72, 144], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_MODULES': 3,
                 'NUM_BRANCHES': 4, 'FUSE_METHOD': 'SUM'}
hrnet_w18_cfg = {'stage1': stage1_18_cfg, 'stage2': stage2_18_cfg, 'stage3': stage3_18_cfg, 'stage4': stage4_18_cfg}

stage1_30_cfg = {'NUM_CHANNELS': [64], 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4], 'NUM_MODULES': 1, 'NUM_BRANCHES': 1,
                 'FUSE_METHOD': 'SUM'}
stage2_30_cfg = {'NUM_CHANNELS': [30, 60], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4], 'NUM_MODULES': 1, 'NUM_BRANCHES': 2,
                 'FUSE_METHOD': 'SUM'}
stage3_30_cfg = {'NUM_CHANNELS': [30, 60, 120], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4], 'NUM_MODULES': 4,
                 'NUM_BRANCHES': 3, 'FUSE_METHOD': 'SUM'}
stage4_30_cfg = {'NUM_CHANNELS': [30, 60, 120, 240], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_MODULES': 3,
                 'NUM_BRANCHES': 4, 'FUSE_METHOD': 'SUM'}
hrnet_w30_cfg = {'stage1': stage1_30_cfg, 'stage2': stage2_30_cfg, 'stage3': stage3_30_cfg, 'stage4': stage4_30_cfg}

stage1_40_cfg = {'NUM_CHANNELS': [64], 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4], 'NUM_MODULES': 1, 'NUM_BRANCHES': 1,
                 'FUSE_METHOD': 'SUM'}
stage2_40_cfg = {'NUM_CHANNELS': [40, 80], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4], 'NUM_MODULES': 1, 'NUM_BRANCHES': 2,
                 'FUSE_METHOD': 'SUM'}
stage3_40_cfg = {'NUM_CHANNELS': [40, 80, 160], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4], 'NUM_MODULES': 4,
                 'NUM_BRANCHES': 3, 'FUSE_METHOD': 'SUM'}
stage4_40_cfg = {'NUM_CHANNELS': [40, 80, 160, 320], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_MODULES': 3,
                 'NUM_BRANCHES': 4, 'FUSE_METHOD': 'SUM'}
hrnet_w40_cfg = {'stage1': stage1_40_cfg, 'stage2': stage2_40_cfg, 'stage3': stage3_40_cfg, 'stage4': stage4_40_cfg}

stage1_48_cfg = {'NUM_CHANNELS': [64], 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4], 'NUM_MODULES': 1, 'NUM_BRANCHES': 1,
                 'FUSE_METHOD': 'SUM'}
stage2_48_cfg = {'NUM_CHANNELS': [48, 96], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4], 'NUM_MODULES': 1, 'NUM_BRANCHES': 2,
                 'FUSE_METHOD': 'SUM'}
stage3_48_cfg = {'NUM_CHANNELS': [48, 96, 192], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4], 'NUM_MODULES': 4,
                 'NUM_BRANCHES': 3, 'FUSE_METHOD': 'SUM'}
stage4_48_cfg = {'NUM_CHANNELS': [48, 96, 192, 384], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_MODULES': 3,
                 'NUM_BRANCHES': 4, 'FUSE_METHOD': 'SUM'}
hrnet_w48_cfg = {'stage1': stage1_48_cfg, 'stage2': stage2_48_cfg, 'stage3': stage3_48_cfg, 'stage4': stage4_48_cfg}

stage1_64_cfg = {'NUM_CHANNELS': [64], 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4], 'NUM_MODULES': 1, 'NUM_BRANCHES': 1,
                 'FUSE_METHOD': 'SUM'}
stage2_64_cfg = {'NUM_CHANNELS': [64, 128], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4], 'NUM_MODULES': 1, 'NUM_BRANCHES': 2,
                 'FUSE_METHOD': 'SUM'}
stage3_64_cfg = {'NUM_CHANNELS': [64, 128, 256], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4], 'NUM_MODULES': 4,
                 'NUM_BRANCHES': 3, 'FUSE_METHOD': 'SUM'}
stage4_64_cfg = {'NUM_CHANNELS': [64, 128, 256, 512], 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_MODULES': 3,
                 'NUM_BRANCHES': 4, 'FUSE_METHOD': 'SUM'}
hrnet_w64_cfg = {'stage1': stage1_64_cfg, 'stage2': stage2_64_cfg, 'stage3': stage3_64_cfg, 'stage4': stage4_64_cfg}

# 光谱注意力机制 SE-Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SEBlock, self).__init__()

        self.gap = ops.AdaptiveAvgPool2D(output_size=1)

        self.seq = nn.SequentialCell(
            nn.Dense(in_channels, reduction_ratio),
            nn.ReLU(),
            nn.Dense(reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_ = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        v = self.gap(x_)
        v = v.reshape(x.shape[0], x.shape[1], x.shape[2], v.shape[2], v.shape[3])
        score = self.seq(v.reshape(v.shape[0], v.shape[1], v.shape[2]))
        y = x * score.reshape(score.shape[0], score.shape[1], score.shape[2], 1, 1)
        return y

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     pad_mode='pad', padding=1, has_bias=False)

# 卷积模块 BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

# 卷积模块 Bottleneck
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               pad_mode='pad', padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               has_bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out

# 多分辨率融合模块
class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()

        self.upsample = nn.ResizeBilinear()
        self.relu = nn.ReLU()

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM),
            )

        layers = nn.SequentialCell([])
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return layers

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = nn.CellList([])

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return branches

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        # fuse_layers = []
        fuse_layers = nn.CellList([])
        for i in range(num_branches if self.multi_scale_output else 1):
            # fuse_layer = []
            fuse_layer = nn.CellList([])
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.SequentialCell(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  kernel_size=1,
                                  stride=1,
                                  has_bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    # fuse_layer.append(None)
                    fuse_layer.append(nn.SequentialCell([]))
                else:
                    # conv3x3s = []
                    conv3x3s = nn.SequentialCell([])
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.SequentialCell(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                               momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.SequentialCell(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                               momentum=BN_MOMENTUM),
                                nn.ReLU()))
                    fuse_layer.append(conv3x3s)
            fuse_layers.append(fuse_layer)

        return fuse_layers

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + self.upsample(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse

blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck,
}

# HRNet-3D整体结构
class HigherHRNet_Binary(nn.Module):

    def __init__(self, num_classes=2, hr_cfg='w48'):
        super(HigherHRNet_Binary, self).__init__()
        self.hr_cfg = hr_cfg

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, pad_mode='pad', padding=1,
                               has_bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, pad_mode='pad', padding=1,
                               has_bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU()
        if hr_cfg == 'w64':
            hrnet_cfg = hrnet_w64_cfg
        if hr_cfg == 'w48':
            hrnet_cfg = hrnet_w48_cfg
            ly1dim = 64
        if hr_cfg == 'w40':
            hrnet_cfg = hrnet_w40_cfg
            ly1dim = 64
        if hr_cfg == 'w30':
            hrnet_cfg = hrnet_w30_cfg
            ly1dim = 64
        if hr_cfg == 'w18':
            ly1dim = 64
            hrnet_cfg = hrnet_w18_cfg
        if hr_cfg == 'w18_256':
            ly1dim = 64
            hrnet_cfg = hrnet_w18_cfg
            self.conv1 = nn.Conv2d(3, 64, kernel_size=1, stride=1, pad_mode='same',
                                   has_bias=False)
            self.conv2 = nn.Conv2d(64, 64, kernel_size=1, stride=1, pad_mode='same',
                                   has_bias=False)
        if hr_cfg == 'w18_4band':
            hrnet_cfg = hrnet_w18_cfg
            ly1dim = 64
            self.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=2, pad_mode='pad', padding=1,
                                   has_bias=False)
        if hr_cfg == 'w18_249band':
            hrnet_cfg = hrnet_w18_cfg
            ly1dim = 64
            self.conv1 = nn.Conv2d(249, 64, kernel_size=3, stride=2, pad_mode='pad', padding=1,
                                   has_bias=False)

        # 3D卷积与光谱注意力机制
        if hr_cfg == 'w18_3d2d_at':
            ly1dim = 192
            hrnet_cfg = hrnet_w18_cfg

            self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(17, 1, 1), stride=(2, 1, 1), pad_mode='pad')
            self.bn1 = nn.BatchNorm3d(64, momentum=BN_MOMENTUM)
            self.atblock = SEBlock(8, 2)
            self.conv2 = nn.Conv3d(64, 64, kernel_size=(4, 3, 3), stride=(2, 1, 1), pad_mode='pad', padding=(0, 0, 1, 1, 1, 1),
                                   has_bias=False)
            self.bn2 = nn.BatchNorm3d(64, momentum=BN_MOMENTUM)
        
        # HRNet网络
        self.stage1_cfg = hrnet_cfg['stage1']
        num_channels = self.stage1_cfg['NUM_CHANNELS'][0]
        block = blocks_dict[self.stage1_cfg['BLOCK']]
        num_blocks = self.stage1_cfg['NUM_BLOCKS'][0]
        self.layer1 = self._make_layer(block, ly1dim, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        self.stage2_cfg = hrnet_cfg['stage2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = hrnet_cfg['stage3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = hrnet_cfg['stage4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        last_inp_channels = np.int(np.sum(pre_stage_channels))

        input_channels = last_inp_channels

        self.fusion = nn.SequentialCell(
            nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=5, stride=1, pad_mode='same'),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(),
        )

        self.upsample = nn.ResizeBilinear()

        self.pre_binary = nn.Conv2d(last_inp_channels, num_classes, kernel_size=1, stride=1)

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = nn.CellList([])
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.SequentialCell(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  kernel_size=3,
                                  stride=1,
                                  pad_mode='same',
                                  has_bias=False),
                        nn.BatchNorm2d(
                            num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU()))
                else:
                    transition_layers.append(nn.SequentialCell([]))
            else:
                conv3x3s = nn.SequentialCell([])
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.SequentialCell(
                        nn.Conv2d(
                            inchannels, outchannels, kernel_size=3, stride=2, pad_mode='pad', padding=1, has_bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU()))
                transition_layers.append(conv3x3s)

        return transition_layers

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, has_bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = nn.SequentialCell([])
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return layers

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = nn.SequentialCell([])
        for i in range(num_modules):
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return modules, num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.hr_cfg == 'w18_3d2d_at':
            x = self.atblock(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        if self.hr_cfg == 'w18_3d2d_at':
            x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])

        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.stage2_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.stage3_cfg['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        x0_h, x0_w = y_list[0].shape[2], y_list[0].shape[3]
        x0 = self.upsample(y_list[0], size=(x0_h, x0_w), align_corners=True)
        x1 = self.upsample(y_list[1], size=(x0_h, x0_w), align_corners=True)
        x2 = self.upsample(y_list[2], size=(x0_h, x0_w), align_corners=True)
        x3 = self.upsample(y_list[3], size=(x0_h, x0_w), align_corners=True)

        concat = ops.Concat(axis=1)
        x = concat((x0, x1, x2, x3))

        x = self.fusion(x)

        binary = self.pre_binary(x)

        return binary
