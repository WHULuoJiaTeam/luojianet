# FreeNet网络模型

import luojianet_ms.context as context
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

# 光谱注意力机制 SE-Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(SEBlock, self).__init__()
        self.gap = ops.AdaptiveAvgPool2D(output_size=1)
        self.seq = nn.SequentialCell(
            nn.Dense(in_channels, in_channels // reduction_ratio, weight_init='he_uniform'),
            nn.ReLU(),
            nn.Dense(in_channels // reduction_ratio, in_channels, weight_init='he_uniform'),
            nn.Sigmoid()
        )

    def forward(self, x):
        v = self.gap(x)
        score = self.seq(v.reshape(v.shape[0], v.shape[1]))
        y = x * score.reshape(score.shape[0], score.shape[1], 1, 1)
        return y

# 3×3卷积模块
def conv3x3_bn_relu(in_channel, out_channel):
    return nn.SequentialCell(
        nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mode='same', weight_init='he_uniform', has_bias=True),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    )

# 光谱注意力机制模块
def repeat_block(block_channel, r, n):
    layers = nn.SequentialCell([])
    for _ in range(n):
        layers.append(nn.SequentialCell(
            SEBlock(block_channel, r),
            conv3x3_bn_relu(block_channel, block_channel)
        ))

    return layers

# 网络总体结构
class FreeNet(nn.Module):
    def __init__(self, config):
        super(FreeNet, self).__init__()
        self.config = config
        r = int(16 * self.config['reduction_ratio'])
        block1_channels = int(self.config['block_channels'][0] * self.config['reduction_ratio'] / r) * r
        block2_channels = int(self.config['block_channels'][1] * self.config['reduction_ratio'] / r) * r
        block3_channels = int(self.config['block_channels'][2] * self.config['reduction_ratio'] / r) * r
        block4_channels = int(self.config['block_channels'][3] * self.config['reduction_ratio'] / r) * r

        self.feature_ops_1 = nn.SequentialCell(
            conv3x3_bn_relu(self.config['in_channels'], block1_channels),
            repeat_block(block1_channels, r, self.config['num_blocks'][0]),
        )
        self.feature_ops_2 = nn.SequentialCell(
            conv3x3_bn_relu(block1_channels, block2_channels),
            repeat_block(block2_channels, r, self.config['num_blocks'][1]),
        )
        self.feature_ops_3 = nn.SequentialCell(
            conv3x3_bn_relu(block2_channels, block3_channels),
            repeat_block(block3_channels, r, self.config['num_blocks'][2]),
        )
        self.feature_ops_4 = nn.SequentialCell(
            conv3x3_bn_relu(block3_channels, block4_channels),
            repeat_block(block4_channels, r, self.config['num_blocks'][3]),
        )

        self.identity = ops.Identity()

        inner_dim = int(self.config['inner_dim'] * self.config['reduction_ratio'])
        self.reduce_1x1convs = nn.CellList([
            nn.Conv2d(block1_channels, inner_dim, kernel_size=1, weight_init='he_uniform', has_bias=True),
            nn.Conv2d(block2_channels, inner_dim, kernel_size=1, weight_init='he_uniform', has_bias=True),
            nn.Conv2d(block3_channels, inner_dim, kernel_size=1, weight_init='he_uniform', has_bias=True),
            nn.Conv2d(block4_channels, inner_dim, kernel_size=1, weight_init='he_uniform', has_bias=True),
        ])
        self.fuse_3x3convs = nn.CellList([
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, stride=1, pad_mode='same', weight_init='he_uniform', has_bias=True),
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, stride=1, pad_mode='same', weight_init='he_uniform', has_bias=True),
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, stride=1, pad_mode='same', weight_init='he_uniform', has_bias=True),
            nn.Conv2d(inner_dim, inner_dim, kernel_size=3, stride=1, pad_mode='same', weight_init='he_uniform', has_bias=True),
        ])
        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config['num_classes'], kernel_size=1, weight_init='he_uniform', has_bias=True)
    
    # 跳跃连接
    def top_down(self, top, lateral):
        return lateral + top

    def forward(self, x):
        feat_list = []

        x1 = self.feature_ops_1(x)
        feat_list.append(x1)
        x2 = self.feature_ops_2(x1)
        feat_list.append(x2)
        x3 = self.feature_ops_3(x2)
        feat_list.append(x3)
        x4 = self.feature_ops_4(x3)
        feat_list.append(x4)

        inner_feat_list = []
        for i, feat in enumerate(feat_list):
            inner_feat = self.reduce_1x1convs[i](feat)
            inner_feat_list.append(inner_feat)

        inner_feat_list_new = []
        for i in range(4):
            inner_feat_list_new.append(self.identity(inner_feat_list[3 - i]))
        inner_feat_list = inner_feat_list_new

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(inner_feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i + 1])
            out = self.fuse_3x3convs[i](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)
        return logit
