import luojianet_ms.nn as nn
import luojianet_ms as ms


def conv3x3_gn_relu(in_channel, out_channel, num_group):
    return nn.SequentialCell(
        nn.Conv2d(in_channel, out_channel, 3, 1, pad_mode='same', weight_init="Uniform", has_bias=False),
        nn.GroupNorm(num_group, out_channel),
        nn.ReLU()
    )


def downsample2x(in_channel, out_channel):
    return nn.SequentialCell(
        nn.Conv2d(in_channel, out_channel, 3, 2, pad_mode='pad', padding=1, weight_init="Uniform", has_bias=False),
        nn.ReLU()
    )


class SEBlock(nn.Module):
    def __init__(self, pool_outsize, in_channels, reduction_ratio):
        super(SEBlock, self).__init__()
        self.gap = nn.AvgPool2d(kernel_size=pool_outsize)
        self.seq = nn.SequentialCell(
            nn.Dense(in_channels, in_channels // reduction_ratio, weight_init="Uniform", has_bias=False),
            nn.ReLU(),
            nn.Dense(in_channels // reduction_ratio, in_channels, weight_init="Uniform", has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        v = self.gap(x)
        score = self.seq(v.view(v.shape[0], v.shape[1]))
        y = x * score.view(score.shape[0], score.shape[1], 1, 1)
        return y

class Idendity(nn.Module):
    def __init__(self, ):
        super(Idendity, self).__init__()

    def forward(self, x):
        return x


def repeat_block(pool_outsize, block_channel, r, n):
    layers = [
        nn.SequentialCell(
            SEBlock(pool_outsize, block_channel, r),
            conv3x3_gn_relu(block_channel, block_channel, r)
        )
        for _ in range(n)
    ]
    return nn.SequentialCell(layers)


class FreeNet(nn.Module):
    def __init__(self, config):
        super(FreeNet, self).__init__()
        self.config = config
        r = int(16 * self.config['reduction_ratio'])
        block1_channels = int(self.config['block_channels'][0] * self.config['reduction_ratio'] / r) * r
        block2_channels = int(self.config['block_channels'][1] * self.config['reduction_ratio'] / r) * r
        block3_channels = int(self.config['block_channels'][2] * self.config['reduction_ratio'] / r) * r
        block4_channels = int(self.config['block_channels'][3] * self.config['reduction_ratio'] / r) * r

        self.feature_ops = nn.CellList([
            conv3x3_gn_relu(self.config['in_channels'], block1_channels, r),

            repeat_block(config['pool_outsize1'], block1_channels, r, self.config['num_blocks'][0]),
            Idendity(),
            downsample2x(block1_channels, block2_channels),

            repeat_block(config['pool_outsize2'], block2_channels, r, self.config['num_blocks'][1]),
            Idendity(),
            downsample2x(block2_channels, block3_channels),

            repeat_block(config['pool_outsize3'], block3_channels, r, self.config['num_blocks'][2]),
            Idendity(),
            downsample2x(block3_channels, block4_channels),

            repeat_block(config['pool_outsize4'], block4_channels, r, self.config['num_blocks'][3]),
            Idendity(),
        ])

        inner_dim = int(self.config['inner_dim'] * self.config['reduction_ratio'])

        self.reduce_1x1convs = nn.CellList([
            nn.Conv2d(block1_channels, inner_dim, 1, weight_init="Uniform", has_bias=False),
            nn.Conv2d(block2_channels, inner_dim, 1, weight_init="Uniform", has_bias=False),
            nn.Conv2d(block3_channels, inner_dim, 1, weight_init="Uniform", has_bias=False),
            nn.Conv2d(block4_channels, inner_dim, 1, weight_init="Uniform", has_bias=False),
        ])

        self.fuse_3x3convs = nn.CellList([
            nn.Conv2d(inner_dim, inner_dim, 3, 1, pad_mode='same', weight_init="Uniform", has_bias=False),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, pad_mode='same', weight_init="Uniform", has_bias=False),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, pad_mode='same', weight_init="Uniform", has_bias=False),
            nn.Conv2d(inner_dim, inner_dim, 3, 1, pad_mode='same', weight_init="Uniform", has_bias=False),
        ])

        self.cls_pred_conv = nn.Conv2d(inner_dim, self.config['num_classes'], 1, weight_init="Uniform", has_bias=False)

    def top_down(self, top, lateral):
        function = nn.ResizeBilinear()
        top2x = function(top, scale_factor=2)
        return lateral + top2x

    # def top_down(self, top, lateral):
    #     top2x = F.interpolate(top, scale_factor=2.0, mode='nearest')
    #     return lateral + top2x

    def forward(self, x, y=None, w=None, **kwargs):
        feat_list = []
        for op in self.feature_ops:
            x = op(x)
            if isinstance(op, Idendity):
                feat_list.append(x)

        inner_feat_list = [self.reduce_1x1convs[i](feat) for i, feat in enumerate(feat_list)]
        inner_feat_list.reverse()

        out_feat_list = [self.fuse_3x3convs[0](inner_feat_list[0])]
        for i in range(len(inner_feat_list) - 1):
            inner = self.top_down(out_feat_list[i], inner_feat_list[i + 1])
            out = self.fuse_3x3convs[i](inner)
            out_feat_list.append(out)

        final_feat = out_feat_list[-1]

        logit = self.cls_pred_conv(final_feat)

        return logit
