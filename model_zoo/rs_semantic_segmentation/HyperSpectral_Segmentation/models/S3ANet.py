import luojianet_ms.nn as nn
import luojianet_ms as ms
from functools import reduce
import math
import sys
from luojianet_ms import ops
#sys.path.append('/home/luojianet/Hyperspectral_classification/S3ANet_LuojiaNet/S3ANet_Luojianet')
from configs.S3ANet_HH_config import config as S3ANet_HH_config


def conv3x3_gn_relu(in_channel, out_channel, num_group, stride):
    return nn.SequentialCell(
        nn.Conv2d(in_channel, out_channel, 3, stride, pad_mode='same', weight_init="Uniform", ),
        nn.GroupNorm(num_group, out_channel),
        nn.ReLU(),
    )


def conv3x3(in_channel, out_channel, stride=1):
    return nn.SequentialCell(
        nn.Conv2d(in_channel, out_channel, 3, stride, pad_mode='same', weight_init="Uniform",),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(),
    )


class SpectralAttention(nn.Module):
    def __init__(self, in_planes, pool_outsize, ratio=16):
        super(SpectralAttention, self).__init__()
        self.avg_pool = ms.nn.AvgPool2d(kernel_size=pool_outsize)
        self.max_pool = ms.nn.MaxPool2d(kernel_size=pool_outsize)
        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, weight_init="Uniform")
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, weight_init="Uniform")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        x = x * self.sigmoid(out)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, pad_mode='pad', weight_init="Uniform",)
        self.sigmoid = nn.Sigmoid()
        self.concat = ms.ops.Concat(axis=1)


    def forward(self, x):
        avg_out = ms.ops.ReduceMean(keep_dims=True)(x, 1)
        max_out, _ = ms.ops.ArgMaxWithValue(axis=1, keep_dims=True)(x)
        max_out = ms.Tensor(max_out, ms.dtype.float32)
        x_sa = self.concat((avg_out, max_out))
        x_sa = self.conv1(x_sa)
        x = x * self.sigmoid(x_sa)
        return x


class ScaleAttention(nn.Module):
    def __init__(self, pool_size, in_channels, out_channels, stride=1, M=5, r=16, L=32):
        super(ScaleAttention, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.CellList()
        dilation_rate = [1, 3, 6, 9, 12]
        for i in range(M):
            self.conv.append(
                nn.SequentialCell(
                    nn.Conv2d(in_channels, out_channels, 3, stride, padding=dilation_rate[i], dilation=dilation_rate[i], pad_mode='pad', weight_init="Uniform",),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                )
            )

        self.global_pool = ms.nn.AvgPool2d(kernel_size=pool_size)
        self.fc1 = nn.SequentialCell(
            nn.Conv2d(out_channels, d, 1, weight_init="Uniform",),
            nn.ReLU()
        )
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, weight_init="Uniform",)
        self.softmax = nn.Softmax(axis=1)
        self.reshape = ms.ops.Reshape()

    def forward(self, image):
        batch_size = image.shape[0]
        output = []
        # the part of split
        for i, conv in enumerate(self.conv):
            output.append(conv(image))
        U = reduce(lambda x, y: x + y, output)
        s = self.global_pool(U)
        z = self.fc1(s)
        a_b = self.fc2(z)
        a_b = self.reshape(a_b, (batch_size, self.M, self.out_channels, -1))
        a_b = self.softmax(a_b)
        # the part of selection
        # a_b = list(a_b.chunk(self.M, axis=1))  # split to a and b
        split = ms.ops.Split(axis=1, output_num=self.M)
        a_b = list(split(a_b))
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1), a_b))
        V = list(map(lambda x, y: x * y, output, a_b))
        V = reduce(lambda x, y: x + y, V)
        return V


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin

            cos(theta + m)
        """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.reshape = ms.ops.Reshape()
        self.transpose = ms.ops.Transpose()
        self.normalize = ms.ops.L2Normalize(1,1e-12)
        self.sqrt =  ms.ops.Sqrt()
        self.pow =  ms.ops.Pow()
        self.oneslike = ms.ops.OnesLike()
        self.zeros = ms.ops.Zeros()


    def forward(self, input_data, label_data, weight):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        input = self.reshape(input_data, (input_data.shape[1], input_data.shape[2] * input_data.shape[3]))
        input = self.transpose(input, (1, 0))
        label = self.reshape(label_data, (input_data.shape[2] * input_data.shape[3], 1))
        cosine = ms.ops.matmul(self.normalize(input), self.normalize(weight).transpose())

        min_value = ms.Tensor(0, ms.float32)
        max_value = ms.Tensor(1, ms.float32)
        sine = self.sqrt(ms.ops.clip_by_value((1.0 - self.pow(cosine, 2)), min_value, max_value))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = ms.numpy.where(cosine > 0, phi, cosine)
        else:
            phi = ms.numpy.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        label = ms.numpy.where(label < 0, 0, label)

        self.one_hot = nn.OneHot(depth=cosine.shape[1], axis=-1)
        one_hot = self.one_hot(label[:,0])
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output



class S3ANET(nn.Module):
    def __init__(self, config,  training=True):
        super(S3ANET, self).__init__()
        self.config = config
        self.training = training

        # self.conv1 = conv3x3_gn_relu(self.config.in_channels, 64, 16,stride=1)
        self.conv1 = conv3x3(self.config['in_channels'], 64, stride=1, )

        self.spe1 = SpectralAttention(64, pool_outsize=config['pool_outsize1'])
        self.conv2 = conv3x3(64, 96, stride=1, )

        self.spe2 = SpectralAttention(96, pool_outsize=config['pool_outsize1'])
        self.conv3 = conv3x3(96, 128, stride=2)

        self.spe3 = SpectralAttention(128, pool_outsize=config['pool_outsize2'])
        self.conv4 = conv3x3(128, 192, stride=2, )

        self.spe4 = SpectralAttention(192, pool_outsize=config['pool_outsize3'])
        self.conv5 = conv3x3(192, 256, stride=2, )

        self.conv6 = conv3x3(256, 256, stride=1, )

        self.spa = SpatialAttention()

        self.sca = ScaleAttention(pool_size=config['pool_outsize4'], in_channels=256, out_channels=256, stride=1)

        self.reduce_conv1 = nn.Conv2d(96, 96, 1, weight_init="Uniform",)
        self.reduce_conv2 = nn.Conv2d(128, 128, 1, weight_init="Uniform",)
        self.reduce_conv3 = nn.Conv2d(192, 192, 1, weight_init="Uniform",)
        self.reduce_conv4 = nn.Conv2d(256, 256, 1, weight_init="Uniform",)

        self.conv_r1 = nn.Conv2d(512, 192, 1, weight_init="Uniform",)
        self.conv_r1_1 = nn.Conv2d(512, 256, 1, weight_init="Uniform",)
        self.conv_r2 = nn.Conv2d(384, 128, 1, weight_init="Uniform",)
        self.conv_r2_1 = nn.Conv2d(384, 192, 1, weight_init="Uniform",)
        self.conv_r3 = nn.Conv2d(256, 96, 1, weight_init="Uniform",)
        self.conv_r3_1 = nn.Conv2d(256, 128, 1, weight_init="Uniform",)
        self.conv_r4 = nn.Conv2d(192, 128, 1, weight_init="Uniform",)
        self.conv_r4_1 = nn.Conv2d(192, 96, 1, weight_init="Uniform",)

        self.cls_pred_conv = nn.Conv2d(128, self.config['num_classes'], 1, weight_init='xavier_uniform')

        self.metric_fc = ArcMarginProduct(128, self.config['num_classes'], s=30, m=0.5, easy_margin=False)

        self.upsample = nn.ResizeBilinear()
        self.concat = ms.ops.Concat(axis=1)
        self.transpose = ms.ops.Transpose()
        self.reshape = ms.ops.Reshape()

    def forward(self, x, y=None, w=None, ):
        # encoder
        x = self.conv1(x)

        block1_out = self.spe1(x)
        block1_out = self.conv2(block1_out)

        block2_out = self.spe2(block1_out)
        block2_out = self.conv3(block2_out)

        block3_out = self.spe3(block2_out)
        block3_out = self.conv4(block3_out)

        block4_out = self.spe4(block3_out)
        block4_out = self.conv5(block4_out)

        skconv_feat = self.sca(block4_out)

        # decoder

        UP1_out = self.conv6(skconv_feat)
        a = self.concat((UP1_out, block4_out))
        UP1_out = self.spa(a)
        UP1_out = self.conv_r1_1(UP1_out)
        UP1_out = self.concat((UP1_out, block4_out))
        UP1_out = self.conv_r1(UP1_out)
        UP1_out = self.upsample(UP1_out, scale_factor=2)

        a = self.concat((UP1_out, block3_out))
        UP2_out = self.spa(a)
        UP2_out = self.conv_r2_1(UP2_out)
        UP2_out = self.concat((UP2_out, UP1_out))
        UP2_out = self.conv_r2(UP2_out)
        UP2_out = self.upsample(UP2_out, scale_factor=2)

        a = self.concat((UP2_out, block2_out))
        UP3_out = self.spa(a)
        UP3_out = self.conv_r3_1(UP3_out)
        UP3_out = self.concat((UP3_out, UP2_out))
        UP3_out = self.conv_r3(UP3_out)
        UP3_out = self.upsample(UP3_out, scale_factor=2)

        a = self.concat((UP3_out, block1_out))
        UP4_out = self.spa(a)
        UP4_out = self.conv_r4_1(UP4_out)
        UP4_out = self.concat((UP4_out, UP3_out))
        final_feat = self.conv_r4(UP4_out)

        if self.training:
            logit = self.metric_fc(final_feat, y, self.cls_pred_conv.weight.squeeze())
            logit = self.reshape(logit, (final_feat.shape[0], final_feat.shape[2], final_feat.shape[3], self.config['num_classes']))
            logit = self.transpose(logit, (0, 3, 1, 2))
            return logit

        else:
            logit = self.cls_pred_conv(final_feat)
            return logit
