
from luojianet_ms import ops, nn
from luojianet_ms import Tensor
from luojianet_ms import dtype as mstype
import numpy as np
from luojianet_ms.ops import functional as F


class Conv2DReLuBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv2DReLuBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, has_bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DeConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DeConvBnReLU, self).__init__()
        self.conv = nn.Conv2dTranspose(in_channels, out_channels, kernel_size,
                                       stride=stride, has_bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv3DReLuBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(Conv3DReLuBN, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, has_bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DeConv3DReLuBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(DeConv3DReLuBN, self).__init__()
        self.conv = nn.Conv3dTranspose(in_channels, out_channels, kernel_size,
                                       stride=stride, has_bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, base_channels=32):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv2DReLuBN(base_channels, base_channels, 3)
        self.conv2 = Conv2DReLuBN(base_channels, base_channels, 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)

        return x + x2


class FeatureExtractor(nn.Module):
    def __init__(self, base_channels=32):
        super(FeatureExtractor, self).__init__()

        self.conv1 = Conv2DReLuBN(3, base_channels, 5, 2)
        self.res_conv1 = ResidualBlock(base_channels)
        self.res_conv2 = ResidualBlock(base_channels)
        self.res_conv3 = ResidualBlock(base_channels)
        self.res_conv4 = ResidualBlock(base_channels)
        self.res_conv5 = ResidualBlock(base_channels)
        self.res_conv6 = ResidualBlock(base_channels)
        self.res_conv7 = ResidualBlock(base_channels)
        self.res_conv8 = ResidualBlock(base_channels)
        self.conv18 = nn.Conv2d(base_channels, base_channels, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res_conv1(x)
        x = self.res_conv2(x)
        x = self.res_conv3(x)
        x = self.res_conv4(x)
        x = self.res_conv5(x)
        x = self.res_conv6(x)
        x = self.res_conv7(x)
        x = self.res_conv8(x)
        x = self.conv18(x)

        return x


class ConstructCostVolume(nn.Module):
    def __init__(self, max_disp):
        super(ConstructCostVolume, self).__init__()
        self.max_disp = max_disp
        # self.dsize = max_disp
        self.dsize = int(self.max_disp/2 - 1)
        self.concat_axis0 = ops.operations.Concat(axis=0)
        self.concat_axis1 = ops.operations.Concat(axis=1)
        # self.concat_axis2 = ops.operations.Concat(axis=2)
        self.stack_op = ops.Stack(axis=2)

    def forward(self, left_feature, right_feature):
        cost_volume = []
        elw = self.concat_axis1([left_feature, right_feature])
        # elw = (left_feature* right_feature)

        cost_volume.append(elw)

        for d in range(self.dsize):
            pad_op = ops.Pad(((0, 0), (0, 0), (0, 0), (d+1, 0)))
            pad_right_feature = pad_op(right_feature[:, :, :, :-1-d])
            elw = self.concat_axis1([left_feature, pad_right_feature])

            cost_volume.append(elw)

        cost_volume = self.stack_op(cost_volume)

        return cost_volume


class Regularization(nn.Module):
    def __init__(self, base_channels=32):
        super(Regularization, self).__init__()
        self.conv19 = Conv3DReLuBN(base_channels * 2, base_channels, 3)
        self.conv20 = Conv3DReLuBN(base_channels, base_channels, 3)
        self.conv21 = Conv3DReLuBN(base_channels * 2, base_channels * 2, 3, 2)
        self.conv22 = Conv3DReLuBN(base_channels * 2, base_channels * 2, 3)
        self.conv23 = Conv3DReLuBN(base_channels * 2, base_channels * 2, 3)
        self.conv24 = Conv3DReLuBN(base_channels * 2, base_channels * 2, 3, 2)
        self.conv25 = Conv3DReLuBN(base_channels * 2, base_channels * 2, 3)
        self.conv26 = Conv3DReLuBN(base_channels * 2, base_channels * 2, 3)
        self.conv27 = Conv3DReLuBN(base_channels * 2, base_channels * 2, 3, 2)
        self.conv28 = Conv3DReLuBN(base_channels * 2, base_channels * 2, 3)
        self.conv29 = Conv3DReLuBN(base_channels * 2, base_channels * 2, 3)
        self.conv30 = Conv3DReLuBN(base_channels * 2, base_channels * 4, 3, 2)
        self.conv31 = Conv3DReLuBN(base_channels * 4, base_channels * 4, 3)
        self.conv32 = Conv3DReLuBN(base_channels * 4, base_channels * 4, 3)
        self.conv33 = DeConv3DReLuBN(base_channels * 4, base_channels * 2, 3, 2)
        self.conv34 = DeConv3DReLuBN(base_channels * 2, base_channels * 2, 3, 2)
        self.conv35 = DeConv3DReLuBN(base_channels * 2, base_channels * 2, 3, 2)
        self.conv36 = DeConv3DReLuBN(base_channels * 2, base_channels, 3, 2)
        self.conv37 = nn.Conv3dTranspose(base_channels, 1, 3, 2)

    def forward(self, x):
        x20 = self.conv20(self.conv19(x))
        x21 = self.conv21(x)
        x23 = self.conv23(self.conv22(x21))
        x24 = self.conv24(x21)
        x26 = self.conv26(self.conv25(x24))
        x27 = self.conv27(x24)
        x29 = self.conv29(self.conv28(x27))
        x30 = self.conv30(x27)
        x32 = self.conv32(self.conv31(x30))
        x33 = self.conv33(x32) + x29
        x34 = self.conv34(x33) + x26
        x35 = self.conv35(x34) + x23
        x36 = self.conv36(x35) + x20
        x37 = self.conv37(x36).squeeze(1)

        return x37


class GCNet(nn.Module):
    """
    An implementation of the GCNet (Geometry and Context Network)
    using LuoJiaNET. For the details of the network, please refer
    to the paper:
    Kendall A, Martirosyan H, Dasgupta S, et al.
    End-to-en learning of geometry and context for deep stereo regression[C]
    //Proceedings of the IEEE international conference on
    computer vision. 2017: 66-75.
    """
    def __init__(self, max_disp, base_channels=32):
        super(GCNet, self).__init__()
        self.max_disp = max_disp
        self.features = FeatureExtractor(base_channels)
        self.cost_volume = ConstructCostVolume(self.max_disp)
        self.regularization = Regularization()
        self.probability = ops.Softmax(axis=1)
        self.depth_range = Tensor(np.linspace(0, self.max_disp - 1, self.max_disp), dtype=mstype.float32)
        self.sum_op = ops.ReduceSum(keep_dims=True)

    def forward(self, images):
        left_img = images[:, 0, :, :]
        right_img = images[:, 1, :, :]

        # Feature Extraction
        left_feature = self.features(left_img)
        right_feature = self.features(right_img)

        # Construct Cost Volume
        cost_volume = self.cost_volume(left_feature, right_feature)  # [B, 2C, d, H, W]

        # Regularization
        regularized_cost_volume = self.regularization(cost_volume)

        # Regression
        probability_volume = self.probability(regularized_cost_volume)
        depth_range = self.depth_range.reshape(1, self.max_disp, 1, 1)
        disparity = self.sum_op(probability_volume * depth_range, 1)

        return disparity.squeeze(1)


if __name__ == "__main__":
    import os
    import luojianet_ms.context as context
    from luojianet_ms.common.initializer import Zero

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    img = Tensor(dtype=mstype.float32, shape=(1, 2, 3, 256, 512), init=Zero())
    model = GCNet()

    output = model(img)

    print(output.shape)
