
from luojianet_ms import nn, ops
from luojianet_ms import dtype as mstype
from src.module import *
from src.homography_warping import HomoWarp
from luojianet_ms.ops import operations as P


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.conv0 = Conv2DReLuBN(3, 8, 3, 1)
        self.conv1 = Conv2DReLuBN(8, 8, 3, 1)

        self.conv2 = Conv2DReLuBN(8, 16, 5, 2)
        self.conv3 = Conv2DReLuBN(16, 16, 3, 1)
        self.conv4 = Conv2DReLuBN(16, 16, 3, 1)

        self.conv5 = Conv2DReLuBN(16, 32, 5, 2)
        self.conv6 = Conv2DReLuBN(32, 32, 3, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1)

    def call(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))

        return x


class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.SequentialCell(
            nn.Conv3dTranspose(64, 32, kernel_size=3, stride=2, has_bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU())

        self.conv9 = nn.SequentialCell(
            nn.Conv3dTranspose(32, 16, kernel_size=3, stride=2, has_bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU())

        self.conv11 = nn.SequentialCell(
            nn.Conv3dTranspose(16, 8, kernel_size=3, stride=2, has_bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU())

        self.prob = nn.Conv3d(8, 1, 3, stride=1)

    def call(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

        self.cat_axis_1 = ops.Concat(axis=1)
        self.resize = nn.ResizeBilinear()

    def call(self, img, depth_init, depth_values):
        """ refine depth image with the image """
        # normalization
        depth_shape = depth_init.shape
        depth_start = depth_values[:, 0].view(depth_shape[0], 1, depth_shape[2], depth_shape[3])
        depth_end = depth_values[:, -1].view(depth_shape[0], 1, depth_shape[2], depth_shape[3])
        depth_scale = depth_end - depth_start

        # normalize depth map (to 0~1)
        init_norm_depth_map = (depth_init - depth_start) / depth_scale

        # resize normalized image to the same size of depth image
        resized_image = self.resize(img, (depth_shape[2], depth_shape[3]))

        concat = self.cat_axis_1((resized_image, init_norm_depth_map))
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = init_norm_depth_map + depth_residual

        depth_refined = depth_refined * depth_scale + depth_start

        return depth_refined


def depth_regression(p, depth_values):
    """
    p: probability volume (B, D, H, W)
    depth_values: discrete depth values (B, D, H, W) or (D)
    inverse: depth_values is inverse depth or not
    """
    if depth_values.ndim <= 2:
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    cumsum = P.ReduceSum(True)
    depth = cumsum(p * depth_values, 1)

    return depth


class MVSNet(nn.Module):
    def __init__(self, height, width, refine=True):
        super(MVSNet, self).__init__()
        self.refine = refine
        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()
        if self.refine:
            self.refine_network = RefineNet()
        self.expand_dims = ops.ExpandDims()
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.homo_warp = HomoWarp(int(height/4), int(width/4))
        self.softmax = nn.Softmax(axis=1)
        self.pow = ops.Pow()

    def call(self, imgs, proj_matrices, depth_values):
        # step 1. feature extraction
        D = depth_values.shape[1]
        B, V, C, H, W = imgs.shape

        imgs = imgs.reshape(B * V, C, H, W)

        features = self.feature(imgs)
        features = features.view(B, V, *features.shape[1:])

        imgs = imgs.reshape(B, V, C, H, W)

        # step 2. differentiable homograph, build cost volume
        ref_feat, src_feats = features[:, 0], features[:, 1:]

        ref_volume = self.expand_dims(features[:, 0], 2)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2

        # For training...
        for i in range(src_feats.shape[1]):
            src_feat = src_feats[:, i]
            proj_mat = proj_matrices[:, i+1]

            warped_src = self.homo_warp(src_feat, proj_mat, depth_values)

            volume_sum += warped_src
            volume_sq_sum += self.pow(warped_src, 2)

        volume_variance = volume_sq_sum / V - self.pow(volume_sum / V, 2.0)

        # step 3. cost volume regularization
        regularized_volume = self.cost_regularization(volume_variance)

        cost_reg = regularized_volume.squeeze(1)

        prob_volume = self.softmax(cost_reg)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        # step 4. depth map refinement
        if not self.refine:
            return depth
        else:
            refined_depth = self.refine_network(imgs[:, 0], depth, depth_values)

            return refined_depth


if __name__ == "__main__":
    net = MVSNet()
