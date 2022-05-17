"""Different custom layers"""

import luojianet_ms as luojia
from luojianet_ms import nn, ops


OPS = {
    "conv1x1": lambda C_in, C_out, stride, affine, repeats=1: nn.SequentialCell(
        conv1x1(C_in, C_out, stride=stride),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(),
    ),
    "conv3x3": lambda C_in, C_out, stride, affine, repeats=1: nn.SequentialCell(
        conv3x3(C_in, C_out, stride=stride),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(),
    ),
    "conv3x3_dil3": lambda C_in, C_out, stride, affine, repeats=1: nn.SequentialCell(
        conv3x3(C_in, C_out, stride=stride, dilation=3),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(),
    ),
    "conv3x3_dil12": lambda C_in, C_out, stride, affine, repeats=1: nn.SequentialCell(
        conv3x3(C_in, C_out, stride=stride, dilation=12),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(),
    ),
    "sep_conv_3x3": lambda C_in, C_out, stride, affine, repeats=1: SepConv(
        C_in, C_out, 3, stride, 1, affine=affine, repeats=repeats
    ),
    "sep_conv_5x5": lambda C_in, C_out, stride, affine, repeats=1: SepConv(
        C_in, C_out, 5, stride, 2, affine=affine, repeats=repeats
    ),
    "sep_conv_7x7": lambda C_in, C_out, stride, affine, repeats=1: SepConv(
        C_in, C_out, 7, stride, 3, affine=affine, repeats=repeats
    ),
    "avg_pool_3x3": lambda C_in, C_out, stride, affine, repeats=1: Pool(
        C_in, C_out, stride, repeats, ksize=3, mode="avg"
    ),
    "max_pool_3x3": lambda C_in, C_out, stride, affine, repeats=1: Pool(
        C_in, C_out, stride, repeats, ksize=3, mode="max"
    ),
    "global_average_pool": lambda C_in, C_out, stride, affine, repeats=1: GAPConv1x1(
        C_in, C_out
    ),
    "sobel_operator": lambda C_in, C_out, stride, affine, repeats=1: Sobel(
        C_in, C_out
    ),
    "laplacian_operator": lambda C_in, C_out, stride, affine, repeats=1: Laplacian(
        C_in, C_out
    )
    # 'edge_operator':  lambda C_in, C_out, stride, affine, repeats=1: Edge(
    #     C_in, C_out
    # ),
    # "gaussian_operator": lambda C_in, C_out, stride, affine, repeats=1: Gaussian(
    #     C_in, C_out
    # ),
    # "median_operator": lambda C_in, C_out, stride, affine, repeats=1: Median(
    #         C_in, C_out
    # )
}

OPS_mini = {
    "sep_conv_3x3": lambda C_in, C_out, stride, affine, repeats=1: SepConv(
        C_in, C_out, 3, stride, 1, affine=affine, repeats=repeats
    ),
    "sep_conv_5x5": lambda C_in, C_out, stride, affine, repeats=1: SepConv(
        C_in, C_out, 5, stride, 2, affine=affine, repeats=repeats
    ),
    "sep_conv_7x7": lambda C_in, C_out, stride, affine, repeats=1: SepConv(
        C_in, C_out, 7, stride, 3, affine=affine, repeats=repeats
    )
}

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=0,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, padding=0
    )


def conv_bn(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.SequentialCell(
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(C_out, affine=affine),
    )


def conv_bn_relu(C_in, C_out, kernel_size, stride, padding, affine=True):
    return nn.SequentialCell(
        nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(C_out, affine=affine),
        nn.ReLU(),
    )


class Pool(nn.Module):
    """Conv1x1 followed by pooling"""

    def __init__(self, C_in, C_out, stride, repeats, ksize, mode):
        super(Pool, self).__init__()
        self.conv1x1 = conv_bn(C_in, C_out, 1, 1, 0)
        if mode == "avg":
            self.pool = nn.AvgPool2d(
                ksize, stride=stride, pad_mode="same"
            )
        elif mode == "max":
            self.pool = nn.MaxPool2d(ksize, stride=stride, pad_mode="same")
        else:
            raise ValueError("Unknown pooling method {}".format(mode))

    def call(self, x):
        x = self.conv1x1(x)
        return self.pool(x)


class GAPConv1x1(nn.Module):
    """Global Average Pooling + conv1x1"""

    def __init__(self, C_in, C_out):
        super(GAPConv1x1, self).__init__()
        self.conv1x1 = conv_bn_relu(C_in, C_out, 1, stride=1, padding=0)

    def call(self, x):
        size = x.shape[2:]
        out = x.mean(2).mean(2)
        out = ops.ExpandDims()(out, 2)
        out = ops.ExpandDims()(out, 3)
        out = self.conv1x1(out)
        out = nn.ResizeBilinear()(out, size, align_corners=True)
        return out

class SepConv(nn.Module):
    """Separable convolution"""

    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        stride,
        padding,
        dilation=1,
        affine=True,
        repeats=1,
    ):
        super(SepConv, self).__init__()

        def basic_op(C_in, C_out):
            return nn.SequentialCell(
                nn.Conv2d(
                    C_in,
                    C_in,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    pad_mode='pad',
                    dilation=dilation,
                    group=C_in,
                ),
                nn.Conv2d(C_in, C_out, kernel_size=1, padding=0),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(),
            )

        self.op = nn.SequentialCell()
        for idx in range(repeats):
            if idx > 0:
                C_in = C_out
            self.op.append(basic_op(C_in, C_out))

    def call(self, x):
        return self.op(x)

class Edge(nn.Module):
    def __init__(self, C_in, C_out):
        super(Edge, self).__init__()
        self.out_channels = C_out

        self.filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        G = luojia.Tensor([[1.0, 2.0, -1.0], [2.0, 0.0, -2.0], [1.0, -2.0, -1.0]])
        G = G.unsqueeze(0).unsqueeze(0)
        self.filter.weight = luojia.Parameter(G)

    def call(self, img):
        b, c, w, h = img.shape
        x = img.mean(1, True)
        x = self.filter(x)
        x.repeat(1, self.out_channels, 1, 1)
        return x

class Sobel(nn.Module):
    def __init__(self, C_in, C_out):
        super(Sobel, self).__init__()
        self.out_channels = C_out
        self.conv1x1 = conv_bn_relu(C_in, C_out, 1, 1, 0) # TODO:遥感算子

        self.filter = sobel
        # Gx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
        # Gy = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
        # G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        # G = G.unsqueeze(1)
        # self.filter.weight = nn.Parameter(G, requires_grad=False)

    def call(self, img):
        x = img.mean(1, True)
        x = self.filter(x, (3, 3))
        x.repeat(1, self.out_channels, 1, 1)
        return x

class Laplacian(nn.Module):
    def __init__(self, C_in, C_out):
        super(Laplacian, self).__init__()
        self.out_channels = C_out
        self.conv1x1 = conv_bn_relu(C_in, C_out, 1, 1, 0)

        self.filter = laplacian

    def call(self, img):
        x = img.mean(1, True)
        x = self.filter(x, 3, normalized=False)
        x.repeat(1, self.out_channels, 1, 1)
        return x

class Gaussian(nn.Module):
    def __init__(self, C_in, C_out):
        super(Gaussian, self).__init__()
        self.out_channels = C_out
        self.conv1x1 = conv_bn_relu(C_in, C_out, 1, 1, 0)

        self.filter = gaussian_blur2d

    def call(self, img):
        img = self.conv1x1(img)
        denoise_img = self.filter(img, (3, 3), (1.5, 1.5))
        return denoise_img

class Median(nn.Module):
    def __init__(self, C_in, C_out):
        super(Median, self).__init__()
        self.out_channels = C_out
        self.conv1x1 = conv_bn_relu(C_in, C_out, 1, 1, 0)

        self.filter = median_blur

    def call(self, img):
        img = self.conv1x1(img)
        denoise_img = self.filter(img, (3, 3))
        return denoise_img

class Denoising(nn.Module):
    def __init__(self, C_in, C_out):
        super(Denoising, self).__init__()
        self.out_channels = C_out

        self.filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=False)
        G = luojia.Tensor([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
        G = G.unsqueeze(0).unsqueeze(0)
        self.filter.weight = luojia.Parameter(G)

    def call(self, img):
        b, c, w, h = img.shape
        x = img.mean(1, True)
        x = self.filter(x)
        x.repeat(1, self.out_channels, 1, 1)
        return x


