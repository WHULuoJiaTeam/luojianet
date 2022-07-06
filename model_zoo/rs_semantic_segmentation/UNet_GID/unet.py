import luojianet_ms
import luojianet_ms.nn as ljnn
import luojianet_ms.ops.operations as P

class ResizeBilinear_m(ljnn.Module):
    def __init__(self, scale_factor=None, size=None):
        super(ResizeBilinear_m, self).__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.rb = ljnn.ResizeBilinear()
    def forward(self, x):
        if self.scale_factor is not None:
            x = self.rb(x, scale_factor=self.scale_factor)
        if self.size is not None:
            x = self.rb(x, size=(self.size, self.size))
        return x
class ResizeNearestNeighbor_m(ljnn.Module):
    def __init__(self, scale_factor=None, size=None):
        super(ResizeNearestNeighbor_m, self).__init__()
        self.scale_factor = scale_factor
        self.size = size
    def forward(self, x):
        if self.scale_factor is not None:
            x = P.ResizeNearestNeighbor((x.shape[2] * self.scale_factor, x.shape[2] * self.scale_factor))(x)
        if self.size is not None:
            x = P.ResizeNearestNeighbor((self.size, self.size))(x)
        return x
###//////////////////conv2d//////////////////////###
# TODO Pytorch
# class conv2D(ljnn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='SAME', dilation=1):
#         super(conv2D, self).__init__()
#         if padding == 'SAME':
#             padding = int((kernel_size - 1) / 2)
#         self.conv2d = ljnn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
#                                 stride=stride, padding=padding, has_bias=True, pad_mode='pad', dilation=dilation)
#
#     def forward(self, x):
#         x = self.conv2d(x)
#         return x

###/////////////////////////////////////////////###

###////////////////// bn  //////////////////////###

###/////////////////////////////////////////////###

###//////////////////Upmodel//////////////////////###
###/////////////////////////////////////////////###
class Upmodel(ljnn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upmodel, self).__init__()
        self.upbl = ResizeBilinear_m(scale_factor=2)
        self.conv1 = ljnn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                    stride=1, padding=1, has_bias=True, pad_mode='pad', dilation=1)

        self.conv2 = ljnn.Conv2d(in_channels=out_channels *2, out_channels=out_channels, kernel_size=3,
                    stride=1, padding=1, has_bias=True, pad_mode='pad', dilation=1)
        self.bn2 = ljnn.BatchNorm2d(num_features=out_channels)
        self.relu = ljnn.ReLU()
        self.conv3 = ljnn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                    stride=1, padding=1, has_bias=True, pad_mode='pad', dilation=1)
        self.bn3 = ljnn.BatchNorm2d(num_features=out_channels)
    def forward(self, x, y):
        x = self.upbl(x)
        x = self.conv1(x)
        x = luojianet_ms.ops.Concat(1)([x, y])
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x


###/////////////////////////////////////////////###


###//////////////////Downmodel//////////////////////###
###/////////////////////////////////////////////###
class Downmodel(ljnn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super(Downmodel, self).__init__()
        self.pool = pool
        self.conv1 = ljnn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                    stride=1, padding=1, has_bias=True, pad_mode='pad', dilation=1)
        self.bn1 = ljnn.BatchNorm2d(num_features=out_channels)
        self.relu = ljnn.ReLU()
        self.conv1_1 = ljnn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                    stride=1, padding=1, has_bias=True, pad_mode='pad', dilation=1)
        self.bn1_1 = ljnn.BatchNorm2d(num_features=out_channels)
        if pool:
            self.pool1 = ljnn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.conv1(x)                                   # ---> torch.Size([4/batchsize, 32, 512, 512])
        x = self.bn1(x)                                   # ---> torch.Size([4/batchsize, 32, 512, 512])
        x = self.relu(x)                                   # ---> torch.Size([4/batchsize, 32, 512, 512])
        x = self.conv1_1(x)                                   # ---> torch.Size([4/batchsize, 32, 512, 512])
        x = self.bn1_1(x)                                   # ---> torch.Size([4/batchsize, 32, 512, 512])
        x = self.relu(x)                                   # ---> torch.Size([4/batchsize, 32, 512, 512])
        bn1_1 = x
        if self.pool:
            x = self.pool1(x)                                   # ---> torch.Size([4/batchsize, 32, 256, 256])
            return x, bn1_1
        else:
            return bn1_1


###/////////////////////////////////////////////###


class unet(ljnn.Module):
    def __init__(self, output_class):
        super(unet, self).__init__()

        self.down1 = Downmodel(in_channels=3, out_channels=32)
        self.down2 = Downmodel(in_channels=32, out_channels=64)
        self.down3 = Downmodel(in_channels=64, out_channels=128)
        self.down4 = Downmodel(in_channels=128, out_channels=256)
        self.down5 = Downmodel(in_channels=256, out_channels=512, pool=False)

        self.up1 = Upmodel(in_channels=512, out_channels=256)
        self.up2 = Upmodel(in_channels=256, out_channels=128)
        self.up3 = Upmodel(in_channels=128, out_channels=64)
        self.up4 = Upmodel(in_channels=64, out_channels=32)

        self.conv = ljnn.Conv2d(in_channels=32, out_channels=output_class, kernel_size=1,
                    stride=1, padding=0, has_bias=True, pad_mode='pad', dilation=1)
    def forward(self, x):

        x, bn1_1 = self.down1(x)
        x, bn2_1 = self.down2(x)
        x, bn3_1 = self.down3(x)
        x, bn4_1 = self.down4(x)
        x = self.down5(x)

        x = self.up1(x, bn4_1)
        x = self.up2(x, bn3_1)
        x = self.up3(x, bn2_1)
        x = self.up4(x, bn1_1)

        x = self.conv(x)
        return x





if __name__ == '__main__':

    import os
    from luojianet_ms.common.initializer import Normal
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    input = luojianet_ms.Tensor(shape=(4, 3, 512, 512), dtype=luojianet_ms.float32, init=Normal())
    print('input:')
    print(input.shape)

    net = unet(output_class=6)
    output = net(input)

    print('output:')
    print(output.shape)
    # for op in output:
    #     print(op.shape)


