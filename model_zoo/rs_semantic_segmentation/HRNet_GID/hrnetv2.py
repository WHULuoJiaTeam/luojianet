import luojianet_ms.nn as nn
import luojianet_ms
from luojianet_ms.ops import Identity
import luojianet_ms.ops.operations as P

class Identity_m(nn.Module):
    def __init__(self):
        super(Identity_m, self).__init__()

    def forward(self, x):
        return Identity()(x)

class ResizeBilinear_m(nn.Module):
    def __init__(self, scale_factor=None, size=None):
        super(ResizeBilinear_m, self).__init__()
        self.scale_factor = scale_factor
        self.size = size
        self.rb = nn.ResizeBilinear()
    def forward(self, x):
        if self.scale_factor is not None:
            x = self.rb(x, scale_factor=self.scale_factor)
        if self.size is not None:
            x = self.rb(x, size=(self.size, self.size))
        return x

class ResizeNearestNeighbor_m(nn.Module):
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
class conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding='SAME'):
        super(conv2D, self).__init__()
        if padding == 'SAME':

            padding = int((kernel_size - 1) / 2)
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, has_bias=True, pad_mode='pad')

    def forward(self, x):
        x = self.conv2d(x)
        return x
###/////////////////////////////////////////////###
###////////////////// bn  //////////////////////###
###/////////////////////////////////////////////###
# # TODO Pytorch
# class bn(nn.Module):
#     def __init__(self, num_features):
#         super(bn, self).__init__()
#         # self.bn = nn.BatchNorm2d(num_features=num_features, eps=1e-3, momentum=0.99,)
#         self.bn = nn.BatchNorm2d(num_features=num_features)
#
#     def forward(self, x):
#         self.bn(x)
#         return x


###/////////////////////////////////////////////###

###//////////////////Bottleneck//////////////////////###
###/////////////////////////////////////////////###
# TODO Pytorch
class Bottleneck(nn.Module):
    def __init__(self, in_channels, size, downsampe=False):
        super(Bottleneck, self).__init__()
        self.downsampe = downsampe

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = conv2D(in_channels=in_channels, out_channels=size, kernel_size=1, padding=0)

        self.bn2 = nn.BatchNorm2d(size)
        self.conv2 = conv2D(in_channels=size, out_channels=size, kernel_size=3)

        self.bn3 = nn.BatchNorm2d(size)
        self.conv3 = conv2D(in_channels=size, out_channels=size*4, kernel_size=1, padding=0)

        self.bn_residual = nn.BatchNorm2d(in_channels)
        self.conv_residual = conv2D(in_channels=in_channels, out_channels=size*4, kernel_size=1, padding=0)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsampe:
            residual = self.bn_residual(x)
            residual = self.relu(residual)
            residual = self.conv_residual(residual)
        out = out + residual
        return out
###/////////////////////////////////////////////###

###//////////////////layer1//////////////////////###
###/////////////////////////////////////////////###
# TODO Pytorch
class layer1(nn.Module):
    def __init__(self, in_channels):
        super(layer1, self).__init__()
        self.bottleneck1 = Bottleneck(in_channels=in_channels, size=64, downsampe=True)
        self.bottleneck2 = Bottleneck(in_channels=64*4, size=64)
        self.bottleneck3 = Bottleneck(in_channels=64*4, size=64)
        self.bottleneck4 = Bottleneck(in_channels=64*4, size=64)
    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        return x


###/////////////////////////////////////////////###


###//////////////////transition_layer//////////////////////###
###/////////////////////////////////////////////###
# in_channels-->[256]
# out_channels-->[32, 64]
# TODO Pytorch
class transition_layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(transition_layer, self).__init__()
        self.num_in = len(in_channels)
        self.num_out = len(out_channels)
        self.modl1 = nn.CellList([])

        for i in range(self.num_out):
            if i < self.num_in:
                if in_channels[i] != out_channels[i]:
                    modl_i = nn.CellList([nn.BatchNorm2d(num_features=in_channels[i]), nn.ReLU(),
                                            conv2D(in_channels=in_channels[i], out_channels=out_channels[i],
                                                   kernel_size=3)])
                    self.modl1.append(modl_i)
                else:
                    modl_i = nn.CellList([Identity_m()])
                    self.modl1.append(modl_i)
            else:
                modl_i = nn.CellList([nn.BatchNorm2d(num_features=in_channels[-1]), nn.ReLU(),
                                        conv2D(in_channels=in_channels[-1], out_channels=out_channels[i], kernel_size=3,
                                               stride=2)])
                self.modl1.append(modl_i)

    def forward(self, x):
        out = []
        for i in range(self.num_out):
            if i < self.num_in:
                residual = x[i]
                for layer in self.modl1[i]:
                    residual = layer(residual)
                out.append(residual)
            else:
                residual = x[-1]
                for layer in self.modl1[i]:
                    residual = layer(residual)
                out.append(residual)
        return out

###/////////////////////////////////////////////###

###//////////////////BasicBlock//////////////////////###
###/////////////////////////////////////////////###
# TODO Pytorch
class BasicBlock(nn.Module):
    def __init__(self, size):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=size)
        self.bn2 = nn.BatchNorm2d(num_features=size)

        self.conv1 = conv2D(in_channels=size, out_channels=size, kernel_size=3)
        self.conv2 = conv2D(in_channels=size, out_channels=size, kernel_size=3)

        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = out + residual
        return out
###/////////////////////////////////////////////###

###//////////////////branches//////////////////////###
###/////////////////////////////////////////////###
# TODO Pytorch
class branches(nn.Module):
    def __init__(self, block_num, channels):
        super(branches, self).__init__()

        self.resblocks = nn.CellList([nn.CellList([BasicBlock(i) for _ in range(block_num)]) for i in channels])
    def forward(self, x):
        out = []
        for ind, resbs in enumerate(self.resblocks):
            residual = x[ind]
            for resb in resbs:
                residual = resb(residual)
            out.append(residual)
        return out

###/////////////////////////////////////////////###

###//////////////////fuse_layers//////////////////////###
###/////////////////////////////////////////////###
# TODO Pytorch
class fuse_layers(nn.Module):
    def __init__(self, channels, multi_scale_output=True):
        super(fuse_layers, self).__init__()
        self.channels = channels
        self.multi_scale_output = multi_scale_output
        if multi_scale_output:
            self.layers = nn.CellList([nn.CellList([nn.CellList([]) for _ in channels]) for _ in channels])
            # [ [ [....],[....],[....]  ],      [],         []]
            #   64(64     128    256)           128()      256()
            for i in range(len(channels)):
                for j in range(len(channels)):
                    if j > i:
                        self.layers[i][j].append(nn.BatchNorm2d(num_features=channels[j]))
                        self.layers[i][j].append(nn.ReLU())
                        self.layers[i][j].append(
                            conv2D(in_channels=channels[j], out_channels=channels[i], kernel_size=1, padding=0))
                        self.layers[i][j].append(ResizeNearestNeighbor_m(scale_factor=2 ** (j - i)))
                    elif j < i:
                        for k in range(i - j):
                            self.layers[i][j].append(nn.BatchNorm2d(num_features=channels[j]))
                            self.layers[i][j].append(nn.ReLU())
                            if k == i - j - 1:
                                self.layers[i][j].append(conv2D(in_channels=channels[j], out_channels=channels[i], kernel_size=3, stride=2))
                            else:
                                self.layers[i][j].append(conv2D(in_channels=channels[j], out_channels=channels[j], kernel_size=3, stride=2))
                    elif j == i:
                        self.layers[i][j] =Identity_m()
        else:
            self.layers = nn.CellList([nn.CellList([]) for _ in channels])
            i = 0
            for j in range(len(channels)):
                if j > i:
                    self.layers[j].append(nn.BatchNorm2d(num_features=channels[j]))
                    self.layers[j].append(nn.ReLU())
                    self.layers[j].append(conv2D(in_channels=channels[j], out_channels=channels[i], kernel_size=1, padding=0))
                    self.layers[j].append(ResizeNearestNeighbor_m(scale_factor=2 ** (j - i)))
                else:
                    self.layers[j] = Identity_m()

    def forward(self, x):
        out = []
        if self.multi_scale_output:
            for i in range(len(self.channels)):
                residual = x[i]
                for j in range(len(self.channels)):
                    if j == i:
                        continue
                    y = x[j]
                    for layer in self.layers[i][j]:

                        y = layer(y)
                    residual = residual + y
                out.append(residual)
        else:
            i = 0
            residual = x[i]
            for j in range(len(self.channels)):
                y = x[j]
                if j > i:
                    for _, layer in enumerate(self.layers[j]):
                        y = layer(y)
                residual = residual + y
            out.append(residual)
        return out
###/////////////////////////////////////////////###

###//////////////////HighResolutionModule//////////////////////###
###/////////////////////////////////////////////###
# TODO Pytorch
class HighResolutionModule(nn.Module):
    def __init__(self, channels, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self.branches = branches(4, channels)
        self.fuse_layers = fuse_layers(channels, multi_scale_output=multi_scale_output)
    def forward(self, x):
        x = self.branches(x)
        x = self.fuse_layers(x)
        return x

###/////////////////////////////////////////////###



###//////////////////stage//////////////////////###
###/////////////////////////////////////////////###
# TODO Pytorch
class stage(nn.Module):
    def __init__(self, num_modules, channels, multi_scale_output=True):
        super(stage, self).__init__()

        if multi_scale_output:
            self.convblocks = nn.CellList([HighResolutionModule(channels) for _ in range(num_modules)])
        else:
            self.convblocks = nn.CellList([HighResolutionModule(channels) for _ in range(num_modules - 1)] + [
                HighResolutionModule(channels, multi_scale_output=False)])
    def forward(self, x):
        for layer in self.convblocks:
            x = layer(x)
        return x


###/////////////////////////////////////////////###






###/////////////////////////////////////////////###
class hrnetv2(nn.Module):
    def __init__(self, output_class):
        super(hrnetv2, self).__init__()
        self.channels_2 = [32, 64]
        self.channels_3 = [32, 64, 128]
        self.channels_4 = [32, 64, 128, 256]
        self.num_modules_2 = 1
        self.num_modules_3 = 4
        self.num_modules_4 = 3

        self.conv1 = conv2D(in_channels=3, out_channels=64, kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU()
        self.conv2 = conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(num_features=64)

        self.layer1 = layer1(in_channels=64)
        self.transition_layer1 = transition_layer(in_channels=[256], out_channels=self.channels_2)

        self.stage1 = stage(num_modules=self.num_modules_2, channels=self.channels_2)
        self.transition_layer2 = transition_layer(in_channels=self.channels_2, out_channels=self.channels_3)
        self.stage2 = stage(num_modules=self.num_modules_3, channels=self.channels_3)
        self.transition_layer3 = transition_layer(in_channels=self.channels_3, out_channels=self.channels_4)
        self.stage3 = stage(num_modules=self.num_modules_4, channels=self.channels_4, multi_scale_output=False)

        self.upbi1 = ResizeBilinear_m(size=256)
        self.upbn1 = nn.BatchNorm2d(num_features=32)
        self.upconv1 = conv2D(in_channels=32, out_channels=32, kernel_size=3)
        self.upbi2 = ResizeBilinear_m(size=512)
        self.upbn2 = nn.BatchNorm2d(num_features=32)
        self.upconv2 = conv2D(in_channels=32, out_channels=32, kernel_size=3)
        self.upbn3 = nn.BatchNorm2d(num_features=32)
        self.upconv3 = conv2D(in_channels=32, out_channels=output_class, kernel_size=1, padding=0)
    def forward(self, x):
        # input                                             # ---> torch.Size([4/batchsize, 3, 512, 512])
        x = self.conv1(x)                                   # ---> torch.Size([bs, 64, 256, 256])
        x = self.bn1(x)                                     # ---> torch.Size([bs, 64, 256, 256])
        x = self.relu(x)                                    # ---> torch.Size([bs, 64, 256, 256])
        x = self.conv2(x)                                   # ---> torch.Size([bs, 64, 128, 128])
        x = self.bn2(x)                                     # ---> torch.Size([bs, 64, 128, 128])
        x = self.relu(x)                                    # ---> torch.Size([bs, 64, 128, 128])

        x = self.layer1(x)                                  # ---> torch.Size([bs, 256, 128, 128])
        x = self.transition_layer1([x])                     # ---> [torch.Size([bs, 32, 128, 128]), torch.Size([bs, 64, 64, 64])] ---> [path1, path2]

        x = self.stage1(x)                                  # ---> [torch.Size([bs, 32, 128, 128]), torch.Size([bs, 64, 64, 64])] ---> [path1, path2]
        x = self.transition_layer2(x)                       # ---> [torch.Size([bs, 32, 128, 128]), torch.Size([bs, 64, 64, 64]), torch.Size([4, 128, 32, 32])] ---> [path1, path2, path3]

        x = self.stage2(x)                                  # ---> [torch.Size([bs, 32, 128, 128]), torch.Size([bs, 64, 64, 64]), torch.Size([4, 128, 32, 32])] ---> [path1, path2, path3]
        x = self.transition_layer3(x)                       # ---> [torch.Size([bs, 32, 128, 128]), torch.Size([bs, 64, 64, 64]), torch.Size([4, 128, 32, 32]), torch.Size([4, 256, 16, 16])] ---> [path1, path2, path3, path4]
        x = self.stage3(x)                                  # ---> torch.Size([bs, 32, 128, 128])

        x = self.upbi1(x[0])                                # ---> torch.Size([bs, 32, 256, 256])
        x = self.upbn1(x)                                   # ---> torch.Size([bs, 32, 256, 256])
        x = self.relu(x)                                    # ---> torch.Size([bs, 32, 256, 256])
        x = self.upconv1(x)                                 # ---> torch.Size([bs, 32, 256, 256])

        x = self.upbi2(x)                                   # ---> torch.Size([bs, 32, 512, 512])
        x = self.upbn2(x)                                   # ---> torch.Size([bs, 32, 512, 512])
        x = self.relu(x)                                    # ---> torch.Size([bs, 32, 512, 512])
        x = self.upconv2(x)                                 # ---> torch.Size([bs, 32, 512, 512])

        x = self.upbn3(x)                                   # ---> torch.Size([bs, 32, 512, 512])
        x = self.relu(x)                                    # ---> torch.Size([bs, 32, 512, 512])
        x = self.upconv3(x)                                 # ---> torch.Size([bs, 1, 512, 512])

        return x


if __name__ == '__main__':
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    from luojianet_ms.common.initializer import Normal
    input = luojianet_ms.Tensor(shape=(4, 3, 512, 512), dtype=luojianet_ms.float32, init=Normal())
    print('input:')
    print(input.shape)

    net = hrnetv2(output_class=4)
    output = net(input)

    print('output:')
    print(output.shape)
    # for op in output:
    #     print(op.shape)




