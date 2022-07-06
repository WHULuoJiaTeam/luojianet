from luojianet_ms import nn
from luojianet_ms.ops import operations as P


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, weight_init='HeUniform')


def conv3x3(in_planes, out_planes, stride=1, dilation=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad', padding=padding,
                     dilation=dilation, weight_init='HeUniform')


class Resnet(nn.Module):
    def __init__(self, block, block_num, output_stride, use_batch_statistics=True):
        super(Resnet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, pad_mode='pad', padding=3,
                               weight_init='HeUniform')
        self.bn1 = nn.BatchNorm2d(self.inplanes, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 64, block_num[0], use_batch_statistics=use_batch_statistics)
        self.layer2 = self._make_layer(block, 128, block_num[1], stride=2, use_batch_statistics=use_batch_statistics)

        if output_stride == 16:
            self.layer3 = self._make_layer(block, 256, block_num[2], stride=2,
                                           use_batch_statistics=use_batch_statistics)
            self.layer4 = self._make_layer(block, 512, block_num[3], stride=1, base_dilation=2, grids=[1, 2, 4],
                                           use_batch_statistics=use_batch_statistics)
        elif output_stride == 8:
            self.layer3 = self._make_layer(block, 256, block_num[2], stride=1, base_dilation=2,
                                           use_batch_statistics=use_batch_statistics)
            self.layer4 = self._make_layer(block, 512, block_num[3], stride=1, base_dilation=4, grids=[1, 2, 4],
                                           use_batch_statistics=use_batch_statistics)

    def _make_layer(self, block, planes, blocks, stride=1, base_dilation=1, grids=None, use_batch_statistics=True):
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, use_batch_statistics=use_batch_statistics)
            ])

        if grids is None:
            grids = [1] * blocks

        layers = [
            block(self.inplanes, planes, stride, downsample, dilation=base_dilation * grids[0],
                  use_batch_statistics=use_batch_statistics)
        ]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=base_dilation * grids[i],
                      use_batch_statistics=use_batch_statistics))

        return nn.SequentialCell(layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.layer1(out)
        low_level_feat = out
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out, low_level_feat


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, use_batch_statistics=True):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics)

        self.conv2 = conv3x3(planes, planes, stride, dilation, dilation)
        self.bn2 = nn.BatchNorm2d(planes, use_batch_statistics=use_batch_statistics)

        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, use_batch_statistics=use_batch_statistics)

        self.relu = nn.ReLU()
        self.downsample = downsample



    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)


        out = out + identity
        out = self.relu(out)
        return out


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate=1, use_batch_statistics=True):
        super(ASPPConv, self).__init__()
        if atrous_rate == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False, weight_init='HeUniform')
        else:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='pad', padding=atrous_rate,
                             dilation=atrous_rate, weight_init='HeUniform')
        bn = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        relu = nn.ReLU()
        self.aspp_conv = nn.SequentialCell([conv, bn, relu])

    def forward(self, x):
        out = self.aspp_conv(x)
        return out

def SeparableConv2d(in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,bias=True):
    dephtwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,pad_mode='pad', padding=padding,
                               dilation=dilation, group=in_channels, has_bias=False,weight_init='HeUniform')
    pointwise_conv = nn.Conv2d(in_channels,out_channels,kernel_size=1,has_bias=bias,weight_init='HeUniform')
    return nn.SequentialCell([dephtwise_conv,pointwise_conv])
class ASPPSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate=1, use_batch_statistics=True):
        super(ASPPSeparableConv, self).__init__()
        if atrous_rate == 1:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, has_bias=False, weight_init='HeUniform')
        else:
            conv = SeparableConv2d(in_channels,out_channels,kernel_size=3,padding=atrous_rate,
                                   dilation=atrous_rate,bias=False)
            # conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, pad_mode='pad', padding=atrous_rate,
            #                  dilation=atrous_rate, weight_init='xavier_uniform')
        bn = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        relu = nn.ReLU()
        self.aspp_conv = nn.SequentialCell([conv, bn, relu])

    def forward(self, x):
        out = self.aspp_conv(x)
        return out

class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_statistics=True):
        super(ASPPPooling, self).__init__()
        self.conv = nn.SequentialCell([
            nn.Conv2d(in_channels, out_channels, kernel_size=1, weight_init='HeUniform'),
            nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics),
            nn.ReLU()
        ])
        self.shape = P.Shape()

    def forward(self, x):
        size = self.shape(x)
        out = nn.AvgPool2d(size[2])(x)
        out = self.conv(out)
        out = P.ResizeBilinear((size[2], size[3]), True)(out)
        return out


class ASPP(nn.Module):
    def __init__(self, atrous_rates, phase='train', in_channels=2048, num_classes=21,
                 use_batch_statistics=True):
        super(ASPP, self).__init__()
        self.phase = phase
        out_channels = 256
        self.aspp1 = ASPPSeparableConv(in_channels, out_channels, atrous_rates[0], use_batch_statistics=use_batch_statistics)
        self.aspp2 = ASPPSeparableConv(in_channels, out_channels, atrous_rates[1], use_batch_statistics=use_batch_statistics)
        self.aspp3 = ASPPSeparableConv(in_channels, out_channels, atrous_rates[2], use_batch_statistics=use_batch_statistics)
        self.aspp4 = ASPPSeparableConv(in_channels, out_channels, atrous_rates[3], use_batch_statistics=use_batch_statistics)
        self.aspp_pooling = ASPPPooling(in_channels, out_channels)
        self.conv1 = nn.Conv2d(out_channels * (len(atrous_rates) + 1), out_channels, kernel_size=1,
                               weight_init='HeUniform')
        self.bn1 = nn.BatchNorm2d(out_channels, use_batch_statistics=use_batch_statistics)
        self.relu = nn.ReLU()
        self.concat = P.Concat(axis=1)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.aspp_pooling(x)

        x = self.concat((x1, x2))
        x = self.concat((x, x3))
        x = self.concat((x, x4))
        x = self.concat((x, x5))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #if self.phase == 'train':
        x = self.drop(x)
        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, phase='train', num_classes=21, output_stride=16,aspp_atrous_rates=[1, 6, 12, 18], freeze_bn=False):
        super(DeepLabV3Plus, self).__init__()
        self.use_batch_statistics = None
        self.phase=phase
        self.resnet = Resnet(Bottleneck, [3, 4, 6, 3], output_stride=output_stride,
                             use_batch_statistics=self.use_batch_statistics)
        self.aspp = nn.SequentialCell([ASPP(aspp_atrous_rates, self.phase, 2048, num_classes,
                         use_batch_statistics=self.use_batch_statistics),
                                       SeparableConv2d(256, 256,kernel_size=3, padding=1, bias=False),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU(),
                                       ])
        self.shape = P.Shape()
        self.conv2 = nn.Conv2d(256, 48, kernel_size=1, weight_init='HeUniform')
        self.bn2 = nn.BatchNorm2d(48, use_batch_statistics=self.use_batch_statistics)
        self.relu = nn.ReLU()
        self.concat = P.Concat(axis=1)
        self.last_conv = nn.SequentialCell([
            SeparableConv2d(304,256,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(256, use_batch_statistics=self.use_batch_statistics),
            nn.ReLU(),
            conv1x1(256, num_classes, stride=1)
        ])
        self.act = P.Softmax(axis=1) if num_classes > 1 else P.Sigmoid()
    def forward(self, x):
        size = self.shape(x)
        out, low_level_features = self.resnet(x)
        size2 = self.shape(low_level_features)
        out = self.aspp(out)
        out = P.ResizeBilinear((size2[2], size2[3]), True)(out)
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)
        out = self.concat((out, low_level_features))
        out = self.last_conv(out)
        out = P.ResizeBilinear((size[2], size[3]), True)(out)
        return out
