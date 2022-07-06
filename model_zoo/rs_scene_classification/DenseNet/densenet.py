"""dense net in luojianet
[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger.

    Densely Connected Convolutional Networks
    https://arxiv.org/abs/1608.06993v5
"""
from luojianet_ms import ops, nn
from luojianet_ms import context

#"""Bottleneck layers. Although each layer only produces k
#output feature-maps, it typically has many more inputs. It
#has been noted in [37, 11] that a 1×1 convolution can be in-
#troduced as bottleneck layer before each 3×3 convolution
#to reduce the number of input feature-maps, and thus to
#improve computational efficiency."""
class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        #"""In  our experiments, we let each 1×1 convolution
        #produce 4k feature-maps."""
        inner_channel = 4 * growth_rate

        #"""We find this design especially effective for DenseNet and
        #we refer to our network with such a bottleneck layer, i.e.,
        #to the BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3) version of H ` ,
        #as DenseNet-B."""
        self.bottle_neck = nn.SequentialCell(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, inner_channel, kernel_size=1, has_bias=False, pad_mode='pad'),
            nn.BatchNorm2d(inner_channel),
            nn.ReLU(),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, has_bias=False, pad_mode='pad')
        )
        self.cat = ops.Concat(axis=1)
    def forward(self, x):
        return self.cat((x, self.bottle_neck(x)))

#"""We refer to layers between blocks as transition
#layers, which do convolution and pooling."""
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        #"""The transition layers used in our experiments
        #consist of a batch normalization layer and an 1×1
        #convolutional layer followed by a 2×2 average pooling
        #layer""".
        self.down_sample = nn.SequentialCell(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1, has_bias=False, pad_mode='pad'),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)

#DesneNet-BC
#B stands for bottleneck layer(BN-RELU-CONV(1x1)-BN-RELU-CONV(3x3))
#C stands for compression factor(0<=theta<=1)
class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_class=100):
        super().__init__()
        self.growth_rate = growth_rate

        #"""Before entering the first dense block, a convolution
        #with 16 (or twice the growth rate for DenseNet-BC)
        #output channels is performed on the input images."""
        inner_channels = 2 * growth_rate

        #For convolutional layers with kernel size 3×3, each
        #side of the inputs is zero-padded by one pixel to keep
        #the feature-map size fixed.
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=3, padding=1, has_bias=False, pad_mode='pad')

        self.features = nn.SequentialCell()

        for index in range(len(nblocks) - 1):
            self.features.append(self._make_dense_layers(block, inner_channels, nblocks[index]))
            inner_channels += growth_rate * nblocks[index]

            #"""If a dense block contains m feature-maps, we let the
            #following transition layer generate θm output feature-
            #maps, where 0 < θ ≤ 1 is referred to as the compression
            #fac-tor.
            out_channels = int(reduction * inner_channels) # int() will automatic floor the value
            self.features.append(Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.append(self._make_dense_layers(block, inner_channels, nblocks[len(nblocks)-1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.append(nn.BatchNorm2d(inner_channels))
        self.features.append(nn.ReLU())

        self.avgpool = ops.ReduceMean()

        self.linear = nn.Dense(inner_channels, num_class)

    def forward(self, x):
        b,c,_,__ = x.shape
        output = self.conv1(x)
        output = self.features(output)
        output = self.avgpool(output, (2, 3))
        output = output.view(b, -1)
        output = self.linear(output)
        return output

    def _make_dense_layers(self, block, in_channels, nblocks):
        dense_block = nn.SequentialCell()
        for index in range(nblocks):
            dense_block.append(block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

def densenet121(num_class):
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32, num_class=num_class)

def densenet161(num_class):
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48, num_class=num_class)

def densenet169(num_class):
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32, num_class=num_class)

def densenet201(num_class):
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32, num_class=num_class)
