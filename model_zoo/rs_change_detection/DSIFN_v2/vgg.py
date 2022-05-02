"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''
import luojianet_ms as ms
from luojianet_ms import ops, nn
from luojianet_ms.common.initializer import Normal
from luojianet_ms.ops import operations as P

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_classes=100):
        super().__init__()
        self.features = features

        self.classifier = nn.SequentialCell(
            nn.Dense(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Dense(4096, num_classes)
        )

    def call(self, x):
        output = self.features(x)
        output = ops.Reshape()(output,(output.shape[0], -1))
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=True):
    layers = []

    in_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channel, l, kernel_size=3, padding=1, pad_mode='pad')
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(l), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channel = l
 
    return nn.SequentialCell(layers)

def vgg16_bn(num_classes=100):
    return VGG(make_layers(cfg['D'], batch_norm=True),num_classes)

# ### 测试
# net = vgg11_bn()
# a = ops.StandardNormal()((1,3,224,224))
# b = net(a)
# print(b.shape)