'''VGG11/13/16/19 in LuojiaNet.'''
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

    def forward(self, x):
        output = self.features(x)
        output = ops.Reshape()(output,(output.shape[0], -1))
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1,pad_mode='pad')]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU()]
        input_channel = l

    return nn.SequentialCell(layers)

def vgg11_bn(num_classes):
    return VGG(make_layers(cfg['A'], batch_norm=True),num_classes)

def vgg13_bn(num_classes):
    return VGG(make_layers(cfg['B'], batch_norm=True),num_classes)

def vgg16_bn(num_classes):
    return VGG(make_layers(cfg['D'], batch_norm=True),num_classes)

def vgg19_bn(num_classes):
    return VGG(make_layers(cfg['E'], batch_norm=True),num_classes)

# ### test
# net = vgg11_bn(num_classes=100)
# a = ops.StandardNormal()((1,3,224,224))
# b = net(a)
# print(b.shape)
