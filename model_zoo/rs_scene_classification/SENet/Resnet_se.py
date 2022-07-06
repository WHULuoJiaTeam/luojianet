import luojianet_ms as ms
from luojianet_ms import ops, nn
from luojianet_ms.common.initializer import Normal
from luojianet_ms.ops import operations as P
from Resnet import ResNet
from SEblock import SELayer
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride)

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,reduction=16):
        super().__init__()

        #residual function
        self.residual_function = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * SEBasicBlock.expansion, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(out_channels * SEBasicBlock.expansion),
            SELayer(out_channels * SEBasicBlock.expansion,reduction)
        )

        #shortcut
        self.shortcut = nn.SequentialCell()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != SEBasicBlock.expansion * out_channels:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(in_channels, out_channels * SEBasicBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * SEBasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU()(self.residual_function(x) + self.shortcut(x))


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super().__init__()
        self.residual_function = nn.SequentialCell(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, pad_mode='pad'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * SEBottleneck.expansion, kernel_size=1),
            nn.BatchNorm2d(out_channels * SEBottleneck.expansion),
            SELayer(out_channels * SEBottleneck.expansion,reduction)
        )
        self.shortcut = nn.SequentialCell()

        if stride != 1 or in_channels != out_channels * SEBottleneck.expansion:
            self.shortcut = nn.SequentialCell(
                nn.Conv2d(in_channels, out_channels * SEBottleneck.expansion, stride=stride, kernel_size=1),
                nn.BatchNorm2d(out_channels * SEBottleneck.expansion)
            )

    def forward(self, x):
        return nn.ReLU()(self.residual_function(x) + self.shortcut(x))

def se_resnet18(num_classes=1000):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [2, 2, 2, 2],num_classes=num_classes)
    return model


def se_resnet34(num_classes=1000):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    return model


def se_resnet50(num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    return model


def se_resnet101(num_classes=1000):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)

    return model


def se_resnet152(num_classes=1000):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)

    return model
