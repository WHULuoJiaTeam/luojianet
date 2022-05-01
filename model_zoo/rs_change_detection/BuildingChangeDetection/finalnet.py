import numpy as np
import luojianet_ms
import luojianet_ms.nn as nn

class up_conv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(up_conv, self).__init__()
        self.upsample = nn.ResizeBilinear()
        self.relu = nn.ReLU()
        self.bn = nn.GroupNorm(4, out_channel)
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1)
        #self.UpConv = nn.SequentialCell(self.upsample(scale_factor=2), self.conv, self.bn, self.relu)

    def call(self, input):
        x = self.upsample(input, scale_factor=2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class finalnet(nn.Module):
    def __init__(self):
        super(finalnet, self).__init__()

        self.conv1_1 = up_conv(64, 32)
        self.conv1_2 = up_conv(32, 32)
        self.conv1_3 = up_conv(32, 32)

        self.conv2_1 = up_conv(64, 32)
        self.conv2_2 = up_conv(32, 32)

        self.conv3_1 = up_conv(64, 32)

        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
        self.bn = nn.GroupNorm(4, 32)
        self.relu = nn.ReLU()

        self.conv6 = nn.Conv2d(32, 1, kernel_size=1, stride=1)

        self.sigmoid = nn.Sigmoid()



    def call(self, p3, p4, p5, p6):
        p6 = self.conv1_1(p6)
        p6 = self.conv1_2(p6)
        p6 = self.conv1_3(p6)

        p5 = self.conv2_1(p5)
        p5 = self.conv2_2(p5)

        p4 = self.conv3_1(p4)

        p3 = self.conv5(p3)
        p3 = self.bn(p3)
        p3 = self.relu(p3)

        p = p3 + p4 + p5 + p6

        p1 = self.conv6(p)
        p1 = self.sigmoid(p1)
        p1 = luojianet_ms.ops.clip_by_value(p1, 1e-4, 1 - 1e-4)

        return p1