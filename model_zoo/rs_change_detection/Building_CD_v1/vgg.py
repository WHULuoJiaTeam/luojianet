import numpy as np
import luojianet_ms
import luojianet_ms.nn as nn



# 网络搭建
class FCNnet(nn.Module):
    def __init__(self):
        super(FCNnet, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.net1_conv1 = nn.Conv2d(3, 64, 3)
        self.net1_conv2 = nn.Conv2d(64, 64, 3)
        self.net1_BN1 = nn.BatchNorm2d(64)
        self.net1_conv3 = nn.Conv2d(128, 64, 3)
        self.net1_conv4 = nn.Conv2d(64, 32, 3)
        self.net1_conv5 = nn.Conv2d(32, 1, 1)

        self.net2_max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.net2_conv1 = nn.Conv2d(64, 128, 3)
        self.net2_conv2 = nn.Conv2d(128, 128, 3)
        self.net2_BN1 = nn.BatchNorm2d(128)
        self.net2_conv3 = nn.Conv2d(256, 128, 3)
        self.net2_conv4 = nn.Conv2d(128, 64, 3)
        self.net2_conv5 = nn.Conv2d(64, 32, 3)
        self.net2_conv6 = nn.Conv2d(32, 1, 1)
        self.net2_convtrans = nn.Conv2dTranspose(64, 64, 2, 2)

        self.net3_max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.net3_conv1 = nn.Conv2d(128, 256, 3)
        self.net3_conv2 = nn.Conv2d(256, 256, 3)
        self.net3_BN1 = nn.BatchNorm2d(256)
        self.net3_conv3 = nn.Conv2d(256, 256, 3)
        self.net3_BN2 = nn.BatchNorm2d(256)
        self.net3_conv4 = nn.Conv2d(512, 256, 3)
        self.net3_conv5 = nn.Conv2d(256, 128, 3)
        self.net3_conv6 = nn.Conv2d(128, 32, 3)
        self.net3_conv7 = nn.Conv2d(32, 1, 1)
        self.net3_convtrans = nn.Conv2dTranspose(128, 128, 2, 2)

        self.net4_max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.net4_conv1 = nn.Conv2d(256, 512, 3)
        self.net4_conv2 = nn.Conv2d(512, 512, 3)
        self.net4_BN1 = nn.BatchNorm2d(512)
        self.net4_conv3 = nn.Conv2d(512, 512, 3)
        self.net4_BN2 = nn.BatchNorm2d(512)
        self.net4_conv4 = nn.Conv2d(1024, 512, 3)
        self.net4_conv5 = nn.Conv2d(512, 256, 3)
        self.net4_conv6 = nn.Conv2d(256, 32, 3)
        self.net4_conv7 = nn.Conv2d(32, 1, 1)
        self.net4_convtrans = nn.Conv2dTranspose(256, 256, 2, 2)

        self.net5_max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.net5_conv1 = nn.Conv2d(512, 512, 3)
        self.net5_conv2 = nn.Conv2d(512, 512, 3)
        self.net5_BN1 = nn.BatchNorm2d(512)
        self.net5_conv3 = nn.Conv2d(512, 512, 3)
        self.net5_BN2 = nn.BatchNorm2d(512)
        self.net5_convtrans = nn.Conv2dTranspose(512, 512, 2, 2)

        self.cat = luojianet_ms.ops.Concat(axis=1)

    def forward(self, x):
        x1 = self.net1_conv1(x)
        x1 = self.relu(x1)
        x1 = self.net1_conv2(x1)
        x1 = self.net1_BN1(x1)
        x1 = self.relu(x1)
        out1 = x1

        x2 = self.net2_max_pool(x1)
        x2 = self.net2_conv1(x2)
        x2 = self.relu(x2)
        x2 = self.net2_conv2(x2)
        x2 = self.net2_BN1(x2)
        x2 = self.relu(x2)
        out2 = x2

        x3 = self.net3_max_pool(x2)
        x3 = self.net3_conv1(x3)
        x3 = self.relu(x3)
        x3 = self.net3_conv2(x3)
        x3 = self.net3_BN1(x3)
        x3 = self.relu(x3)
        x3 = self.net3_conv3(x3)
        x3 = self.net3_BN2(x3)
        x3 = self.relu(x3)
        out3 = x3

        x4 = self.net4_max_pool(x3)
        x4 = self.net4_conv1(x4)
        x4 = self.relu(x4)
        x4 = self.net4_conv2(x4)
        x4 = self.net4_BN1(x4)
        x4 = self.relu(x4)
        x4 = self.net4_conv3(x4)
        x4 = self.net4_BN2(x4)
        x4 = self.relu(x4)
        out4 = x4

        x5 = self.net5_max_pool(x4)
        x5 = self.net5_conv1(x5)
        x5 = self.relu(x5)
        x5 = self.net5_conv2(x5)
        x5 = self.net5_BN1(x5)
        x5 = self.relu(x5)
        x5 = self.net5_conv3(x5)
        x5 = self.net5_BN2(x5)
        x5 = self.relu(x5)

        x5 = self.net5_convtrans(x5)
        x4 = self.cat((x4, x5))
        x4 = self.net4_conv4(x4)
        x4 = self.relu(x4)
        x4 = self.net4_conv5(x4)
        x4 = self.relu(x4)


        x4 = self.net4_convtrans(x4)
        x3 = self.cat((x3, x4))
        x3 = self.net3_conv4(x3)
        x3 = self.relu(x3)
        x3 = self.net3_conv5(x3)
        x3 = self.relu(x3)


        x3 = self.net3_convtrans(x3)
        x2 = self.cat((x2, x3))
        x2 = self.net2_conv3(x2)
        x2 = self.relu(x2)
        x2 = self.net2_conv4(x2)
        x2 = self.relu(x2)


        x2 = self.net2_convtrans(x2)
        x1 = self.cat((x1, x2))
        x1 = self.net1_conv3(x1)
        x1 = self.relu(x1)
        x1 = self.net1_conv4(x1)
        x1 = self.relu(x1)
        x1 = self.net1_conv5(x1)
        out5 = self.sigmoid(x1)
        out5 = luojianet_ms.ops.clip_by_value(out5, 1e-4, 1-1e-4)

        return out1, out2, out3, out4, out5

# a = torch.randn((1,3,512,512))
# net = FCNnet()
# out1, out2, out3, out4, out5 = net(a)
# print(out1.size(), out2.size(), out3.size(), out4.size(), out5.size())




