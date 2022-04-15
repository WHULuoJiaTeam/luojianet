import numpy as np
import luojianet_ms
import luojianet_ms.nn as nn


class fpn(nn.Module):

    def __init__(self):
        super(fpn, self).__init__()
        # Top layer
        self.toplayer = nn.Conv2d(512, 64, kernel_size=1, stride=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(256, 64, kernel_size=1, stride=1)
        self.latlayer2 = nn.Conv2d(128, 64, kernel_size=1, stride=1)
        self.latlayer3 = nn.Conv2d(64, 64, kernel_size=1, stride=1)

        self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.smooth3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.upsample = nn.ResizeBilinear()


    # def _upsample_add(self, x, y):
    #     _, _, H, W = y.size()
    #     return self.upsample(x, size=(H, W)) + y

    def call(self, p2, p3, p4, p5):
        # top-down
        p5 = self.toplayer(p5)
        p4 = self.upsample(p5, scale_factor = 2) + self.latlayer1(p4)
        p3 = self.upsample(p4, scale_factor = 2) + self.latlayer2(p3)
        p2 = self.upsample(p3, scale_factor = 2) + self.latlayer3(p2)

        # smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        p2 = self.smooth3(p2)

        return p2, p3, p4, p5