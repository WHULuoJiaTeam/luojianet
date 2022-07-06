import numpy as np
import luojianet_ms as ms
from luojianet_ms import ops, nn
from luojianet_ms.common.initializer import Normal
from luojianet_ms.ops import operations as P

#import torch
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.mean = ops.ReduceMean()
        self.fc1 = nn.Dense(channel, channel//reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Dense(channel // reduction, channel)
        self.sigmoid = nn.Sigmoid()
        self.expand = ops.ExpandDims()
    def forward(self, x):
        shape = x.shape
        y = self.mean(x,(2,3))
        y = self.sigmoid(self.fc2(self.relu(self.fc1(y))))
        y = self.expand(self.expand(y,2),2)
        y = ops.BroadcastTo(shape)(y)
        return x*y
        
# 实例化网络
# net = SELayer(64)
# a = ops.StandardNormal()((1,64,128,128))
# # b = a.view(1,-1)
# b = net(a)
# print(b.shape)
