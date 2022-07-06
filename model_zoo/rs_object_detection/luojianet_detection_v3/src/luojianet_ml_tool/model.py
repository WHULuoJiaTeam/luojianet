# -*- coding: utf-8 -*-
import numpy as np
import luojianet_ms
import luojianet_ms.nn as nn
from luojianet_ms.common.initializer import Normal
from luojianet_ms import load_checkpoint, load_param_into_net

from .resnet import resnet152
from luojianet_ms import ms_function


class LeNet5(nn.Module):
    def __init__(self, num_class=10, num_channel=3):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(400, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class ResNet152_ml(nn.Module):
    def __init__(self, cfg, num_class=16):
        super(ResNet152_ml, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.batch_size
        if self.cfg.MODEL.backbone == 'resnet_152':
            self.backbone = resnet152(class_num=1001)
            if cfg.MODEL.pre_trained and cfg.MODEL.load_model_path != "":
                self.checkpoint_dict = load_checkpoint(cfg.MODEL.load_model_path)

                for oldkey in list(self.checkpoint_dict.keys()):
                    if not oldkey.startswith(
                            ('backbone', 'global_step', 'learning_rate', 'moments', 'momentum')):  # 'end_point'
                        data = self.checkpoint_dict.pop(oldkey)
                        newkey = 'backbone.' + oldkey
                        self.checkpoint_dict[newkey] = data
                for item in list(self.checkpoint_dict.keys()):
                    if not item.startswith('backbone'):
                        self.checkpoint_dict.pop(item)

                for key, value in self.checkpoint_dict.items():
                    tensor = value.asnumpy().astype(np.float32)
                    self.checkpoint_dict[key] = luojianet_ms.Parameter(tensor, key)

                load_param_into_net(self.backbone, self.checkpoint_dict)
                self.backbone.set_train(True)
                # print(self.backbone.conv1.weight.data[0, 0, :, :])

        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Dense(1001, num_class, weight_init=Normal(0.1))

        self.flatten = nn.Flatten()

    # @ms_function()
    def forward(self, x):
        embedding = self.backbone(x)
        embedding = self.flatten(embedding)
        embedding = self.dropout1(embedding)
        embedding = self.fc1(embedding)
        return embedding

