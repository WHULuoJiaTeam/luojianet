import numpy as np
import luojianet_ms as luojia
from luojianet_ms import nn
from luojianet_ms import ops
import numpy
from model.cell import ReLUConvBN, MixedRetrainCell
from collections import OrderedDict


class RetrainNet(nn.Module):

    def __init__(self, layers, depth, connections, cell_arch, dataset, num_classes, base_multiplier=40):
        '''
        Args:
            layers: layer × depth： one or zero, one means ture
            depth: the model scale depth
            connections: the node connections
            cell: cell type
            dataset: dataset
            num_classes: the number of classes
            base_multiplier: base scale multiplier
        '''
        super(RetrainNet, self).__init__()
        self.block_multiplier = 1
        self.base_multiplier = base_multiplier
        self.depth = depth
        self.layers = layers
        self.connections = connections

        self.node_add_num = np.zeros([len(layers), self.depth])

        cell = MixedRetrainCell

        half_base = int(base_multiplier // 2)
        if 'GID' in dataset:
            input_channel = 4
        else:
            input_channel = 3
        self.stem0 = nn.SequentialCell(
            nn.Conv2d(input_channel, half_base * self.block_multiplier, 3, stride=2, padding=0),
            nn.BatchNorm2d(half_base * self.block_multiplier),
            nn.ReLU()
        )
        self.stem1 = nn.SequentialCell(
            nn.Conv2d(half_base * self.block_multiplier, half_base * self.block_multiplier, 3, stride=1, padding=0),
            nn.BatchNorm2d(half_base * self.block_multiplier),
            nn.ReLU()
        )
        self.stem2 = nn.SequentialCell(
            nn.Conv2d(half_base * self.block_multiplier, self.base_multiplier * self.block_multiplier, 3, stride=2,padding=0),
            nn.BatchNorm2d(self.base_multiplier * self.block_multiplier),
            nn.ReLU()
        )
        self.mycells = nn.CellList()
        self.cells_index = OrderedDict()
        multi_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        num_last_features = 0

        for connection in self.connections:

            if connection[0][0] == -1:
                self.mycells.append(cell(self.base_multiplier * multi_dict[0],
                                             self.base_multiplier * multi_dict[connection[1][1]], cell_arch[connection[1][0]][connection[1][1]]))
                self.cells_index[str(connection)] = int(len(self.mycells) - 1)
            else:
                self.mycells.append(cell(self.base_multiplier * multi_dict[connection[0][1]],
                                    self.base_multiplier * multi_dict[connection[1][1]], cell_arch[connection[1][0]][connection[1][1]]))
                self.cells_index[str(connection)] = int(len(self.mycells) - 1)
            self.node_add_num[connection[1][0]][connection[1][1]] += 1

        for i in range(self.depth):
            if self.node_add_num[int(len(self.layers) - 1)][i] != 0:
                num_last_features += self.base_multiplier * multi_dict[i]

        self.last_conv = nn.SequentialCell(nn.Conv2d(num_last_features, 256, kernel_size=3, stride=1, padding=0),
                                       nn.BatchNorm2d(256),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
                                       nn.BatchNorm2d(256),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        print('connections number: \n' + str(self.node_add_num))


    def call(self, x):
        features = []

        temp = self.stem0(x)
        temp = self.stem1(temp)
        pre_feature = self.stem2(temp)


        for i in range(14):
            features.append([])
            for j in range(self.depth):
                features[i].append(0)

        for connection in self.connections:
            if connection[0][0] == -1:
                index = self.cells_index[str(connection)]
                features[connection[1][0]][connection[1][1]] += self.mycells[index](pre_feature)
            else:
                if isinstance(features[connection[0][0]][connection[0][1]], int):
                    continue
                index = self.cells_index[str(connection)]
                features[connection[1][0]][connection[1][1]] += self.mycells[index](features[connection[0][0]][connection[0][1]])

        last_features = features[len(self.layers)-1]

        true_last_features = []

        for last_feature in last_features:
            if not isinstance(last_feature, int):
                true_last_features.append(
                    nn.ResizeBilinear()(last_feature, size=last_features[0].shape[2:], align_corners=True))


        result = ops.Concat(axis=1)(true_last_features)
        result = self.last_conv(result)
        result = nn.ResizeBilinear()(result, size=x.shape[2:], align_corners=True)
        # result = ops.Transpose()(result, (0, 2, 3, 1))
        return result
