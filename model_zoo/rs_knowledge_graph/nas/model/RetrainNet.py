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

        self.cells_index = []
        multi_dict = {0: 1, 1: 2, 2: 4, 3: 8}
        num_last_features = 0
        for i in range(len(self.layers)):
            self.mycells.append(nn.CellList())
            self.cells_index.append([])
            for j in range(self.depth):

                self.mycells[i].append(nn.CellList())
                self.cells_index[i].append(OrderedDict())

                num_connect = 0
                for connection in self.connections:
                    if ([i, j] == connection[1]).all():
                        num_connect += 1
                        if connection[0][0] == -1:
                            self.mycells[i][j].append(cell(self.base_multiplier * multi_dict[0],
                                                         self.base_multiplier * multi_dict[connection[1][1]], cell_arch[i][j]))
                            self.cells_index[i][j][str(connection[0])] = int(len(self.mycells[i][j]) - 1)
                        else:
                            self.mycells[i][j].append(cell(self.base_multiplier * multi_dict[connection[0][1]],
                                                self.base_multiplier * multi_dict[connection[1][1]], cell_arch[i][j]))
                            self.cells_index[i][j][str(connection[0])] = int(len(self.mycells[i][j]) - 1)
                self.node_add_num[i][j] = num_connect

                if i == len(self.layers) -1 and num_connect != 0:
                    num_last_features += self.base_multiplier * multi_dict[j]

        self.last_conv = nn.SequentialCell(nn.Conv2d(num_last_features, 256, kernel_size=3, stride=1, padding=0),
                                       nn.BatchNorm2d(256),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
                                       nn.BatchNorm2d(256),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        print('connections number: \n' + str(self.node_add_num))


    def forward(self, x):
        features = []

        temp = self.stem0(x)
        temp = self.stem1(temp)
        pre_feature = self.stem2(temp)


        for i in range(14):
            features.append([])
            for j in range(self.depth):
                features[i].append(0)
                k = 0
                for connection in self.connections:
                    if ([i, j] == connection[1]).all():
                        if connection[0][0] == -1:
                            index = self.cells_index[i][j][str(connection[0])]
                            features[i][j] += self.mycells[i][j][index](pre_feature)
                        else:
                            if isinstance(features[connection[0][0]][connection[0][1]], int):
                                continue
                            index = self.cells_index[i][j][str(connection[0])]
                            features[i][j] += self.mycells[i][j][index](features[connection[0][0]][connection[0][1]])
                        k += 1

        last_features = features[len(self.layers)-1]

        true_last_features = []

        for last_feature in last_features:
            if not isinstance(last_feature, int):
                true_last_features.append(
                    nn.ResizeBilinear()(last_feature, size=last_features[0].shape[2:], align_corners=True))


        result = ops.Concat(axis=1)(true_last_features)
        result = self.last_conv(result)
        result = nn.ResizeBilinear()(result, size=x.shape[2:], align_corners=True)
        result = ops.Transpose()(result, (0, 2, 3, 1))
        return result
