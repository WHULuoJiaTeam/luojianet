import numpy as np
from model.RetrainNet import RetrainNet
from model.StageNet1 import SearchNet1
from model.cell import ReLUConvBN
import luojianet_ms.nn as nn
import luojianet_ms.ops as ops
import luojianet_ms.dataset as ds
from luojianet_ms import Parameter, Tensor, context
from luojianet_ms import ParameterTuple

np.random.seed(2)
context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=2)
# class LinearNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.w = Parameter(Tensor(np.array([1.0, 1.0], np.float32)), name='wang')
#         self.relu = nn.ReLU()
#         self.dense1_0 = nn.Dense(5, 32)
#         self.dense1_1 = nn.Dense(5, 32)
#         self.dense2 = nn.Dense(32, 1)
#
#     def call(self, x):
#         print(self.w)
#         x1 = self.w[0] * self.dense1_0(x)
#         x2 = self.w[1] * self.dense1_1(x)
#         x = x1 + x2
#         x = self.relu(x)
#         x = self.dense2(x)
#         return x
# class DatasetGenerator:
#     def __init__(self):
#         self.data = np.random.randn(5, 5).astype(np.float32)
#         self.label = np.random.randn(5, 1).astype(np.int32)
#
#     def __getitem__(self, index):
#         return self.data[index], self.label[index]
#
#     def __len__(self):
#         return len(self.data)
#
# class TrainOneStepCell(nn.Module):
#     def __init__(self, network, optimizer, sens=1.0):
#         """参数初始化"""
#         super(TrainOneStepCell, self).__init__(auto_prefix=False)
#         self.network = network
#         # 使用tuple包装weight
#         self.weights = ParameterTuple(list(filter(lambda x: 'wang' in x.name, self.get_parameters())))
#         self.optimizer = optimizer
#         # 定义梯度函数
#         self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
#         self.sens = sens
#
#     def call(self, data, label):
#         """构建训练过程"""
#         weights = self.weights
#         loss = self.network(data, label)
#         # 为反向传播设定系数
#         sens = ops.Fill()(ops.DType()(loss), ops.Shape()(loss), self.sens)
#         grads = self.grad(self.network, weights)(data, label, sens)
#         # for grad in grads:
#         #     print(grad)
#         return loss, self.optimizer(grads)
#
# # 对输入数据进行处理
# dataset_generator = DatasetGenerator()
# dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=True)
# dataset = dataset.batch(32)
# # 实例化网络
# net = LinearNet()
# # 设定损失函数
# crit = nn.MSELoss()
# # 设定优化器
# opt = nn.Adam(params=list(filter(lambda x: 'wang' in x.name, net.get_parameters())), learning_rate=0.2)
# # 引入损失函数
# net_with_criterion = nn.WithLossCell(net, crit)
# # 自定义网络训练
# train_net = TrainOneStepCell(net_with_criterion, opt)
#
# # 获取训练过程数据
# for i in range(3000):
#     for d in dataset.create_dict_iterator():
#         train_net(d["data"], d["label"])
#         print(net_with_criterion(d["data"], d["label"]))
#         print()

layers = np.ones([14, 4])
cell_arch = np.load(
    '/media/dell/DATA/wy/Seg_NAS/run/uadataset/search/experiment_0/cell_arch/2_cell_arch_epoch_nors24.npy')
connections = np.load(
    '/media/dell/DATA/wy/Seg_NAS/run/uadataset/search/experiment_0/connections/2_connections_epoch37.npy')
net = RetrainNet(layers, 4, connections, cell_arch, 'uadataset', 12)

# layers = np.ones([14, 4])
# connections = np.load("/media/dell/DATA/wy/Seg_NAS/model/model_encode/GID-5/14layers_mixedcell1_3operation/first_connect_4.npy")
# net = SearchNet1(layers, 4, connections, ReLUConvBN, 'uadataset', 12)
a = (1e-3 * ops.StandardNormal()((4, 3, 64, 64)))
b = net(a)
print(b)