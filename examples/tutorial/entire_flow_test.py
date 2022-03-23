"""
This script is about mid/low leval api usage.
"""
import numpy as np
import luojianet_ms as ms
from luojianet_ms import ops, nn
import luojianet_ms.dataset as ds
import luojianet_ms.common.initializer as init


def get_data(data_num, data_size):
    for _ in range(data_num):
        data = np.random.randn(data_size)
        p = np.array([1, 0, -3, 5])
        label = np.polyval(p, data).sum()
        yield data.astype(np.float32), np.array([label]).astype(np.float32)

def create_dataset(data_num, data_size, batch_size=32, repeat_size=1):
    """定义数据集"""
    input_data = ds.GeneratorDataset(list(get_data(data_num, data_size)), column_names=['data', 'label'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data

class MyNet(nn.Module):
    """定义网络"""
    def __init__(self, input_size=32):
        super(MyNet, self).__init__()
        self.fc1 = nn.Dense(input_size, 120, weight_init=init.Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=init.Normal(0.02))
        self.fc3 = nn.Dense(84, 1, weight_init=init.Normal(0.02))
        self.relu = nn.ReLU()

    def call(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class MyL1Loss(nn.LossBase):
    """定义损失"""
    def __init__(self, reduction="mean"):
        super(MyL1Loss, self).__init__(reduction)
        self.abs = ops.Abs()

    def call(self, base, target):
        x = self.abs(base - target)
        return self.get_loss(x)

class MyMomentum(nn.Optimizer):
    """使用ApplyMomentum算子定义优化器"""
    def __init__(self, params, learning_rate, momentum=0.9, use_nesterov=False):
        super(MyMomentum, self).__init__(learning_rate, params)
        self.moments = self.parameters.clone(prefix="moments", init="zeros")
        self.momentum = momentum
        self.opt = ops.ApplyMomentum(use_nesterov=use_nesterov)

    def call(self, gradients):
        params = self.parameters
        success = None
        for param, mom, grad in zip(params, self.moments, gradients):
            success = self.opt(param, mom, self.learning_rate, grad, self.momentum)
        return success

class MyWithLossCell(nn.Module):
    """定义损失网络"""
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def call(self, data, label):
        out = self.backbone(data)
        return self.loss_fn(out, label)

    def backbone_network(self):
        return self.backbone

class MyTrainStep(nn.TrainOneStepCell):
    """定义训练流程"""
    def __init__(self, network, optimizer):
        """参数初始化"""
        super(MyTrainStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def call(self, data, label):
        """构建训练过程"""
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads)

# 生成多项式分布的数据
dataset_size = 32
ds_train = create_dataset(2048, dataset_size)
# 网络
net = MyNet()
# 损失函数
loss_func = MyL1Loss()
# 优化器
opt = MyMomentum(net.trainable_params(), 0.01)
# 构建损失网络
net_with_criterion = MyWithLossCell(net, loss_func)
# 构建训练网络
train_net = MyTrainStep(net_with_criterion, opt)
# 执行训练，每个epoch打印一次损失值
epochs = 5
for epoch in range(epochs):
    for train_x, train_y in ds_train:
        train_net(train_x, train_y)
        loss_val = net_with_criterion(train_x, train_y)
    print(loss_val)

class MyMAE(nn.Metric):
    """定义metric"""
    def __init__(self):
        super(MyMAE, self).__init__()
        self.clear()

    def clear(self):
        self.abs_error_sum = 0
        self.samples_num = 0

    def update(self, *inputs):
        y_pred = inputs[0].asnumpy()
        y = inputs[1].asnumpy()
        error_abs = np.abs(y.reshape(y_pred.shape) - y_pred)
        self.abs_error_sum += error_abs.sum()
        self.samples_num += y.shape[0]

    def eval(self):
        return self.abs_error_sum / self.samples_num


class MyWithEvalCell(nn.Module):
    """定义验证流程"""
    def __init__(self, network):
        super(MyWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def call(self, data, label):
        outputs = self.network(data)
        return outputs, label

# 获取验证数据
ds_eval = create_dataset(128, dataset_size, 1)
# 定义评估网络
eval_net = MyWithEvalCell(net)
eval_net.set_train(False)
# 定义评估指标
mae = MyMAE()
# 执行推理过程
for eval_x, eval_y in ds_eval:
    output, eval_y = eval_net(eval_x, eval_y)
    mae.update(output, eval_y)

mae_result = mae.eval()
print("mae: ", mae_result)

ms.save_checkpoint(net, "./MyNet1.ckpt")
