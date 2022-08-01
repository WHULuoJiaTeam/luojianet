import luojianet_ms.nn as nn
from luojianet_ms import ops
class MyWithLossCell(nn.Module):
    """定义损失网络"""
    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def call(self, data, label):
        out = self.backbone(data)
        return self.loss_fn(out, label), out

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
        loss, out = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads), out

class MyWithEvalCell(nn.Module):
    """定义验证流程"""
    def __init__(self, network):
        super(MyWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def call(self, data, label):
        outputs = self.network(data)
        return outputs, label