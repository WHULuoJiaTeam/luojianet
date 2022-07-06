import luojianet_ms.nn as nn
import luojianet_ms as ms
import luojianet_ms.dataset as ds


class LossCell(nn.Module):
    def __init__(self, backbone, loss_fn):
        super(LossCell, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn
        
    def forward(self, data, label):
        # out = self.backbone(data, label)
        out = self.backbone(data)
        return self.loss_fn(out, label)
        
    def backbone_network(self,):
        return self.backbone

        
class TrainStep(nn.TrainOneStepCell):
    def __init__(self, network, optimizer):
        super(TrainStep, self).__init__(network, optimizer)
        self.grad = ms.ops.GradOperation(get_by_list=True)
        
        
    def forward(self, data, label):
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads)
