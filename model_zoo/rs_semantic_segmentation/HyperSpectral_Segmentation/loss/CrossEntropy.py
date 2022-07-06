from luojianet_ms import Tensor
from luojianet_ms import nn,ops
from luojianet_ms.ops import operations as P
from luojianet_ms import dtype as mstype


class SoftmaxCrossEntropyLoss(nn.loss.LossBase):
    def __init__(self, num_cls=22, ignore_label=-1):
        super(SoftmaxCrossEntropyLoss, self).__init__()
        self.one_hot = P.OneHot(axis=-1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.not_equal = P.NotEqual()
        self.num_cls = num_cls
        self.ignore_label = ignore_label
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def forward(self, logits, labels):
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))#[b*h*w,]
        logits_ = self.transpose(logits, (0, 2, 3, 1))#[b,h,w,c]
        logits_ = self.reshape(logits_, (-1, self.num_cls))#[b*h*w,c]
        weights = self.not_equal(labels_int, self.ignore_label)
        weights = self.cast(weights, mstype.float32)
        one_hot_labels = self.one_hot(labels_int, self.num_cls, self.on_value, self.off_value)#[b*h*w,c]
        loss = self.ce(logits_, one_hot_labels)
        loss = self.mul(weights, loss)
        loss = self.div(self.sum(loss), self.sum(weights))
        return loss
