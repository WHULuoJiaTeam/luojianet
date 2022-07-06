import argparse

import luojianet_ms.nn as nn
import luojianet_ms.ops as P
import luojianet_ms.common.dtype as mstype
from luojianet_ms import Tensor, context
import numpy as np

# context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self,
                 cls_weight=None,
                 ignore_label=-1,
                 sparse=True):
        super(WeightedCrossEntropyLoss, self).__init__()

        self.ignore_label = ignore_label

        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=sparse, reduction='none')
        self.not_equal = P.NotEqual()
        self.equal = P.Equal()
        self.zeros = P.Zeros()
        self.ones = P.Ones()
        self.mul = P.Mul()
        self.sum = P.ReduceSum(False)
        self.div = P.RealDiv()
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

        if cls_weight is not None and cls_weight.dtype != mstype.float32:
            self.cls_weight = self.cast(cls_weight, mstype.float32)
        else:
            self.cls_weight = cls_weight

    def forward(self, logits, labels):

        # if logits.dim() == 4 and labels.dim() == 3:
        num_cls = logits.shape[1]
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, num_cls))

        if self.cls_weight is None:
            weights = self.not_equal(labels_int, self.ignore_label)
            weights = self.cast(weights, mstype.float32)
        else:
            weights = self.zeros(labels_int.shape, mstype.float32)
            for idx, ele in enumerate(self.cls_weight):
                one_weight = self.equal(labels_int, idx)
                one_weight = self.cast(one_weight, mstype.float32)
                one_weight = one_weight * ele
                weights = weights + one_weight

        labels_int[labels_int == self.ignore_label] = 0

        ce_loss = self.ce(logits_, labels_int)
        ce_loss = self.mul(weights, ce_loss)
        ce_loss = self.div(self.sum(ce_loss), self.sum(weights))
        return ce_loss

def build_criterion(args):

    print("=> Trying bulid {:}loss".format(args.criterion))

    if args.criterion == 'ce':
        return WeightedCrossEntropyLoss(cls_weight=args.weight, ignore_label=args.ignore_label, sparse=True)
    else:
        raise ValueError('unknown criterion : {:}'.format(args.criterion))
class CrossEntropyLoss(nn.Module):
    def __init__(self, sparse=True):
        super(CrossEntropyLoss, self).__init__()
        self.cast = P.Cast()
        self.ce = nn.SoftmaxCrossEntropyWithLogits(sparse=sparse, reduction='mean')
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()

    def forward(self, logits, labels):
        num_cls = logits.shape[1]
        labels_int = self.cast(labels, mstype.int32)
        labels_int = self.reshape(labels_int, (-1,))
        logits_ = self.transpose(logits, (0, 2, 3, 1))
        logits_ = self.reshape(logits_, (-1, num_cls))

        ce_loss = self.ce(logits_, labels_int)

        return ce_loss

class MCEloss(nn.loss.loss._Loss):
    def __init__(self, reduction='mean'):
        super().__init__(reduction)
        self.creti = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    def forward(self, data, label):
        batch, channel, height, width = P.Shape()(data)
        data = P.Transpose()(data, (0,2,3,1,))
        data = P.Reshape()(data, (batch * height * width, channel))
        batch, height, width = P.Shape()(label)
        label = P.Reshape()(label, (batch * height * width,))
        x = self.creti(data,label)
        return self.get_loss(x)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="luojianet_ms Loss")
    parser.add_argument('--ignore_label', type=int, default=-1, help='weight standardization')
    Args = parser.parse_args()

    weights = np.array([1, 2, 3, 4])

    Args.weight = Tensor(weights)
    Args.criterion = 'ce'

    wce_loss_luojia = build_criterion(Args)

    uniformreal = P.UniformReal(seed=2)

    a = uniformreal((3, 4, 769, 769))
    b = np.random.randint(0, 5, size=(3, 769, 769), dtype=np.int32) - 1
    b = Tensor(b)

    #luojianet
    a_luojia = Tensor(a)
    b_luojia = Tensor(b)

    print(a_luojia)
    print(b_luojia)

    print(wce_loss_luojia(a_luojia, b_luojia))
