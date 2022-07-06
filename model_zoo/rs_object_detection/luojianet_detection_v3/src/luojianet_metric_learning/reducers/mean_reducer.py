# import torch
import luojianet_ms.ops as P

from .base_reducer import BaseReducer


class MeanReducer(BaseReducer):
    def element_reduction(self, losses, *_):
        # check by xwj
        # return torch.mean(losses)
        return P.reduce_mean(losses)

    def pos_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def neg_pair_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)

    def triplet_reduction(self, losses, *args):
        return self.element_reduction(losses, *args)
