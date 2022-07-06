import luojianet_ms.ops as P
from luojianet_ms import Tensor

# from .base_distance import BaseDistance
from ..distances.base_distance import BaseDistance
import numpy as np


class DotProductSimilarity(BaseDistance):
    def __init__(self, **kwargs):
        super().__init__(is_inverted=True, **kwargs)
        assert self.is_inverted

    def compute_mat(self, query_emb, ref_emb):  # function checked by xwj
        # checked by xwj
        # return torch.matmul(query_emb, ref_emb.t())
        return P.matmul(query_emb, ref_emb.T)

    def pairwise_distance(self, query_emb, ref_emb):  # function checked by xwj
        # checked by xwj
        r = np.sum(query_emb.asnumpy() * ref_emb.asnumpy(), axis=1)
        return Tensor.from_numpy(r)
        # return torch.sum(query_emb * ref_emb, dim=1)



