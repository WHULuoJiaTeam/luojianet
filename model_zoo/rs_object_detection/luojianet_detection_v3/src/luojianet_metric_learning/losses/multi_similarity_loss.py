# from ..distances.cosine_similarity import CosineSimilarity  # checked by xwj
# from ..utils import common_functions as c_f  # checked by xwj
# from ..utils import loss_and_miner_utils as lmu  # logsumexp, checked by xwj
# from .generic_pair_loss import GenericPairLoss  # checked by xwj

from ..distances.cosine_similarity import CosineSimilarity  # checked by xwj
from ..utils import common_functions as c_f  # checked by xwj
from ..utils import loss_and_miner_utils as lmu  # logsumexp, checked by xwj
from ..losses.generic_pair_loss import GenericPairLoss  # checked by xwj

import luojianet_ms
import luojianet_ms.ops as P


class MultiSimilarityLoss(GenericPairLoss):
    """
    modified from https://github.com/MalongTech/research-ms-loss/
    Args:
        alpha: The exponential weight for positive pairs
        beta: The exponential weight for negative pairs
        base: The shift in the exponent applied to both positive and negative pairs
    """

    def __init__(self, alpha=2, beta=50, base=0.5, **kwargs):
        super().__init__(mat_based_loss=True, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.add_to_recordable_attributes(
            list_of_names=["alpha", "beta", "base"], is_stat=False
        )

    def _compute_loss(self, mat, pos_mask, neg_mask):
        pos_exp = self.distance.margin(mat, self.base)
        neg_exp = self.distance.margin(self.base, mat)

        # TODO: the output is not stable, the value will be inf because of float32 computing
        # pos_loss = (1.0 / self.alpha) * lmu.logsumexp(
        #     self.alpha * pos_exp, keep_mask=pos_mask.bool(), add_one=True
        # )
        # neg_loss = (1.0 / self.beta) * lmu.logsumexp(
        #     self.beta * neg_exp, keep_mask=neg_mask.bool(), add_one=True
        # )
        pos_mask_bool = P.cast(pos_mask, luojianet_ms.bool_)
        neg_mask_bool = P.cast(neg_mask, luojianet_ms.bool_)
        pos_loss = (1.0 / self.alpha) * lmu.logsumexp(
            self.alpha * pos_exp, keep_mask=pos_mask_bool, add_one=True
        )
        neg_loss = (1.0 / self.beta) * lmu.logsumexp(
            self.beta * neg_exp, keep_mask=neg_mask_bool, add_one=True
        )

        return {
            "loss": {
                "losses": pos_loss + neg_loss,
                "indices": c_f.torch_arange_from_size(mat),
                "reduction_type": "element",
            }
        }

    def get_default_distance(self):  # checked
        return CosineSimilarity()


if __name__ == "__main__":
    import luojianet_ms
    import numpy as np
    import luojianet_ms.context as context

    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')

    msloss = MultiSimilarityLoss()

    m = np.array([[1.0, 2, 3], [4, 5, 6], [0, 1, 2]])
    p = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    n = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])

    m = luojianet_ms.Tensor(m, dtype=luojianet_ms.float32)
    n = luojianet_ms.Tensor(n, dtype=luojianet_ms.float32)
    p = luojianet_ms.Tensor(p, dtype=luojianet_ms.float32)

    # output = msloss.get_default_distance()
    output = msloss._compute_loss(m, p, n)

    print(output)


