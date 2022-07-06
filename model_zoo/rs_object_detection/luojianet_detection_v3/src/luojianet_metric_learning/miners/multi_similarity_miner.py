# import torch
import numpy as np
import luojianet_ms
import luojianet_ms.ops as P

from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from ..miners.base_miner import BaseTupleMiner


class MultiSimilarityMiner(BaseTupleMiner):
    def __init__(self, epsilon=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.add_to_recordable_attributes(name="epsilon", is_stat=False)

    def mine(self, embeddings, labels, ref_emb, ref_labels):
        mat = self.distance(embeddings, ref_emb)
        a1, p, a2, n = lmu.get_all_pairs_indices(labels, ref_labels)

        if len(a1) == 0 or len(a2) == 0:
            # empty = torch.tensor([], device=labels.device, dtype=torch.long)
            empty = luojianet_ms.Tensor([], dtype=luojianet_ms.int64)
            # checked by xwj, only Parameter support clone()
            # return empty.clone(), empty.clone(), empty.clone(), empty.clone()
            return empty, empty, empty, empty

        mat_neg_sorting = mat
        # mat_pos_sorting = mat.clone()  # support graph retrace
        mat_pos_sorting = mat.copy()

        dtype = mat.dtype
        pos_ignore = (
            c_f.pos_inf(dtype) if self.distance.is_inverted else c_f.neg_inf(dtype)
        )
        neg_ignore = (
            c_f.neg_inf(dtype) if self.distance.is_inverted else c_f.pos_inf(dtype)
        )

        mat_pos_sorting[a2, n] = pos_ignore
        mat_neg_sorting[a1, p] = neg_ignore
        if embeddings is ref_emb:
            ## checked pytorch implementation
            # mat_pos_sorting.fill_diagonal_(pos_ignore)
            # mat_neg_sorting.fill_diagonal_(neg_ignore)
            ## luojianet implementation v1
            # mat_pos_sorting = mat_pos_sorting.asnumpy()
            # np.fill_diagonal(mat_pos_sorting, pos_ignore)
            # mat_pos_sorting = luojianet_ms.Tensor(mat_pos_sorting)
            #
            # mat_neg_sorting = mat_neg_sorting.asnumpy()
            # np.fill_diagonal(mat_neg_sorting, neg_ignore)
            # mat_neg_sorting = luojianet_ms.Tensor(mat_neg_sorting)

            # luojianet implementation v2, checked
            mat_pos_sorting = P.cast(mat_pos_sorting, luojianet_ms.float32)
            diag_mask = P.Eye()(*mat_pos_sorting.shape, luojianet_ms.float32)
            zeros = P.zeros_like(mat_pos_sorting) + pos_ignore
            zeros = P.cast(zeros, luojianet_ms.float32)
            mat_pos_sorting = P.Select()(~P.cast(diag_mask, luojianet_ms.bool_), mat_pos_sorting, zeros)

            mat_neg_sorting = P.cast(mat_neg_sorting, luojianet_ms.float32)
            diag_mask = P.Eye()(*mat_neg_sorting.shape, luojianet_ms.float32)
            zeros = P.zeros_like(mat_neg_sorting) + neg_ignore
            zeros = P.cast(zeros, luojianet_ms.float32)
            mat_neg_sorting = P.Select()(~P.cast(diag_mask, luojianet_ms.bool_), mat_neg_sorting, zeros)

        # sort checked by xwj
        # pos_sorted, pos_sorted_idx = torch.sort(mat_pos_sorting, dim=1)
        # neg_sorted, neg_sorted_idx = torch.sort(mat_neg_sorting, dim=1)
        # sort = luojianet_ms.ops.Sort(axis=1, descending=False)
        mat_pos_sorting_n = mat_pos_sorting.asnumpy()
        mat_neg_sorting_n = mat_neg_sorting.asnumpy()
        pos_sorted = np.sort(mat_pos_sorting_n, axis=1)
        pos_sorted_idx = np.argsort(mat_pos_sorting_n, axis=1)
        neg_sorted = np.sort(mat_neg_sorting_n, axis=1)
        neg_sorted_idx = np.argsort(mat_neg_sorting_n, axis=1)

        if self.distance.is_inverted:
            hard_pos_idx = np.where(
                pos_sorted - self.epsilon < np.expand_dims(neg_sorted[:, -1], axis=1)
            )
            hard_neg_idx = np.where(
                neg_sorted + self.epsilon > np.expand_dims(pos_sorted[:, 0], axis=1)
            )
            # hard_pos_idx = np.where(
            #     pos_sorted.asnumpy() - self.epsilon < P.expand_dims(neg_sorted[:, -1], 1).asnumpy()
            # )
            # hard_neg_idx = np.where(
            #     neg_sorted.asnumpy() + self.epsilon > P.expand_dims(pos_sorted[:, 0], 1).asnumpy()
            # )

        else:
            hard_pos_idx = np.where(
                pos_sorted + self.epsilon > np.expand_dims(neg_sorted[:, -1], axis=1)
            )
            hard_neg_idx = np.where(
                neg_sorted - self.epsilon < np.expand_dims(pos_sorted[:, 0], axis=1)
            )
            # hard_pos_idx = np.where(
            #     pos_sorted.asnumpy() + self.epsilon > neg_sorted[:, 0].asnumpy().unsqueeze(1)
            # )
            # hard_neg_idx = np.where(
            #     neg_sorted.asnumpy() - self.epsilon < pos_sorted[:, -1].asnumpy().unsqueeze(1)
            # )

        a1 = hard_pos_idx[0]
        p = pos_sorted_idx[a1, hard_pos_idx[1]]
        # p = pos_sorted_idx.asnumpy()[a1, hard_pos_idx[1]]
        a2 = hard_neg_idx[0]
        n = neg_sorted_idx[a2, hard_neg_idx[1]]
        # n = neg_sorted_idx.asnumpy()[a2, hard_neg_idx[1]]
        return a1, p, a2, n

    def get_default_distance(self):
        return CosineSimilarity()


if __name__ == "__main__":
    from luojianet_ms import context
    import os

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=int(os.getenv('DEVICE_ID', '3')))

    a = MultiSimilarityMiner(epsilon=0.1)


