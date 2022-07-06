import luojianet_ms
import luojianet_ms.nn as nn
from luojianet_ms.ops import functional as F
import numpy as np

# from ..utils.module_with_records import ModuleWithRecords
from ..utils.module_with_records import ModuleWithRecords


class BaseDistance(ModuleWithRecords):
    def __init__(
        self, normalize_embeddings=True, p=2, power=1, is_inverted=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.normalize_embeddings = normalize_embeddings
        self.p = p
        self.power = power
        self.is_inverted = is_inverted
        self.add_to_recordable_attributes(list_of_names=["p", "power"], is_stat=False)

    # def forward(self, query_emb, ref_emb=None):
    def forward(self, query_emb, ref_emb=None):
        self.reset_stats()
        query_emb_normalized = self.maybe_normalize(query_emb)
        if ref_emb is None:
            ref_emb = query_emb
            ref_emb_normalized = query_emb_normalized
        else:
            ref_emb_normalized = self.maybe_normalize(ref_emb)
        self.set_default_stats(
            query_emb, ref_emb, query_emb_normalized, ref_emb_normalized
        )
        mat = self.compute_mat(query_emb_normalized, ref_emb_normalized)
        if self.power != 1:
            mat = mat ** self.power
        # assert mat.size() == torch.Size((query_emb.size(0), ref_emb.size(0)))
        # assert mat.size() == np.array((query_emb.size(0), ref_emb.size(0)))
        return mat

    def compute_mat(self, query_emb, ref_emb):
        raise NotImplementedError

    def pairwise_distance(self, query_emb, ref_emb):
        raise NotImplementedError

    def smallest_dist(self, *args, **kwargs):
        if self.is_inverted:
            # check by xwj, need further check
            # return torch.max(*args, **kwargs)
            r = np.max(*args.asnumpy(), **kwargs)
            r = luojianet_ms.Tensor.from_numpy(r)
            return r
            # luojianet_ms.ops.ArgMaxWithValue(**kwargs)(*args)[1]
        # return torch.min(*args, **kwargs)
        r = np.min(*args.asnumpy(), **kwargs)
        r = luojianet_ms.Tensor.from_numpy(r)
        return r
        # return luojianet_ms.ops.ArgMinWithValue(**kwargs)(*args)[1]

    def largest_dist(self, *args, **kwargs):
        if self.is_inverted:
            # need further check by xwj
            # return torch.min(*args, **kwargs)
            # return luojianet_ms.ops.ArgMinWithValue(**kwargs)(*args)[1]
            r = np.min(*args.asnumpy(), **kwargs)
            r = luojianet_ms.Tensor.from_numpy(r)
            return r
        # return torch.max(*args, **kwargs)
        r = np.max(*args.asnumpy(), **kwargs)
        r = luojianet_ms.Tensor.from_numpy(r)
        return r
        # return luojianet_ms.ops.ArgMaxWithValue(**kwargs)(*args)[1]

    # This measures the margin between x and y
    def margin(self, x, y):
        if self.is_inverted:
            return y - x
        return x - y

    def normalize(self, embeddings, dim=1, **kwargs):
        # checked by xwj
        # return torch.nn.functional.normalize(embeddings, p=self.p, dim=dim, **kwargs)
        return luojianet_ms.ops.L2Normalize(axis=dim)(embeddings)

    def maybe_normalize(self, embeddings, dim=1, **kwargs):
        if self.normalize_embeddings:
            return self.normalize(embeddings, dim=dim, **kwargs)
        return embeddings

    def get_norm(self, embeddings, dim=1, **kwargs):
        # checked by xwj
        # return torch.norm(embeddings, p=self.p, dim=dim, **kwargs)
        return nn.Norm(axis=dim)(embeddings)

    def set_default_stats(
        self, query_emb, ref_emb, query_emb_normalized, ref_emb_normalized
    ):
        if self.collect_stats:
            # with torch.no_grad():
            #     stats_dict = {
            #         "initial_avg_query_norm": torch.mean(
            #             self.get_norm(query_emb)
            #         ).item(),
            #         "initial_avg_ref_norm": torch.mean(self.get_norm(ref_emb)).item(), #
            #         "final_avg_query_norm": torch.mean(  # torch.mean
            #             self.get_norm(query_emb_normalized)
            #         ).item(),
            #         "final_avg_ref_norm": torch.mean(  # torch.mean
            #             self.get_norm(ref_emb_normalized)
            #         ).item(),
            #     }
            #     self.set_stats(stats_dict)

            # { add by xwj
            stats_dict = {
                "initial_avg_query_norm": luojianet_ms.ops.ReduceMean()(  # torch.mean
                    self.get_norm(query_emb)),
                # ).item(),
                "initial_avg_ref_norm": luojianet_ms.ops.ReduceMean()(self.get_norm(ref_emb)), # .item(),  # torch.mean
                "final_avg_query_norm": luojianet_ms.ops.ReduceMean()(
                    self.get_norm(query_emb_normalized)),
                # ).item(),
                "final_avg_ref_norm": luojianet_ms.ops.ReduceMean()(  # torch.mean
                    self.get_norm(ref_emb_normalized)),  # .item() convert tensor to a number
                # ).item(),
            }

            # need further check by xwj
            for k, v in stats_dict.items():
                stats_dict[k] = F.stop_gradient(v)

            self.set_stats(stats_dict)
            # end }
        return 0  # add for graph mode

    def set_stats(self, stats_dict):
        for k, v in stats_dict.items():
            self.add_to_recordable_attributes(name=k, is_stat=True)
            setattr(self, k, v)
