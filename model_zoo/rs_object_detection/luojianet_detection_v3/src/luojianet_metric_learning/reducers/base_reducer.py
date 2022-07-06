# import torch
import luojianet_ms
import luojianet_ms.ops as P

from ..utils import common_functions as c_f
from ..utils.module_with_records import ModuleWithRecords


class BaseReducer(ModuleWithRecords):
    def forward(self, loss_dict, embeddings, labels):
        self.reset_stats()
        # assert len(loss_dict) == 1
        loss_name = list(loss_dict.keys())[0]
        loss_info = loss_dict[loss_name]
        self.add_to_recordable_attributes(name=loss_name, is_stat=True)
        losses, loss_indices, reduction_type, kwargs = self.unpack_loss_info(loss_info)
        loss_val = self.reduce_the_loss(
            losses, loss_indices, reduction_type, kwargs, embeddings, labels
        )

        # setattr(self, loss_name, loss_val.item())
        setattr(self, loss_name, loss_val)
        return loss_val

    def unpack_loss_info(self, loss_info):
        return (
            loss_info["losses"],
            loss_info["indices"],
            loss_info["reduction_type"],
            {},
        )

    def reduce_the_loss(
        self, losses, loss_indices, reduction_type, kwargs, embeddings, labels
    ):
        self.set_losses_size_stat(losses)
        if self.input_is_zero_loss(losses):
            return self.zero_loss(embeddings)
        self.assert_sizes(losses, loss_indices, reduction_type)
        reduction_func = self.get_reduction_func(reduction_type)
        return reduction_func(losses, loss_indices, embeddings, labels, **kwargs)

    def already_reduced_reduction(self, losses, loss_indices, embeddings, labels):
        assert losses.ndim == 0 or len(losses) == 1
        return losses

    def element_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def pos_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def neg_pair_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def triplet_reduction(self, losses, loss_indices, embeddings, labels):
        raise NotImplementedError

    def get_reduction_func(self, reduction_type):
        return getattr(self, "{}_reduction".format(reduction_type))

    def assert_sizes(self, losses, loss_indices, reduction_type):
        getattr(self, "assert_sizes_{}".format(reduction_type))(losses, loss_indices)
        return 0  # add for graph mode

    def zero_loss(self, embeddings):
        # return torch.sum(embeddings * 0)
        return P.reduce_sum(embeddings * 0)

    def input_is_zero_loss(self, losses):
        # if (not torch.is_tensor(losses)) and (losses == 0):
        if (not isinstance(losses, luojianet_ms.Tensor)) and (losses == 0):
            return True
        return False

    def assert_sizes_already_reduced(self, losses, loss_indices):
        pass

    def assert_sizes_element(self, losses, loss_indices):
        # assert torch.is_tensor(losses)
        # assert torch.is_tensor(loss_indices)
        # assert len(losses) == len(loss_indices)
        assert isinstance(losses, luojianet_ms.Tensor)
        assert isinstance(loss_indices, luojianet_ms.Tensor)
        assert len(losses) == len(loss_indices)

    def assert_sizes_pair(self, losses, loss_indices):
        # assert torch.is_tensor(losses)
        assert isinstance(losses, luojianet_ms.Tensor)
        assert c_f.is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 2
        # assert all(torch.is_tensor(x) for x in loss_indices)
        assert all(isinstance(x, luojianet_ms.Tensor) for x in loss_indices)
        assert len(losses) == len(loss_indices[0]) == len(loss_indices[1])

    def assert_sizes_pos_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_neg_pair(self, losses, loss_indices):
        self.assert_sizes_pair(losses, loss_indices)

    def assert_sizes_triplet(self, losses, loss_indices):
        # assert torch.is_tensor(losses)
        assert isinstance(losses, luojianet_ms.Tensor)
        assert c_f.is_list_or_tuple(loss_indices)
        assert len(loss_indices) == 3
        assert all(len(x) == len(losses) for x in loss_indices)

    def set_losses_size_stat(self, losses):
        if self.collect_stats:
            self.add_to_recordable_attributes(name="losses_size", is_stat=True)
            # if not torch.is_tensor(losses) or losses.ndim == 0:
            if not isinstance(losses, luojianet_ms.Tensor) or losses.ndim == 0:
                self.losses_size = int(1)
                # self.losses_size = luojianet_ms.Parameter(int(1), requires_grad=False, name='losses_size_for_graph_mode')  # change for graph mode
            else:
                self.losses_size = len(losses)
                # self.losses_size = luojianet_ms.Parameter(int(len(losses)), requires_grad=False, name='losses_size_for_graph_mode')  # change for graph mode  ,

        return 0  # add for graph mode
