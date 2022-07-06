import copy

import numpy as np
# import torch
import luojianet_ms
import luojianet_ms.ops.functional as F
import luojianet_ms.ops as P
import luojianet_ms.nn as nn

from ..distances import CosineSimilarity
import src.luojianet_metric_learning.utils.common_functions as c_f


class MatchFinder:
    def __init__(self, distance, threshold=None):
        self.distance = distance
        self.threshold = threshold

    def operate_on_emb(self, input_func, query_emb, ref_emb=None, *args, **kwargs):
        if ref_emb is None:
            ref_emb = query_emb
        return input_func(query_emb, ref_emb, *args, **kwargs)

    # for a batch of queries
    def get_matching_pairs(
        self, query_emb, ref_emb=None, threshold=None, return_tuples=False
    ):
        # with torch.no_grad():
        #     threshold = threshold if threshold is not None else self.threshold
        #     return self.operate_on_emb(
        #         self._get_matching_pairs, query_emb, ref_emb, threshold, return_tuples
        #     )
        threshold = threshold if threshold is not None else self.threshold
        F.stop_gradient(threshold)
        output = self.operate_on_emb(self._get_matching_pairs, query_emb, ref_emb, threshold, return_tuples)
        F.stop_gradient(output)
        return output

    def _get_matching_pairs(self, query_emb, ref_emb, threshold, return_tuples):
        mat = self.distance(query_emb, ref_emb)
        matches = mat >= threshold if self.distance.is_inverted else mat <= threshold
        # matches = matches.cpu().numpy()
        matches = matches.asnumpy()
        if return_tuples:
            return list(zip(*np.where(matches)))
        return matches

    # where x and y are already matched pairs
    def is_match(self, x, y, threshold=None):
        threshold = threshold if threshold is not None else self.threshold
        # with torch.no_grad():
        #     dist = self.distance.pairwise_distance(x, y)
        #     output = (
        #         dist >= threshold if self.distance.is_inverted else dist <= threshold
        #     )
        #     if output.nelement() == 1:
        #         return output.detach().item()
        #     return output.cpu().numpy()
        dist = self.distance.pairwise_distance(x, y)
        F.stop_gradient(dist)
        output = (
            dist >= threshold if self.distance.is_inverted else dist <= threshold
        )
        F.stop_gradient(output)
        if output.size == 1:
            return output
        return output.asnumpy()


class FaissIndexer:
    def __init__(self, index=None, emb_dim=None):
        import faiss as faiss_module

        self.faiss_module = faiss_module
        self.index = index
        self.emb_dim = emb_dim

    def train_index(self, vectors):
        self.emb_dim = len(vectors[0])
        self.index = self.faiss_module.IndexFlatL2(self.emb_dim)
        self.index.add(vectors)

    def search_nn(self, query_batch, k):
        D, I = self.index.search(query_batch, k)
        return I, D


class InferenceModel:
    def __init__(
        self,
        trunk,
        embedder=None,
        match_finder=None,
        normalize_embeddings=True,
        indexer=None,
        batch_size=64,
    ):
        self.trunk = trunk
        self.embedder = c_f.Identity() if embedder is None else embedder
        self.match_finder = (
            MatchFinder(distance=CosineSimilarity(), threshold=0.9)
            if match_finder is None
            else match_finder
        )
        self.indexer = FaissIndexer() if indexer is None else indexer
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size

    def train_indexer(self, tensors, emb_dim):
        if isinstance(tensors, list):
            # tensors = torch.stack(tensors)
            tensors = P.stack(tensors)

        # embeddings = torch.Tensor(len(tensors), emb_dim)
        embeddings = P.Zeros()((len(tensors), emb_dim), luojianet_ms.float32)
        for i in range(0, len(tensors), self.batch_size):
            embeddings[i : i + self.batch_size] = self.get_embeddings(
                tensors[i : i + self.batch_size], None
            )[0]

        # self.indexer.train_index(embeddings.cpu().numpy())
        self.indexer.train_index(embeddings.asnumpy())

    def get_nearest_neighbors(self, query, k):
        if not self.indexer.index or not self.indexer.index.is_trained:
            raise RuntimeError("Index must be trained by running `train_indexer`")

        query_emb, _ = self.get_embeddings(query, None)

        # indices, distances = self.indexer.search_nn(query_emb.cpu().numpy(), k)
        indices, distances = self.indexer.search_nn(query_emb.asnumpy(), k)
        return indices, distances

    def get_embeddings(self, query, ref):
        if isinstance(query, list):
            # query = torch.stack(query)
            query = P.stack(query)

        # self.trunk.eval()
        # self.embedder.eval()
        self.trunk.set_train(False)
        self.embedder.set_train(False)

        # with torch.no_grad():
        #     query_emb = self.embedder(self.trunk(query))
        #     ref_emb = query_emb if ref is None else self.embedder(self.trunk(ref))
        query_emb = self.embedder(self.trunk(query))
        F.stop_gradient(query_emb)
        ref_emb = query_emb if ref is None else self.embedder(self.trunk(ref))
        F.stop_gradient(ref_emb)

        if self.normalize_embeddings:
            # query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1)
            # ref_emb = torch.nn.functional.normalize(ref_emb, p=2, dim=1)
            query_emb = luojianet_ms.ops.L2Normalize(axis=1)(query_emb)
            ref_emb = luojianet_ms.ops.L2Normalize(axis=1)(ref_emb)
        return query_emb, ref_emb

    # for a batch of queries
    def get_matches(self, query, ref=None, threshold=None, return_tuples=False):
        query_emb, ref_emb = self.get_embeddings(query, ref)
        return self.match_finder.get_matching_pairs(
            query_emb, ref_emb, threshold, return_tuples
        )

    # where x and y are already matched pairs
    def is_match(self, x, y, threshold=None):
        x, y = self.get_embeddings(x, y)
        return self.match_finder.is_match(x, y, threshold)


class LogitGetter(nn.Module):
    possible_layer_names = ["fc", "proxies", "W"]

    def __init__(
        self,
        classifier,
        layer_name=None,
        transpose=None,
        distance=None,
        copy_weights=True,
    ):
        super().__init__()
        self.copy_weights = copy_weights
        ### set layer weights ###
        if layer_name is not None:
            self.set_weights(getattr(classifier, layer_name))
        else:
            for x in self.possible_layer_names:
                layer = getattr(classifier, x, None)
                if layer is not None:
                    self.set_weights(layer)
                    break

        ### set distance measure ###
        self.distance = classifier.distance if distance is None else distance
        self.transpose = transpose

    def forward(self, embeddings):
        w = self.weights
        if self.transpose is True:
            # w = w.t()
            w = w.T
        elif self.transpose is None:
            # if w.size(0) == embeddings.size(1):
            #     w = w.t()
            if w.shape[0] == embeddings.shape[0]:
                w = w.T
        return self.distance(embeddings, w)

    def set_weights(self, layer):
        self.weights = copy.deepcopy(layer) if self.copy_weights else layer
