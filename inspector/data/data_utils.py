import secrets
import pickle

import os
from pathlib import Path

import torch
import numpy as np

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree


class LogarithmOutput(BaseTransform):
    def forward(self, data):
        data.y = torch.log(data.y)
        return data


class AddDegree(BaseTransform):
    def forward(self, data):
        deg = degree(data.edge_index[0], data.num_nodes).view(-1, 1)
        data.x = deg if data.x is None else torch.cat([data.x, deg], dim=-1)
        return data


class StandardizeOutput(BaseTransform):
    def __init__(self, mean=None, std=None):
        self.mean = mean  # set once, used for every graph
        self.std = std

    def forward(self, data):
        if self.mean is None:  # single-graph fallback only
            self.mean = torch.mean(data.y[data.train_mask], dim=0)
            self.std = torch.std(data.y[data.train_mask], dim=0)
            data.original_mean = self.mean
            data.original_std = self.std
            print(f"orig_mean = {self.mean} | orig_std = {self.std}.")

        data.y = (data.y - self.mean) / self.std
        return data


class FeatureNormalisation(BaseTransform):
    def forward(self, data):
        row_inv = torch.pow(data.x.sum(dim=1), -1)
        row_inv[torch.isinf(row_inv)] = 0
        row_mat_inv = torch.diag(row_inv)
        data.x = row_mat_inv @ (data.x)

        return data


class FeatureStandardization(BaseTransform):
    def forward(self, data):
        mean = data.x.mean(dim=0)
        std = data.x.std(dim=0)
        std[std == 0] = 1.0
        data.x = (data.x - mean) / std

        return data


class EdgeFeaturesAsNodeFeatures(BaseTransform):
    """
    Get edge weights of edges to be node features.
    Since it uses src edges only to avoid duplicates, it is only intended to be used on undirected graphs.
    """

    def forward(self, data):
        src = data.edge_index[0]
        edge_w = data.edge_weight

        if edge_w.dim() == 1:
            edge_w = edge_w.reshape(-1, 1)
        elif edge_w.dim() > 2:
            raise NotImplementedError("Do not know what to do with edge weight dim > 2")

        edge_weight_dim = edge_w.size(-1)

        num_nodes = (
            max(data.edge_index[0].max().item(), data.edge_index[1].max().item()) + 1
        )
        deg = degree(src, num_nodes).long()  # out-degree
        max_deg = deg.max().item()

        # Node-by-edge-feature matrix (zero-padded)
        node_edge_feats = torch.zeros(
            num_nodes, max_deg * edge_weight_dim, dtype=edge_w.dtype
        )

        for i in range(num_nodes):
            mask = src == i
            k = deg[i].item()
            assert k == mask.sum().item()
            if k > 0:
                node_edge_feats[i, :k] = edge_w[mask, :].ravel()

        if data.x is None:
            data.x = node_edge_feats
        else:
            data.x = torch.cat([data.x, node_edge_feats], dim=-1)

        return data


class OutputFeatureSubset(BaseTransform):
    """
    Selects a subset of the output features (data.y) in a PyG Data object.
    """

    def __init__(self, indices):
        self.indices = indices

    def forward(self, data):
        if hasattr(data, "y") and data.y is not None:
            data.y = data.y[..., self.indices]

            # if we select a single column and want shape [N] instead of [N, 1]
            # if data.y.shape[-1] == 1:
            #     data.y = data.y.squeeze(-1)

        return data


class PEMSTransform(BaseTransform):
    def forward(self, data):
        transf1, transf2 = StandardizeOutput(), EdgeFeaturesAsNodeFeatures()
        data = transf2(transf1(data))
        return data


class PEMSTransformMaskedOuputs(BaseTransform):
    def forward(self, data):
        transf1 = StandardizeOutput()
        data = transf1(data)

        masked_ys = torch.nan_to_num(data.y)
        masked_ys[data.test_mask] = 0.0
        masked_ys[data.val_mask] = 0.0

        if data.x is None:
            data.x = masked_ys
        else:
            data.x = torch.cat([data.x, masked_ys], dim=-1)

        return data


class ReduceTrain(BaseTransform):
    """
    Randomly reduce the train set to `percentage` of the original
    """

    def __init__(self, percentage, seed):
        self.percentage = percentage
        self.seed = seed

    def forward(self, data):
        rng = np.random.default_rng(self.seed)
        indeces_train = np.where(data.train_mask)[0]
        new_indeces_train = rng.choice(
            indeces_train,
            np.floor(indeces_train.shape[0] * self.percentage).astype(np.int32),
        )
        data.train_mask[:] = False
        data.train_mask[new_indeces_train] = True
        return data


def generate_splits(n, train_frac, test_frac, entropy_path=None, master_seed=None):
    if entropy_path is not None:
        _entropy_path = Path(entropy_path).expanduser().resolve()

        if _entropy_path.exists():
            with _entropy_path.open("rb") as f:
                seed = pickle.load(f)
        else:
            seed = secrets.randbits(128)
            _entropy_path.parent.mkdir(parents=True, exist_ok=True)
            with _entropy_path.open("wb") as f:
                pickle.dump(seed, f)
    else:
        assert master_seed is not None
        seed = master_seed

    rng = np.random.default_rng(seed)

    train_ind = rng.choice(n, int(n * train_frac), replace=False)
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_ind] = True

    candidates = np.setdiff1d(np.arange(n), train_ind)

    test_ind = rng.choice(candidates, int(n * test_frac), replace=False)
    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_ind] = True

    return torch.tensor(test_mask), torch.tensor(train_mask)


def reindex_mask(train_mask, test_mask, n):
    """
    This is ~O(N + M)
    All entries in a dictionary where keys are indices in 0-N and values train or test.
    Sort using the keys as the object of interest. The ordered dict has the indices in 0-M order.
    Is O(N + M log M + M)
    """
    combined_mask = train_mask | test_mask
    selected_indices = np.nonzero(combined_mask)[0]
    reindex = -np.ones(n, dtype=int)
    reindex[selected_indices] = np.arange(len(selected_indices))
    train_mask_reindexed = train_mask[selected_indices]
    test_mask_reindexed = test_mask[selected_indices]
    return torch.tensor(train_mask_reindexed), torch.tensor(test_mask_reindexed)
