import torch, os
import numpy as np
import networkx as nx

import pickle

from torch_geometric.data import Data, InMemoryDataset

torch.set_default_dtype(torch.float64)

NUM_TRAIN = 250


class PEMS(InMemoryDataset):

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log=True,
        force_reload=False,
        notebook_splits=True,
    ):
        self._my_name = "PEMSRegression"
        self.notebook_splits = notebook_splits

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.load(self.processed_paths[0])

        try:
            _stats = torch.load(self.processed_paths[1])
            self.original_mean = _stats["original_mean"]
            self.original_std = _stats["original_std"]
            self.notebook_splits = _stats["notebook_splits"]
        except (AttributeError, FileNotFoundError):
            print("Error loading file, metrics will no be available.")

    def download(self):
        self._raw_data_path = "processed_pems_data.pkl"
        return

    @property
    def num_classes(self):
        return 1

    @property
    def num_nodes(self):
        return self.x.size(0)

    @property
    def num_edge_features(self):
        return 1

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["transformed_data.pt", "stats.pt"]

    def process(self):
        with open(self._raw_data_path, "rb") as f:
            nx_graph, data = pickle.load(f)
        # Splitting data into train and test

        # data[0] contains the index of the labels that are valid (non NaN)
        random_perm = np.random.permutation(np.arange(data[0].shape[0]))
        num_val = int(NUM_TRAIN * 0.3)
        num_train_final = NUM_TRAIN - num_val

        train_vertex = random_perm[:num_train_final]
        validation_vertex = random_perm[num_train_final : num_train_final + num_val]
        test_vertex = random_perm[NUM_TRAIN:]

        xs_train = torch.tensor(
            data[0][train_vertex], dtype=torch.int64
        )  # For index purposes only, not real data
        ys_train = torch.tensor(data[1][train_vertex], dtype=torch.float64)
        
        xs_validation = torch.tensor(
            data[0][validation_vertex], dtype=torch.int64
        )  # For index purposes only, not real data
        ys_validation = torch.tensor(data[1][validation_vertex], dtype=torch.float64)

        xs_test = torch.tensor(
            data[0][test_vertex], dtype=torch.int64
        )  # For index purposes only, not real data
        ys_test = torch.tensor(data[1][test_vertex], dtype=torch.float64)

        num_nodes = len(nx_graph)

        # Note: since there is lots of _actually_ unknown y-s, the `ys` array will have
        # lots of NaN-s.
        ys = torch.full((num_nodes,), np.nan, dtype=torch.float64)
        ys[xs_train.squeeze()] = ys_train.squeeze()
        ys[xs_validation.squeeze()] = ys_validation.squeeze()
        ys[xs_test.squeeze()] = ys_test.squeeze()
        ys = ys[:, None]

        # ys, orig_mean, orig_std = self.pre_transform(ys, ys_train)
        adj_mat = torch.tensor(nx.to_numpy_array(nx_graph))

        # Sanity checks
        assert torch.allclose(adj_mat, adj_mat.T), "Adjacency matrix is not symmetric!"
        assert torch.all(adj_mat >= 0.0), "Adjacency matrix contains negative elements!"
        assert (
            torch.sum(adj_mat.diagonal() ** 2) == 0.0
        ), "Adjacency matrix has non-zeros on diagonal!"

        # xs = None
        # if self.labels_as_features:
        #     xs = torch.nan_to_num(ys)
        #     xs[xs_test.squeeze()] = 0.0
        #     if not self.notebook_splits:
        #         xs[xs_validation.squeeze()] = 0.0

        train_mask = torch.full((num_nodes,), False, dtype=torch.bool)
        validation_mask = torch.full((num_nodes,), False, dtype=torch.bool)
        test_mask = torch.full((num_nodes,), False, dtype=torch.bool)

        test_mask[xs_test.squeeze()] = True
        train_mask[xs_train.squeeze()] = True

        if self.notebook_splits:
            train_mask[xs_validation.squeeze()] = True
        else:
            validation_mask[xs_validation.squeeze()] = True

        # train_ind = np.where(train_mask.numpy())[0]
        # train_masks_size = int(np.ceil(train_ind.size*0.8))
        # train_masks = torch.zeros((10,train_mask.size()[0]), dtype=bool)
        # for i in range(10):
        #     _indx = np.sort(rng.choice(train_ind, train_masks_size, replace=False))
        #     train_masks[i,_indx] = True

        pyg_data = Data(
            x=None,
            edge_index=adj_mat.to_sparse().indices(),
            edge_weight=adj_mat.to_sparse().values(),
            y=ys,
            train_mask=train_mask,
            test_mask=test_mask,
            val_mask=validation_mask,
        )

        if self.pre_transform is not None:
            pyg_data = self.pre_transform(pyg_data)
            torch.save(
                {
                    "original_std": pyg_data.original_std,
                    "original_mean": pyg_data.original_mean,
                    "notebook_splits": self.notebook_splits, 
                },
                self.processed_paths[1],
            )

        self.save([pyg_data], self.processed_paths[0])

    def __str__(self):
        return "{0}({1})".format(self._my_name, len(self))

    def __repr__(self):
        return "{0}({1})".format(self._my_name, len(self))
