import torch
import numpy as np
import networkx as nx
import pandas as pd

import yaml

from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx

from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from torch_sparse import SparseTensor

torch.set_default_dtype(torch.float32)


class ArtnetViews(InMemoryDataset):

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log=True,
        force_reload=False,
        make_shift=False,
    ):
        self._my_name = "ArtnetViews"
        self.edge_list_path = f"{root}/raw/edgelist.csv"
        self.target_path = f"{root}/raw/targets.csv"
        self.feature_path = f"{root}/raw/features.csv"
        self.splits_path = f"{root}/raw/split_masks_RH.csv"
        self.info_path = f"{root}/raw/info.yaml"

        self.make_shift = make_shift
        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.load(self.processed_paths[0])

        try:
            _stats = torch.load(self.processed_paths[1])
            self.original_mean = _stats["original_mean"]
            self.original_std = _stats["original_std"]
        except (AttributeError, FileNotFoundError):
            print("Error loading file, metrics will no be available.")

    def download(self):
        return

    @property
    def num_classes(self):
        return 1

    @property
    def num_nodes(self):
        return self.x.size(0)

    @property
    def num_edge_features(self):
        return 0

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["transformed_data.pt", "stats.pt"]

    def process(self):
        # Make the graph
        df = pd.read_csv(self.edge_list_path)

        G = nx.Graph()
        tuple_edge_list = list(df.itertuples(index=False, name=None))
        # Create empty graph so that nodes are ordered upon insertion
        # This helps pyg keep a "correct" order. That is node at index 0 is node 0
        # Using the arbitrary order is also correct but the one mentioned above is more intuitive
        all_nodes = sorted(set(n for edge in tuple_edge_list for n in edge))
        G.add_nodes_from(all_nodes)
        G.add_edges_from(tuple_edge_list)

        # Add the target
        df_y = pd.read_csv(self.target_path)

        target_map = {}
        for i in range(df_y.shape[0]):
            new_key = int(df_y["node_id"][i])
            new_value = float(df_y["log_num_views_in"][i])
            if new_key in target_map:
                print(
                    f"Node {new_key} has multiple targets {target_map[new_key]}, {new_value}"
                )
            target_map[new_key] = new_value
        assert len(target_map) == G.number_of_nodes()
        nx.set_node_attributes(G, target_map, "y")

        # Add features
        with open(self.info_path, "r") as file:
            info = yaml.safe_load(file)

        fraction_features_names_set = set(info["fraction_features_names"])
        numerical_features_names = [
            name
            for name in info["numerical_features_names"]
            if name not in fraction_features_names_set
        ]

        features_df = pd.read_csv(self.feature_path, index_col=0)
        numerical_features = features_df[numerical_features_names].values.astype(
            np.float32
        )
        fraction_features = features_df[info["fraction_features_names"]].values.astype(
            np.float32
        )
        categorical_features = features_df[
            info["categorical_features_names"]
        ].values.astype(np.float32)

        one_hot_encoder = OneHotEncoder(
            drop="if_binary", sparse_output=False, dtype=np.float32
        )
        categorical_features = one_hot_encoder.fit_transform(categorical_features)

        qt = QuantileTransformer(
            output_distribution="normal", subsample=None, random_state=0, copy=False
        )
        qt_fractional = qt.fit_transform(fraction_features.copy())

        features = np.concatenate(
            [numerical_features, qt_fractional, categorical_features], axis=1
        )
        feature_map = {
            i: torch.tensor(features[i, :]) for i in range(G.number_of_nodes())
        }
        nx.set_node_attributes(G, feature_map, "x")

        pyg_data = from_networkx(G)

        pyg_data.adj = SparseTensor(
            row=pyg_data.edge_index[0],
            col=pyg_data.edge_index[1],
            sparse_sizes=(pyg_data.x.shape[0], pyg_data.x.shape[0]),
        )

        assert torch.all(~torch.isnan(pyg_data.y))
        assert pyg_data.y.ndim == 1
        pyg_data.y = pyg_data.y.unsqueeze(-1)
        assert pyg_data.y.shape == torch.Size([G.number_of_nodes(), 1])

        split_masks_df = pd.read_csv(self.splits_path, index_col=0)

        pyg_data.train_mask = torch.tensor(split_masks_df["train"].values)
        pyg_data.test_mask = torch.tensor(split_masks_df["test"].values)
        pyg_data.val_mask = torch.tensor(split_masks_df["val"].values)

        if self.make_shift:
            num_train = pyg_data.train_mask.sum().item()
            num_test = pyg_data.test_mask.sum().item()
            num_val = pyg_data.val_mask.sum().item()

            clustering_coef = nx.clustering(G)

            sorted_clustering_coef = dict(
                sorted(clustering_coef.items(), key=lambda item: item[1], reverse=True)
            )
            list_sorted_clustering_coef = list(sorted_clustering_coef.items())

            test_indices = list(
                map(lambda x: x[0], list_sorted_clustering_coef[:num_test])
            )
            val_indices = list(
                map(
                    lambda x: x[0],
                    list_sorted_clustering_coef[num_test : (num_test + num_val)],
                )
            )
            train_indices = list(
                map(
                    lambda x: x[0],
                    list_sorted_clustering_coef[
                        (num_test + num_val) : (num_test + num_val + num_train)
                    ],
                )
            )

            assert len(train_indices) == num_train
            assert len(val_indices) == num_val
            assert len(test_indices) == num_test

            test_mask = torch.zeros(num_train + num_val + num_test, dtype=torch.bool)
            test_mask[test_indices] = True
            val_mask = torch.zeros(num_train + num_val + num_test, dtype=torch.bool)
            val_mask[val_indices] = True
            train_mask = torch.zeros(num_train + num_val + num_test, dtype=torch.bool)
            train_mask[train_indices] = True

            pyg_data.train_mask = train_mask
            pyg_data.val_mask = val_mask
            pyg_data.test_mask = test_mask

            assert torch.equal(
                torch.where(pyg_data.train_mask)[0],
                torch.sort(torch.tensor(train_indices)).values,
            )
            assert torch.equal(
                torch.where(pyg_data.val_mask)[0],
                torch.sort(torch.tensor(val_indices)).values,
            )
            assert torch.equal(
                torch.where(pyg_data.test_mask)[0],
                torch.sort(torch.tensor(test_indices)).values,
            )

        if self.pre_transform is not None:
            pyg_data = self.pre_transform(pyg_data)
            torch.save(
                {
                    "original_std": pyg_data.original_std,
                    "original_mean": pyg_data.original_mean,
                },
                self.processed_paths[1],
            )

        self.save([pyg_data], self.processed_paths[0])

    def __str__(self):
        return "{0}({1})".format(self._my_name, len(self))

    def __repr__(self):
        return "{0}({1})".format(self._my_name, len(self))
