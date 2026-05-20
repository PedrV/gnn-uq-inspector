import torch
import numpy as np
import networkx as nx
import pandas as pd

import json

from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_networkx

from sklearn.preprocessing import MultiLabelBinarizer

torch.set_default_dtype(torch.float64)


class Chameleon(InMemoryDataset):

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log=True,
        force_reload=False,
        version="pad",
    ):
        self._my_name = "Chameleon"
        self.edge_list_path = f"{root}/raw/musae_chameleon_edges.csv"
        self.target_path = f"{root}/raw/musae_chameleon_target.csv"
        self.feature_path = f"{root}/raw/musae_chameleon_features.json"
        self.split_path = f"{root}/raw/chameleon_splits.pt"

        self.version = version
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
        df = pd.read_csv(self.edge_list_path)

        # G = nx.from_pandas_edgelist(df, source='id1', target='id2', create_using=nx.Graph())
        G = nx.Graph()
        tuple_edge_list = list(df.itertuples(index=False, name=None))

        # Create empty graph so that nodes are ordered upon insertion
        # This helps pyg keep a "correct" order. That is node at index 0 is node 0
        # Using the arbitrary order is also correct but the one mentioned above is more intuitive
        all_nodes = sorted(set(n for edge in tuple_edge_list for n in edge))
        G.add_nodes_from(all_nodes)
        G.add_edges_from(tuple_edge_list)

        df_y = pd.read_csv(self.target_path)

        target_map = {}
        for i in range(df_y.shape[0]):
            new_key = int(df_y["id"][i])
            new_value = df_y["target"][i].astype(np.float64)
            if new_key in target_map:
                print(
                    f"Node {new_key} has multiple targets {target_map[new_key]}, {new_value}"
                )
            target_map[new_key] = new_value
        assert len(target_map) == G.number_of_nodes()
        nx.set_node_attributes(G, target_map, "y")

        with open(self.feature_path, "r") as f:
            data = json.load(f)

        max_len_features = 0
        max_category_features = 0
        for k in data.keys():
            max_len_features = max(max_len_features, len(data[k]))
            max_category_features = max(max_category_features, max(data[k]))

        mlb = MultiLabelBinarizer()
        mlb.fit(np.arange(max_category_features + 1).reshape(-1, 1))

        feature_map = {}
        for k in sorted(data.keys()):
            new_key = int(k)
            new_value = np.array(data[k], dtype=np.float64)
            if new_key in feature_map:
                print(
                    f"Node {new_key} has multiple feature entries {feature_map[new_key]}, {new_value}"
                )
            if self.version == "pad":
                feature_map[new_key] = torch.from_numpy(
                    np.pad(
                        np.array(new_value),
                        (0, max_len_features - len(new_value)),
                        mode="constant",
                    )
                )
            elif self.version == "binary":
                feature_map[new_key] = torch.from_numpy(
                    mlb.transform(new_value.reshape(1, -1))
                ).squeeze(0)

        assert len(feature_map) == G.number_of_nodes()
        nx.set_node_attributes(G, feature_map, "x")

        pyg_data = from_networkx(G)
        assert pyg_data.y.ndim == 1
        pyg_data.y = pyg_data.y.unsqueeze(-1)
        assert pyg_data.y.shape == torch.Size([G.number_of_nodes(), 1])

        splits = torch.load(self.split_path)
        pyg_data.train_mask = splits["train"]
        pyg_data.test_mask = splits["test"]
        pyg_data.val_mask = splits["val"]

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
