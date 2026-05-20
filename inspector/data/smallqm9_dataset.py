import os
import torch
import numpy as np
import pandas as pd

from torch_geometric.data import InMemoryDataset
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from torch_geometric.datasets import QM9

torch.set_default_dtype(torch.float64)


class GapSmallQM9(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log=True,
        force_reload=False,
        used_features=[4],
    ):
        self._my_name = "GapSmallQM9"
        self.used_features = used_features

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.load(self.processed_paths[0])

        try:
            _stats = torch.load(self.processed_paths[1])
            self.original_mean = _stats["original_mean"]
            self.original_std = _stats["original_std"]
        except (AttributeError, FileNotFoundError):
            print("Error loading file, metrics will no be available.")

        _splits = torch.load(self.processed_paths[2])
        self.train_mask = _splits["train_mask"]
        self.test_mask = _splits["test_mask"]
        self.val_mask = _splits["val_mask"]

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
        return ["transformed_data.pt", "stats.pt", "splits.pt"]

    def process(self):
        qm9dataset = QM9(os.path.join(self.root, "raw", "original_qm9"))
        homolumo_gap_values = qm9dataset.y[:, 4].numpy()

        # Bin the continuous values into 10 groups (deciles)
        # Using qcut ensures each bin has roughly the same number of samples
        homolumo_gap_bins = pd.qcut(homolumo_gap_values, q=10, labels=False)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=42)
        # This returns indices for a "train" and "test" split.
        # We just take the test_index as our 50% subset.
        _, subset_indices = next(
            sss.split(np.zeros(len(homolumo_gap_bins)), homolumo_gap_bins)
        )
        sub_dataset = qm9dataset[torch.tensor(subset_indices)]
        del qm9dataset

        new_homolumo_gap_bins = pd.qcut(sub_dataset.y[:, 4].numpy(), q=10, labels=False)
        indices = np.arange(len(sub_dataset))

        train_idx, rem_idx, train_bins, rem_bins = train_test_split(
            indices,
            new_homolumo_gap_bins,
            test_size=0.25,
            stratify=new_homolumo_gap_bins,
            random_state=42,
        )

        # Split the 20% remainder into two equal 10% halves
        val_idx, test_idx = train_test_split(
            rem_idx, test_size=0.5, stratify=rem_bins, random_state=42
        )

        train_mask = torch.zeros(len(sub_dataset), dtype=torch.bool)
        test_mask = torch.zeros(len(sub_dataset), dtype=torch.bool)
        val_mask = torch.zeros(len(sub_dataset), dtype=torch.bool)
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        val_mask[val_idx] = True

        list_to_save = []
        for d in sub_dataset:
            d.y = d.y[:, self.used_features]
            list_to_save.append(d)

        if self.pre_transform is not None:
            all_y = torch.stack([d.y for d in list_to_save]).squeeze(-1)  # [N, len(used_features)]
            orig_mean = torch.mean(all_y[train_mask], dim=0)
            orig_std = torch.std(all_y[train_mask], dim=0)

            # Inject stats into the transform before looping
            if hasattr(self.pre_transform, "mean"):
                self.pre_transform.mean = orig_mean
                self.pre_transform.std = orig_std
                print(f"orig_mean = {orig_mean} | orig_std = {orig_std}.")

            list_to_save = [self.pre_transform(d) for d in list_to_save]

            torch.save(
                {
                    "original_std": orig_std,
                    "original_mean": orig_mean,
                },
                self.processed_paths[1],
            )

        torch.save(
            {
                "train_mask": train_mask,
                "test_mask": test_mask,
                "val_mask": val_mask,
            },
            self.processed_paths[2],
        )

        self.save(list_to_save, self.processed_paths[0])

    def __str__(self):
        return "{0}({1})".format(self._my_name, len(self))

    def __repr__(self):
        return "{0}({1})".format(self._my_name, len(self))
