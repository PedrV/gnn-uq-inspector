import os
import torch
import numpy as np

from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage

from ogb.graphproppred import PygGraphPropPredDataset

torch.set_default_dtype(torch.float32)
torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])


class MolHIV(InMemoryDataset):
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
        self._my_name = "MolHIV"
        self.make_shift = make_shift

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.load(self.processed_paths[0])

        _splits = torch.load(self.processed_paths[2])
        self.train_mask = _splits["train_mask"]
        self.test_mask = _splits["test_mask"]
        self.val_mask = _splits["val_mask"]

    def download(self):
        return

    @property
    def num_classes(self):
        # OGB uses 2. We use 1 to match out dim
        return 1

    @property
    def num_nodes(self):
        return self.x.size(0)

    @property
    def num_edge_features(self):
        return self.edge_attr.size(-1)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["transformed_data.pt", "pre_transform.pt", "splits.pt", "pre_filter.pt"]

    def process(self):
        molhivdataset = PygGraphPropPredDataset(
            name="ogbg-molhiv", root=os.path.join(self.root, "raw", "original_molhiv")
        )
        molhivdataset._data.y = molhivdataset._data.y.to(
            torch.get_default_dtype()
        ).squeeze(-1)
        split_idx = molhivdataset.get_idx_split()

        list_to_save = []
        for d in molhivdataset:
            list_to_save.append(d)

        if self.pre_transform is not None:
            list_to_save = [self.pre_transform(d) for d in list_to_save]

        if self.make_shift:
            num_conjugated_bonds = []
            for data in molhivdataset:
                bond_types = data.edge_attr[:, 0]

                bond_types, bound_counts = torch.unique(
                    data.edge_attr[:, 0], return_counts=True
                )
                # Double, Triple, Aromatic
                bond_mask = (bond_types == 1) | (bond_types == 2) | (bond_types == 3)
                total_conjugated_bonds = (bound_counts[bond_mask]).sum() / 2

                # Calculate the density relative to the molecule size
                bond_density = total_conjugated_bonds / data.num_nodes
                num_conjugated_bonds.append(bond_density)

            bond_counts = np.array(num_conjugated_bonds)

            # Bottom 33% (Low conjugation density)
            low_threshold = np.percentile(bond_counts, 25)
            # Top 33% (High conjugation density)
            high_threshold = np.percentile(bond_counts, 75)

            train_mask = torch.tensor(bond_counts <= low_threshold)
            validation_mask = torch.tensor(
                (bond_counts > low_threshold) & (bond_counts < high_threshold)
            )
            test_mask = torch.tensor(bond_counts >= high_threshold)

            total_assigned = train_mask.sum() + validation_mask.sum() + test_mask.sum()
            assert total_assigned == len(molhivdataset), (
                f"Mismatch! Assigned {total_assigned} out of {len(molhivdataset)} elements."
            )

        else:
            train_mask = torch.full((len(list_to_save),), False, dtype=torch.bool)
            validation_mask = torch.full((len(list_to_save),), False, dtype=torch.bool)
            test_mask = torch.full((len(list_to_save),), False, dtype=torch.bool)

            test_mask[split_idx["test"]] = True
            train_mask[split_idx["train"]] = True
            validation_mask[split_idx["valid"]] = True

        torch.save(
            {
                "test_mask": test_mask,
                "train_mask": train_mask,
                "val_mask": validation_mask,
            },
            self.processed_paths[2],
        )

        self.save(list_to_save, self.processed_paths[0])

    def __str__(self):
        return "{0}({1})".format(self._my_name, len(self))

    def __repr__(self):
        return "{0}({1})".format(self._my_name, len(self))
