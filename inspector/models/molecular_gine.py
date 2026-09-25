import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GINEConv, global_add_pool, JumpingKnowledge
from torch_geometric.nn.models import MLP as PygMLP

from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from torch_geometric.nn.resolver import activation_resolver

from .registry import register_model


@register_model("molginegraph", task_type="graph")
class MolecularGINE(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers,
        out_dim,
        act="relu",
        jk=None,
        dropout=0,
        gine_mlp_num_layers=2,
        return_hidden_outputs=False,
    ):
        super().__init__()

        self.dropout = dropout
        self.jk = jk
        self.return_hidden_outputs = return_hidden_outputs

        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()

        for layer_idx in range(num_layers):
            mlp = PygMLP(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                num_layers=gine_mlp_num_layers,
                norm="batch_norm",
            )
            self.convs.append(GINEConv(mlp))

            if layer_idx < num_layers - 1:
                self.norms.append(nn.BatchNorm1d(hidden_dim))
                self.activations.append(activation_resolver(act))

        if self.jk is not None:
            self.layers.append(JumpingKnowledge(mode=jk))

            if self.jk == "cat":
                # +1 comes from the atom encoder
                full_dim = (num_layers + 1) * hidden_dim
            else:
                full_dim = hidden_dim

        else:
            full_dim = hidden_dim

        self.lin_pred = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(full_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    @classmethod
    def from_cfg(cls, cfg, in_dim, out_dim):
        return cls(
            cfg.model.hidden_dim,
            cfg.model.num_layers,
            out_dim,
            act=cfg.model.act,
            jk=cfg.model.jk,
            dropout=cfg.model.dropout,
            gine_mlp_num_layers=cfg.model.gine_mlp_num_layers,
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        h = self.atom_encoder(x.type(torch.int))
        edge_emb = self.bond_encoder(edge_attr)

        hidden_reps = []
        if self.jk is not None:
            hidden_reps.append(global_add_pool(h, batch))

        for layer_idx in range(len(self.convs)):
            h = self.convs[layer_idx](h, edge_index, edge_attr=edge_emb)
            if self.jk is not None:
                hidden_reps.append(global_add_pool(h, batch))

            if layer_idx < len(self.norms):
                h = self.norms[layer_idx](h)
                h = self.activations[layer_idx](h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        if self.jk is not None:
            h = self.jk(hidden_reps)
        else:
            h = global_add_pool(h, batch)

        hidden_reps = (
            [e.detach() for e in hidden_reps] if self.return_hidden_outputs else None
        )

        out = self.lin_pred(h)

        return out, hidden_reps

    def reset_parameters(self):
        pass
