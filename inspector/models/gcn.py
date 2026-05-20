import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch_geometric.nn import GCNConv, MultiAggregation, JumpingKnowledge

from .registry import register_model

ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "prelu": nn.PReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "softmax": nn.Softmax,  # requires dim argument when initializing
    "logsoftmax": nn.LogSoftmax,
}


def get_activation(name: str, **kwargs) -> nn.Module:
    """
    Returns a PyTorch activation module given a string.
    Additional kwargs are passed to the activation constructor (like dim for Softmax).
    """
    name = name.lower()
    if name in ACTIVATION_MAP:
        return ACTIVATION_MAP[name](**kwargs)
    else:
        raise ValueError(
            f"Activation '{name}' not recognized. Available: {list(ACTIVATION_MAP.keys())}"
        )


def build_regressor(hidden_dim, out_dim, num_end_layers):
    layers = OrderedDict()
    curr_dim = hidden_dim * 3

    for i in range(1, num_end_layers + 1):
        if i == num_end_layers:
            next_dim = out_dim
            layers[f"lin{i}"] = nn.Linear(curr_dim, next_dim)
        else:
            # progressively halve
            # next_dim = max(out_dim + 1, curr_dim // 2)
            next_dim = 256
            layers[f"lin{i}"] = nn.Linear(curr_dim, next_dim)
            layers[f"relu{i}"] = nn.ReLU()

        curr_dim = next_dim

    return nn.Sequential(layers)


@register_model("gcn", task_type="node")
class GCNModel(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers,
        out_dim,
        act="relu",
        jk=None,
        return_hidden_outputs=False,
        dropout=0,
        force_save_last=False,
        **kwargs,
    ):
        super().__init__()
        self.return_hidden_outputs = return_hidden_outputs
        self.force_save_last = force_save_last

        assert num_layers >= 1
        self.layers = torch.nn.ModuleList([GCNConv(in_dim, hidden_dim, **kwargs)])
        self.jk = jk
        self.dropout = dropout

        self.act = get_activation(act)

        # If no jk, input and output are added separately.
        # Otherwise, jk will be the last layer and only input added separately
        num_hidden_layers = num_layers - 2 if self.jk is None else num_layers - 1
        for _ in range(0, num_hidden_layers):
            self.layers.append(GCNConv(hidden_dim, hidden_dim, **kwargs))

        if self.jk is not None:
            self.layers.append(JumpingKnowledge(mode=jk))

            if self.jk == "cat":
                lin_input_dim = num_layers * hidden_dim
            else:
                lin_input_dim = hidden_dim

            self.layers.append(torch.nn.Linear(lin_input_dim, out_dim))
        elif num_layers >= 2:
            self.layers.append(GCNConv(hidden_dim, out_dim, **kwargs))

    @classmethod
    def from_cfg(cls, cfg, in_dim, out_dim):
        return cls(
            in_dim,
            cfg.model.hidden_dim,
            cfg.model.num_layers,
            out_dim,
            act=cfg.model.act,
            jk=cfg.model.jk,
            **{"add_self_loops": True, "normalize": True, "bias": True},
        )

    def forward(self, x, edge_index, edge_weight=None):
        emb = []

        num_hidden_layers = (
            len(self.layers) - 1 if self.jk is None else len(self.layers) - 2
        )
        for i in range(num_hidden_layers):
            x = self.layers[i](x, edge_index, edge_weight)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk or self.return_hidden_outputs:
                emb.append(x)

        if self.jk is None:
            out = self.layers[-1](x, edge_index, edge_weight)
        else:
            # Apply Jumping Knowledge to the list of representations
            x_jk = self.layers[-2](emb)
            out = self.layers[-1](x_jk)

        hidden_reps = [e.detach() for e in emb] if self.return_hidden_outputs else None

        if self.return_hidden_outputs and self.force_save_last:
            hidden_reps.append(out.detach())

        return (out, hidden_reps)

    def reset_parameters(self):
        pass


@register_model("gcngraph", task_type="graph")
class GCNModelGraphWide(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers,
        out_dim,
        act="relu",
        jk=None,
        return_hidden_outputs=False,
        dropout=0,
        num_end_layers=2,
        **kwargs,
    ):
        super().__init__()
        self.return_hidden_outputs = return_hidden_outputs

        assert num_layers >= 1
        self.gcn = GCNModel(
            in_dim,
            hidden_dim,
            num_layers,
            hidden_dim,
            act,
            jk,
            return_hidden_outputs,
            dropout,
            False,
            **kwargs,
        )

        self.pool_function = MultiAggregation(
            ["mean", "max", "std"],
            mode="cat",
            # mode_kwargs={"in_channels": hidden_dim, "out_channels": out_dim},
        )
        self.regressor = build_regressor(hidden_dim, out_dim, num_end_layers)

    @classmethod
    def from_cfg(cls, cfg, in_dim, out_dim):
        return cls(
            in_dim,
            cfg.model.hidden_dim,
            cfg.model.num_layers,
            out_dim,
            act=cfg.model.act,
            jk=cfg.model.jk,
            dropout=cfg.model.dropout,
            num_end_layers=cfg.model.num_end_layers,
            **{"add_self_loops": True, "normalize": True, "bias": True},
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        gcn_out, hidden_reps = self.gcn(x, edge_index, edge_weight)
        pooled = self.pool_function(gcn_out, index=batch)

        if self.return_hidden_outputs:
            if hidden_reps is None:
                hidden_reps = []

            # Maybe move everything to cpu to spare GPU memory
            hidden_reps.append(pooled.detach())
            curr_x = pooled
            for j, layer in enumerate(self.regressor):
                curr_x = layer(curr_x)
                if j + 1 != len(self.regressor) and len(self.regressor) - j <= 3:
                    hidden_reps.append(curr_x.detach())

            out = curr_x
        else:
            out = self.regressor(pooled)

        return (out, hidden_reps)

    def reset_parameters(self):
        pass
