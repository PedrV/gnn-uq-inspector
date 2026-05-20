import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

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
    "elu": nn.ELU,
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


class FeedForwardModule(nn.Module):
    def __init__(self, dim, num_inputs=1, hidden_dim_multiplier=1, dropout=0, **kwargs):
        super().__init__()
        input_dim = int(dim * num_inputs)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x


@register_model("megagcn", task_type="node")
class MegaGCNModel(torch.nn.Module):
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
        **kwargs,
    ):
        super().__init__()
        self.return_hidden_outputs = return_hidden_outputs
        self.act = get_activation(act)

        assert num_layers >= 1
        self.jk = None
        self.num_layers = num_layers

        self.input_module = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_dim),
            nn.Dropout(p=dropout),
            get_activation(act),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
        )

        self.layers = nn.ModuleList()
        self.ff_modules = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))

            # Add the FeedForward and Norms (to match ResidualWrapper from GraphLand)
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.ff_modules.append(FeedForwardModule(dim=hidden_dim, dropout=dropout))

        # If we cat all layers, the input to the output module
        # will be hidden_dim * num_layers.
        if self.jk == "cat":
            final_in_dim = hidden_dim * num_layers
        else:
            final_in_dim = hidden_dim

        self.output_module = nn.Sequential(
            nn.LayerNorm(final_in_dim),
            nn.Linear(final_in_dim, hidden_dim),
            get_activation(act),
            nn.Linear(hidden_dim, out_dim),
        )

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
        )

    def forward(self, x, edge_index, edge_weight=None):
        x = self.input_module(x)

        layer_outputs = []
        for i in range(self.num_layers):
            # Residual/Wrapper Logic: Norm -> GCN -> FF -> Add
            h = self.norms[i](x)
            h = self.layers[i](h, edge_index, edge_weight=edge_weight)
            h = self.ff_modules[i](h)
            x = x + h  # Residual Connection
            if self.jk is not None:
                layer_outputs.append(x)

        if self.jk == "cat":
            x = torch.cat(layer_outputs, dim=-1)
        elif self.jk == "max":
            x = torch.stack(layer_outputs, dim=0).max(dim=0)[0]

        return self.output_module(x), None

    def reset_parameters(self):
        pass
