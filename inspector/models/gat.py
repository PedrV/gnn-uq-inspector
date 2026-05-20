import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

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


@register_model("gat", task_type="node")
class GATModel(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers,
        out_dim,
        act="relu",
        jk=None,
        return_hidden_outputs=False,
        **kwargs,
    ):
        super().__init__()
        self.return_hidden_outputs = return_hidden_outputs

        assert num_layers >= 1
        assert jk is None, "Currently custom GAT does not support JK"
        self.act = get_activation(act)
        self.num_layers = num_layers

        heads_info_list = [1] * num_layers
        if "heads" in kwargs:
            heads_info = kwargs.pop("heads")

            if isinstance(heads_info, list) and len(heads_info) != num_layers:
                raise ValueError(
                    f"Heads list length ({len(heads_info)}) must match num_layers ({num_layers})"
                )

            if isinstance(heads_info, int):
                if hidden_dim % heads_info != 0:
                    raise ValueError(
                        f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads_info})"
                    )
                heads_info_list = [heads_info] * num_layers

            elif isinstance(heads_info, list):
                for h in heads_info:
                    if not isinstance(h, int) or hidden_dim % h != 0:
                        raise ValueError(
                            f"All heads in list must be integers and divide hidden_dim ({hidden_dim})"
                        )
                heads_info_list = heads_info

        if hidden_dim is None:
            hidden_dim = 0

        current_in = in_dim
        self.layers = torch.nn.ModuleList([])
        for i in range(num_layers):
            # Final layer usually averages heads instead of concatenating
            is_last = i == num_layers - 1
            concat_attention = not is_last

            d_out = out_dim if is_last else hidden_dim  # // heads_info_list[i]

            kwargs["heads"] = heads_info_list[i]
            kwargs["concat"] = concat_attention
            self.layers.append(GATConv(current_in, d_out, **kwargs))

            current_in = out_dim if is_last else hidden_dim * heads_info_list[i]

    @classmethod
    def from_cfg(cls, cfg, in_dim, out_dim):
        return cls(
            in_dim,
            cfg.model.hidden_dim,
            cfg.model.num_layers,
            out_dim,
            act=cfg.model.act,
            jk=cfg.model.jk,
            **{"heads": cfg.model.heads},
        )

    def forward(self, x, edge_index, edge_weight=None):
        emb = []
        for i in range(self.num_layers - 1):
            x = self.act(self.layers[i](x, edge_index, edge_attr=edge_weight))
            if self.return_hidden_outputs:
                emb.append(x)

        out = self.layers[-1](x, edge_index, edge_attr=edge_weight)
        hidden_reps = [e.detach() for e in emb] if self.return_hidden_outputs else None

        return (out, hidden_reps)

    def reset_parameters(self):
        pass
