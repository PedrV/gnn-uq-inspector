from .gcn import GCNModel, GCNModelGraphWide
from .gat import GATModel
from .mega_gat import MegaGATModel
from .mega_gcn import MegaGCNModel

from .registry import get_model_from_registry


def get_model(cfg, in_dim, out_dim):
    return get_model_from_registry(cfg, in_dim, out_dim)
