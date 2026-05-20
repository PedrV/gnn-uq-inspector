import os
from pathlib import Path
from functools import lru_cache

from inspector.consts import MYDTYPE

import torch
import numpy as np

torch.set_default_dtype(torch.float32 if MYDTYPE == "float" else torch.float64)
torch.set_num_threads(8)
torch.set_num_interop_threads(1)

import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import Planetoid

from inspector.data.pems_dataset import PEMS
from inspector.data.tolokers2_dataset import Tolokers2
from inspector.data.artnetviews_dataset import ArtnetViews
from inspector.data.smallqm9_dataset import GapSmallQM9
from inspector.data.chameleon_dataset import Chameleon
from torch_geometric.transforms import Compose

from inspector.data.data_utils import (
    StandardizeOutput,
    PEMSTransformMaskedOuputs,
    FeatureNormalisation,
    LogarithmOutput,
)

from inspector.train.execute import execute_generic

from inspector.metrics.calculate_classification import ClassificationMetrics
from inspector.metrics.calculate_regression import RegressionMetrics

from inspector.seeder import generate_experiment_seeds, seed_everything

TASK_CLASS_MAP = {
    "classification": ClassificationMetrics,
    "regression": RegressionMetrics,
}

def get_base_dataset(cfg):
    """Encapsulates the raw PyG dataset loading logic."""
    name = cfg.dataset.name
    version = cfg.dataset.version
    path = cfg.paths.data_path

    if name in ["cora", "citeseer"]:
        pre_tranf = FeatureNormalisation() if version == "normalised_features" else None
        return Planetoid(path, name=name, pre_transform=pre_tranf)

    if name == "pems":
        if version in ("original"):
            return PEMS(path, pre_transform=PEMSTransformMaskedOuputs())

    if name == "chameleon":
        if version in ("original"):
            pre_tranf = StandardizeOutput()
        elif version in ("binary_features_logarithmic_target"):
            pre_tranf = Compose([LogarithmOutput(), StandardizeOutput()])

        if version == "binary_features_logarithmic_target":
            return Chameleon(path, pre_transform=pre_tranf, version="binary")
        else:
            return Chameleon(path, pre_transform=pre_tranf)

    if name == "gapsmallqm9":
        if version in ("original"):
            return GapSmallQM9(path, pre_transform=StandardizeOutput())

    if name == "tolokers2":
        if version in ("original"):
            return Tolokers2(path)
        if version in ("shift_original"):
            print("'Progressive' shift being used!")
            return Tolokers2(path, make_shift=True)

    if name == "artnetviews":
        if version in ("original"):
            return ArtnetViews(path, pre_transform=StandardizeOutput())
        if version in ("shift_original"):
            print("'Progressive' shift being used!")
            return ArtnetViews(path, pre_transform=StandardizeOutput(), make_shift=True)

    raise ValueError(f"Unknown dataset {name} or version {version}")


@lru_cache(maxsize=1)
def get_random_name():
    """
    Returns a consistent random name for the duration of the process.
    lru_cache(maxsize=1) acts as a thread-safe singleton.
    """
    with open("names1.yaml", "r") as f:
        data = yaml.safe_load(f)

    rng = np.random.default_rng()
    category = rng.choice(list(data.keys()))
    name = rng.choice(data[category])
    return name.lower()


OmegaConf.register_new_resolver("random_name", get_random_name)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.task.load_run_dir is None:
        run_dir = Path(HydraConfig.get().run.dir).resolve()
    else:
        run_dir = Path(cfg.task.load_run_dir).resolve()

    cfg.paths.data_path = os.path.join(
        cfg.paths.root_data_path, cfg.dataset.name, cfg.dataset.version
    )
    cfg.paths.entropy_path = os.path.join(
        cfg.paths.root_entropy_path, cfg.dataset.name, f"entropy-{cfg.dataset.name}.pkl"
    )
    cfg.paths.stats_path = str(run_dir / "stats")
    cfg.paths.h5_path = str(run_dir / "h5files")

    NUM_REPETITIONS = cfg.repetition.num_repetitions
    NUM_MODELS = cfg.repetition.num_models

    if cfg.task.is_bayesian:
        cfg.paths.entropy_path = None
    seeds = generate_experiment_seeds(
        NUM_REPETITIONS * NUM_MODELS + 2,
        entropy_path=cfg.paths.entropy_path,
        master_seed=cfg.master_seed,
    )

    pyg_data = get_base_dataset(cfg)

    torch_dtype_to_use = torch.float32 if MYDTYPE == "float" else torch.float64
    pyg_data._data.x = pyg_data._data.x.to(dtype=torch_dtype_to_use)
    if (
        pyg_data._data.y.dtype == torch.float32
        or pyg_data._data.y.dtype == torch.float64
    ):
        pyg_data._data.y = pyg_data._data.y.to(dtype=torch_dtype_to_use)

    if torch.cuda.is_available():
        torch.set_default_device("cuda")
        device = torch.device("cuda")
    else:
        torch.set_default_device("cpu")
        device = torch.device("cpu")

    if cfg.task.do_train:
        execute_generic(cfg, HydraConfig.get().job.name, pyg_data, seeds, device)
        # pyg_data remains on cpu because when .to() is called
        # inside the execute_generic, a *copy* is placed on the gpu

    torch.set_default_device("cpu")
    original_std, original_mean = torch.tensor(1), torch.tensor(0)
    if cfg.dataset.type == "regression":
        original_std = pyg_data.original_std
        original_mean = pyg_data.original_mean
        assert original_mean.dim() <= 1
        assert original_std.dim() <= 1

        if pyg_data.y.ndim < 2:
            pyg_data._data.y.unsqueeze_(-1)
        assert pyg_data.y.ndim == 2

    if original_mean.dim() == 0:
        original_mean.unsqueeze_(0)
    if original_std.dim() == 0:
        original_std.unsqueeze_(0)

    seed_everything(seeds[-2])
    metrics_cls = TASK_CLASS_MAP[cfg.dataset.type]
    metrics = metrics_cls(
        cfg=cfg,
        seed=cfg.master_seed,
        output_dim=pyg_data.num_classes,
        epoch_sel=cfg.task.epoch,
        cache_predictions=True,
        original_std=original_std,
        original_mean=original_mean,
        stable=False,
    )
    if cfg.task.do_nll:
        metrics.run_metric(
            pyg_data,
            "nll",
            trivial=cfg.task.is_trivial,
            probabilistic=cfg.task.is_bayesian,
            stats_path=cfg.paths.stats_path,
            save_data=True,
        )
        if cfg.task.do_extra:
            metrics.run_metric(
                pyg_data,
                trivial=cfg.task.is_trivial,
                probabilistic=cfg.task.is_bayesian,
                metric="extra",
                stats_path=cfg.paths.stats_path,
                save_data=True,
            )
    if cfg.task.do_uq:
        metrics.run_metric(
            pyg_data,
            "uq",
            trivial=cfg.task.is_trivial,
            probabilistic=cfg.task.is_bayesian,
            stats_path=cfg.paths.stats_path,
            save_data=True,
        )


if __name__ == "__main__":
    main()
