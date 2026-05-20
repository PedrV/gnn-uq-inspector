import os
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
import itertools

from inspector.consts import MYDTYPE

import h5py
import yaml

import math
import random
import torch
import numpy as np

torch.set_default_dtype(torch.float32 if MYDTYPE == "float" else torch.float64)

from .store_and_print import store_metrics

# Path to the metrics info yaml, adjust if it lives elsewhere relative to this file
_METRICS_INFO_PATH = Path(__file__).parent / "metrics_info.yaml"


def _load_metrics_info(dataset_name: str) -> dict:
    """
    Returns the metrics_info entry for the given dataset, e.g.:
        {
            "main_metric": "RMSE",
            "other_metrics": ["R2"],
            "metric_directions": {"RMSE": "min", "R2": "max", "Diagonal NLL": None, ...},
        }
    Returns an empty dict if the dataset is not found.
    """
    if not _METRICS_INFO_PATH.exists():
        raise FileNotFoundError(f"metrics_info.yaml not found at {_METRICS_INFO_PATH}")

    with open(_METRICS_INFO_PATH, "r") as f:
        info = yaml.safe_load(f)

    entry = info.get(dataset_name, {}) or {}
    return {
        "main_metric": entry.get("main_metric", None),
        "other_metrics": entry.get("other_metrics") or [],
        "metric_directions": entry.get("metric_directions") or {},
    }


def _combination_at_rank(n: int, k: int, rank: int) -> tuple:
    """
    Return the combination at lexicographic position `rank` in C(n, k)
    using the combinatorial number system. O(k * n) worst case.

    Small example:
    If we ask for a combination with index 500000 of (50 5), for index 0 for the first index of the combination index
    we would get 211876 combinations. Not enough, we need more. If we try with index 1 for index 0 of the combination
    we are allowed to do an extra 194580.
    We need to get to the index 2 to be able to finally reach past 500000. So we pick 2 for index 0.
    If we were asking for combination 1000, since with index 0 for index 0 of the combination we can do
    211876 we definitely can use 0 and we will be able to reach the combination numbered 1000.
    This is only possible because we assume combinations are ordered in lexicographic order.

    """
    result = []
    start = 0
    for i in range(k, 0, -1):
        for c in range(start, n):
            ways = math.comb(n - c - 1, i - 1)
            if rank < ways:
                result.append(c)
                start = c + 1
                break
            rank -= ways
    return tuple(result)


def _sample_ranks(total_combos: int, k: int, seed) -> list[int]:
    """Sample k distinct ranks from [0, total_combos) for arbitrarily large total_combos."""
    rng = random.Random(seed)
    seen = set()
    while len(seen) < k:
        r = rng.randint(0, total_combos - 1)
        seen.add(r)
    return list(seen)


class BaseMetricsComputation(ABC):
    """
    Base class to handle ensemble predictions loading, repetitions, and metrics storage.
    Metric computation is left to subclass implementations.
    """

    def __init__(
        self,
        cfg,
        output_dim,
        seed,
        epoch_sel=None,
        cache_predictions=True,
        **kwargs,
    ):
        self.cfg = cfg
        self.seed = seed
        self.num_repetitions = cfg.repetition.num_repetitions
        self.num_models = cfg.repetition.num_models
        self.output_dim = output_dim
        self.epoch_sel = epoch_sel
        self.cache_predictions = cache_predictions
        self._predictions_cache = None
        self.baseline = cfg.metrics.uq_baseline
        self.best_baseline = cfg.metrics.uq_best_baseline
        self.uq_type = cfg.metrics.uq_type
        self.save_uq_decomposition = cfg.metrics.save_uq_decomposition
        self.uq_decomposition = {}
        self.baseline_regression_single_models_std = None  # DE-R
        self.baseline_single_model_id = None

        if (
            self.cfg.metrics.virtual_num_models is not None
            and self.cfg.metrics.max_virtual_repetitions is not None
        ):
            self.actual_num_models = self.cfg.metrics.virtual_num_models
            self.actual_num_repetitions = self.cfg.metrics.max_virtual_repetitions
        elif self.cfg.metrics.virtual_repetitions is not None:
            self.actual_num_repetitions = self.cfg.metrics.virtual_repetitions
        else:
            self.actual_num_models = self.num_models
            self.actual_num_repetitions = self.num_repetitions

        if (
            self.cfg.dataset.target is not None
            and self.cfg.dataset.target == "distribution"
        ):
            self.distribution_output = True
        elif self.cfg.dataset.target is None:
            self.distribution_output = False

        # Load metric names for this dataset once
        self._metrics_info = _load_metrics_info(cfg.dataset.name)

    @property
    def main_metric(self) -> str:
        return self._metrics_info["main_metric"]

    @property
    def other_metrics(self) -> list:
        return self._metrics_info["other_metrics"]

    @property
    def metric_directions(self) -> dict:
        """
        Dict mapping metric name (or substring) to 'max', 'min', or None.
        Sourced from the metric_directions block in metrics_info.yaml.
        """
        return self._metrics_info["metric_directions"]

    def load_predictions(
        self,
        num_examples,
        virtual_repetitions=None,
        virtual_num_models=None,
        max_virtual_repetitions=None,
    ):
        """
        virtual_repetitions=None, virtual_num_models=None -> Original behaviour
        virtual_repetitions=int, virtual_num_models=None -> Sequential split
        virtual_repetitions=None, virtual_num_models=int -> Combinatorial split

        Args:
            num_examples (int):
                The number of common examples (e.g., test set size) that each
                individual model/repetition combination was run on.

            virtual_repetitions (int, optional):
                The *new* (virtual) number of repetitions. In combination mode,
                used only as a consistency check on the final count. Safe
                to pass None for this case. In the original sequential-split mode,
                drives the split directly e.g. virtual_repetitions=3 for 150 models
                mean 3 repetitions of 50 models.

            virtual_num_models (int, optional):
                Ensemble size k for combinatorial mode (C(n, k) ensembles).
                If None, the original sequential-split logic is used.

            max_virtual_repetitions (int, optional):
                Maximum number of combinations to materialise. When set, a random
                subset of that size is drawn lazily -- the full C(n,k) enumeration
                is never constructed. When None, all C(n,k) combinations are used.
        """
        if self.cache_predictions and self._predictions_cache is not None:
            return [p.clone() for p in self._predictions_cache]

        per_ensemble_predictions = []
        cur_out = self.output_dim
        if self.distribution_output:
            cur_out *= 2
        for j in range(self.num_repetitions):
            individual_model_predictions = torch.zeros(
                (self.num_models, num_examples, cur_out)
            )
            fps = []
            fps_metrics = []
            try:
                # Collect model files for repetition j
                for f in sorted(os.listdir(self.cfg.paths.h5_path)):
                    if f.endswith(".hdf5") and f.startswith(f"predictions_REP{j}-"):
                        fps.append(
                            h5py.File(os.path.join(self.cfg.paths.h5_path, f), "r")
                        )
                    elif f.endswith(".hdf5") and f.startswith(f"metrics_REP{j}-"):
                        fps_metrics.append(
                            h5py.File(os.path.join(self.cfg.paths.h5_path, f), "r")
                        )

                assert (
                    len(fps) == self.num_models == len(fps_metrics)
                ), f"Expected {self.num_models} models, got {len(fps)} predictions and {fps_metrics} metrics for REP{j}"

                for i, fp in enumerate(fps):
                    best_epoch, best_loss = "", np.inf

                    # Select best epoch or user-defined epoch
                    for l, (k, v) in enumerate(fps_metrics[i]["metrics"].items()):
                        if v[self.cfg.task.validation_logic][()] < best_loss:
                            best_epoch = k
                            best_loss = v[self.cfg.task.validation_logic][()]

                    # Normal preds having NaNs? I notice a lot of NaNs
                    if self.epoch_sel is None:
                        _preds = torch.from_numpy(fp["predictions"][best_epoch][:])
                    else:
                        _ep = "epoch-" + str(self.epoch_sel).zfill(len(str(l)))
                        _preds = torch.from_numpy(fp["predictions"][_ep][:])

                    if _preds.dim() < 2:
                        _preds.unsqueeze_(-1)
                    assert _preds.dim() == 2
                    assert (
                        _preds.shape[0] == num_examples
                    ), f"Pred Shape: {_preds.shape[0]}, Num Examples: {num_examples}"

                    individual_model_predictions[i, :, :] = _preds

            finally:
                for fp in fps:
                    fp.close()
                for fp_m in fps_metrics:
                    fp_m.close()

            per_ensemble_predictions.append(individual_model_predictions)

        if virtual_num_models is not None:
            all_models_flat = torch.cat(per_ensemble_predictions, dim=0)
            total_models = all_models_flat.size(0)
            total_combos = math.comb(total_models, virtual_num_models)

            if max_virtual_repetitions is None:
                combo_iter = itertools.combinations(
                    range(total_models), virtual_num_models
                )
            else:
                if max_virtual_repetitions > total_combos:
                    raise ValueError(
                        f"max_virtual_repetitions ({max_virtual_repetitions}) exceeds "
                        f"the total number of combinations C({total_models}, {virtual_num_models}) "
                        f"= {total_combos}."
                    )
                # Draw exactly max_virtual_repetitions distinct ranks at random,
                # then convert each rank to its combination on the fly
                sampled_ranks = _sample_ranks(
                    total_combos, max_virtual_repetitions, self.seed
                )
                combo_iter = (
                    _combination_at_rank(total_models, virtual_num_models, r)
                    for r in sampled_ranks
                )

            per_ensemble_predictions = [
                all_models_flat[list(combo)] for combo in combo_iter
            ]

            if (
                virtual_repetitions is not None
                and len(per_ensemble_predictions) != virtual_repetitions
            ):
                raise ValueError(
                    f"Number of selected combinations ({len(per_ensemble_predictions)}) does not "
                    f"match virtual_repetitions ({virtual_repetitions}). "
                    f"Adjust max_virtual_repetitions or leave virtual_repetitions=None."
                )
            print(len(per_ensemble_predictions), per_ensemble_predictions[0].shape)

        elif virtual_repetitions is not None:
            all_models_flat = torch.cat(per_ensemble_predictions, dim=0)

            if all_models_flat.size(0) % virtual_repetitions != 0:
                raise ValueError(
                    "Total models not evenly divisible by virtual_repetitions."
                )

            per_ensemble_predictions = list(
                all_models_flat.split(
                    all_models_flat.size(0) // virtual_repetitions, dim=0
                )
            )
            assert len(per_ensemble_predictions) == virtual_repetitions

        if self.cache_predictions:
            self._predictions_cache = per_ensemble_predictions

        return [p.clone() for p in per_ensemble_predictions]

    def run_metric(
        self,
        pyg_data,
        metric,
        trivial=False,
        probabilistic=False,
        stats_path=None,
        save_data=None,
    ):
        if trivial:
            # Forge a single "repetition" of processed data
            # We wrap it in a list to mimic the ensemble structure [rep1, rep2...]
            ensemble_data = [
                self._forge_trivial_data(
                    pyg_data.y, pyg_data.train_mask, pyg_data.test_mask, metric
                )
            ]
            filename = f"trivial_{metric}_stats"
        elif not probabilistic:
            raw_ensemble = self.load_predictions(
                pyg_data.y.shape[0],
                self.cfg.metrics.virtual_repetitions,
                self.cfg.metrics.virtual_num_models,
                self.cfg.metrics.max_virtual_repetitions,
            )
            ensemble_data = [
                self._process_predictions(preds_raw, pyg_data.test_mask, metric)
                for preds_raw in raw_ensemble
            ]
            filename = f"{metric}_{self.uq_type}_stats"

        if metric == "extra":
            filename = f"extra_{self.uq_type}_metrics"

        # Skip extra metrics entirely if none are configured for this dataset
        if metric == "extra" and not self.other_metrics:
            print(
                f"No extra metrics configured for dataset '{self.cfg.dataset.name}'. "
                "Skipping extra_metrics computation."
            )
            return

        if self.cfg.metrics.virtual_repetitions is not None:
            assert self.cfg.metrics.virtual_repetitions == self.actual_num_repetitions
            suffix_filename = f"_{pyg_data.y.shape[0]//self.actual_num_repetitions}x{self.actual_num_repetitions}"
        else:
            suffix_filename = f"_{self.actual_num_models}x{self.actual_num_repetitions}"
        filename += suffix_filename

        if self.baseline:
            filename += "_baseline"
        elif self.best_baseline:
            filename += "_best-baseline"

        stats_accumulator = defaultdict(list)

        for j, preds_processed in enumerate(ensemble_data):
            # for single_model_id in range(self.regression_single_models_std.shape[-1]):
            # self.baseline_single_model_id = single_model_id
            if metric == "nll":
                results = self._compute_nll_for_repetition(
                    preds_processed,
                    pyg_data.y,
                    pyg_data.test_mask,
                    pyg_data.train_mask,
                    pyg_data.val_mask,
                )
                # for k, v in results.items():
                #     stats_accumulator[k].append(v)
            elif metric == "uq":
                results = self._compute_uq_for_repetition(
                    j, preds_processed, pyg_data.y, pyg_data.test_mask, stats_path
                )
            elif metric == "extra":
                results = self._compute_extra_for_repetition(
                    preds_processed, pyg_data.y, pyg_data.test_mask, pyg_data.train_mask
                )
            else:
                raise NotImplementedError(f"Don't know how to {metric}")

            for k, v in results.items():
                stats_accumulator[k].append(v)

        final_metrics = []
        for k, v_list in stats_accumulator.items():
            final_metrics.append((k, np.array(v_list)))

        self.store(
            stats_path,
            filename,
            save_data,
            *final_metrics,
            directions=self.metric_directions,
        )

        if self.save_uq_decomposition:
            np.savez(
                os.path.join(stats_path, f"uq_breakdown{suffix_filename}.npz"),
                **self.uq_decomposition,
            )

    def store(self, stats_path, filename, save_data, *metrics, directions=None):
        store_metrics(
            stats_path, save_data, *metrics, filename=filename, directions=directions
        )

    # --- Abstract Hooks ---
    @abstractmethod
    def _process_predictions(self, raw_preds, test_mask, metric):
        """Converts raw (N_models, N_samples, Dim) to usable form (Probabilities or Mean/Std/Cov)."""
        pass

    @abstractmethod
    def _compute_nll_for_repetition(
        self,
        preds,
        y_true,
        test_mask,
        train_mask,
        val_mask,
    ):
        """Returns dict of NLL/Accuracy/RMSE metrics for one repetition."""
        pass

    @abstractmethod
    def _compute_uq_for_repetition(self, rep_idx, preds, y_true, test_mask, stats_path):
        """Computes UQ metrics, generates plots, returns dict of results."""
        pass

    @abstractmethod
    def _compute_extra_for_repetition(self, preds, y_true, test_mask, train_mask):
        """
        Computes the other_metrics listed in metrics_info.yaml for one repetition.
        Returns a dict mapping metric names to per-dimension arrays, mirroring
        the structure of _compute_nll_for_repetition.
        """
        pass

    @abstractmethod
    def _forge_trivial_data(self, y_true, train_mask, test_mask, metric, **kwargs):
        """Returns processed data structure (e.g. dict or tensor) for the baseline."""
        pass
