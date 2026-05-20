import datetime

import torch
from typing import Type
from dataclasses import dataclass, field


@dataclass
class OptimizerFactory:
    cls: Type[torch.optim.Optimizer]
    kwargs: dict = field(default_factory=dict)

    def build(self, model_params):
        return self.cls(model_params, **self.kwargs)


@dataclass
class SchedulerFactory:
    cls: Type[torch.optim.lr_scheduler.LRScheduler]
    kwargs: dict = field(default_factory=dict)

    _METRIC_SCHEDULERS = (torch.optim.lr_scheduler.ReduceLROnPlateau,)

    def build(self, optimizer):
        return self.cls(optimizer, **self.kwargs)

    def step(self, scheduler, metric=None):
        if isinstance(scheduler, self._METRIC_SCHEDULERS):
            if metric is None:
                raise ValueError(f"{self.cls.__name__} requires a metric to step.")
            scheduler.step(metric)
        else:
            scheduler.step()


class EarlyStopping:
    def __init__(self, patience=20, min_delta=0, grace_period=None):
        self.patience = patience
        self.min_delta = min_delta
        self.grace_period = grace_period if grace_period is not None else 0
        self._reset_state()

    def _reset_state(self):
        self.grace_counter = 0
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def reset(self):
        """Reset internal state between runs, preserving hyperparameters."""
        self._reset_state()

    def __call__(self, val_loss) -> bool:
        """Returns True if training should stop."""
        if self.patience is None:
            return False

        if self.grace_counter < self.grace_period:
            self.grace_counter += 1
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

def format_delta(seconds):
    delta = datetime.timedelta(seconds=seconds)
    hours, remainder = divmod(delta.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        return f"{seconds:.3f}s"
