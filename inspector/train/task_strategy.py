from abc import ABC, abstractmethod


class TrainingTask(ABC):
    """Shared base for all training strategies."""

    def __init__(self, seed: int, device, **kwargs):
        self.seed = seed
        self.device = device

    @abstractmethod
    def prepare(self, data, **kwargs) -> None:
        """Called once before training begins (e.g. build DataLoaders)."""
        pass


class DeterministicTrainingTask(TrainingTask):
    """
    Strategy for deterministic (non-Bayesian) models.

    train_epoch receives the full PyTorch objects it needs and returns a
    scalar loss. The framework parameter is intentionally absent. There is
    no SVI/MCMC object in this path.
    """

    @abstractmethod
    def train_epoch(self, model, optimizer, criterion, data) -> float:
        pass

    def val_epoch(self, model, criterion, data) -> dict:
        return {}
