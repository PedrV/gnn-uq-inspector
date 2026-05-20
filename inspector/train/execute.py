import torch
import numpy as np

from .utils import OptimizerFactory, SchedulerFactory
from .trainer_reduced import UnifiedTrainer
from .training_strategies import (
    node_classification_gcn,
    node_regression_gcn,
    graph_regression_gcn,
)

from ..models.model_factory import get_model

from inspector.seeder import seed_everything


def execute_generic(cfg, expname, dataset, seeds, device):
    NUM_REPETITIONS = cfg.repetition.num_repetitions
    NUM_MODELS = cfg.repetition.num_models

    if cfg.dataset.name in ("pems", "chameleon", "artnetviews"):
        if cfg.model.name in ("gcn", "gat", "megagat", "megagcn"):
            task_strategy = node_regression_gcn.NodeRegressionGCNTask(
                seed=seeds[-1], device=device
            )
        else:
            raise NotImplementedError(f"Unknown model: {cfg.model.name}")

        if cfg.dataset.target != "distribution":
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.GaussianNLLLoss()

    elif cfg.dataset.name in ("citeseer", "cora", "tolokers2"):
        if cfg.model.name in ("gcn", "gat", "megagat", "megagcn"):
            task_strategy = node_classification_gcn.NodeClassificationGCNTask(
                seed=seeds[-1], device=device
            )
        else:
            raise NotImplementedError(f"Unknown model: {cfg.model.name}")
        if cfg.dataset.name in ("tolokers2"):
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()

    elif cfg.dataset.name in ("gapsmallqm9"):
        if cfg.model.name == "gcngraph":
            task_strategy = graph_regression_gcn.GraphRegressionGCNTask(
                seed=seeds[-1],
                device=device,
                batch_size=cfg.training.batch_size,
            )
        else:
            raise NotImplementedError(f"Unknown model: {cfg.model.name}")
        if cfg.dataset.target != "distribution":
            criterion = torch.nn.MSELoss()
        else:
            criterion = torch.nn.GaussianNLLLoss()

    else:
        raise ValueError(f"Unknown task type, {cfg.dataset.name}")

    if not cfg.task.is_bayesian:
        output_dim = dataset.num_classes
        if cfg.dataset.target == "distribution":
            output_dim *= 2
        # Since we called set_default_device in run, model parameters
        # initialized here will automatically be on GPU. However, ...
        model = get_model(cfg, dataset.num_features, output_dim)
        model = model.to(device)

        opfact = OptimizerFactory(
            cls=torch.optim.Adam,
            kwargs={
                "lr": cfg.training.lr,
                "weight_decay": cfg.training.weight_decay,
            },
        )

        scfact = None
        if cfg.training.scheduler:
            scfact = SchedulerFactory(
                cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
                kwargs={
                    "mode": "min",
                    "factor": cfg.training.scheduler_decay_factor,
                    "patience": cfg.training.scheduler_patience,
                },
            )

        trainer = UnifiedTrainer(
            data=dataset,
            model=model,
            optimizer_factory=opfact,
            scheduler_factory=scfact,
            criterion=criterion,
            h5data_path=cfg.paths.h5_path,
            task=task_strategy,
            device=device,
            patience=cfg.model.patience,
            grace_period=cfg.model.grace_period,
            is_large=cfg.dataset.is_large,
            target=cfg.dataset.target,
            init_scheme=cfg.training.init_scheme,
        )

    trainer.clean_h5files()

    for j in range(NUM_REPETITIONS):
        for i in range(NUM_MODELS):
            seed_everything(seeds[(j * NUM_MODELS) + i])
            trainer.reset()

            history = trainer.train(
                cfg.training.epochs,
                model_name=cfg.model.name,
                run_name=f"REP{j}-ENS{i}",
                snapshot_frequency=cfg.logging.snapshot_frequency,
                validate_frequency=cfg.logging.validate_frequency,
                only_print_end_result=False,
                validation_logic=cfg.task.validation_logic,
            )

            best_ind = np.argmin(history[cfg.task.validation_logic])
            best_loss = history[cfg.task.validation_logic][best_ind]
            print(
                f"[{expname}/{cfg.model.name}/{cfg.dataset.name}] Rep {j}, Model {i}: Best Validation Logic Metric {best_loss:.4f} at {best_ind}"
            )

            test_key = [
                k
                for k in history.keys()
                if k.endswith("_test") or k.startswith("test_")
            ][0]
            best_test_metric = history[test_key][best_ind]
            print(
                f"[{expname}/{cfg.model.name}/{cfg.dataset.name}] Rep {j}, Model {i}: Corresponding Test Metric {best_test_metric:.4f} at {best_ind}"
            )
