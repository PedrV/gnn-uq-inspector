import h5py
import os
import torch
import numpy as np
import traceback
import copy
from pathlib import Path
from tqdm import tqdm

from .task_strategy import TrainingTask
from .utils import EarlyStopping


class UnifiedTrainer:
    def __init__(
        self,
        data,
        model,
        optimizer_factory,
        criterion,
        h5data_path,
        task: TrainingTask,
        device,
        patience,
        grace_period,
        is_large=False,
        target=None,
        scheduler_factory=None,
        init_scheme="xavier",
    ):
        self.data = data
        self.model = model
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.criterion = criterion
        self.h5data_path = Path(h5data_path)
        self.device = device  # PyTorch operations preserve the device of the inputs
        self.task = task
        self.patience = patience
        self.grace_period = grace_period
        self.init_scheme = init_scheme
        self.is_large = is_large
        self.target = target

        self._save_embeddings = False
        self._save_weights = True
        self._save_predictions = True

    @property
    def save_embeddings(self):
        return self._save_embeddings

    @save_embeddings.setter
    def save_embeddings(self, value):
        self._save_embeddings = value

    @property
    def save_weights(self):
        return self._save_weights

    @save_weights.setter
    def save_weights(self, value):
        self._save_weights = value

    @property
    def save_predictions(self):
        return self._save_predictions

    @save_predictions.setter
    def save_predictions(self, value):
        self._save_predictions = value

    def reset(self, loss=None):
        """Resets model weights and optimizer state."""
        self._reset_model_weights()
        self._reset_optimizer()
        # self._reset_loss()  # Loss reset if needed

        self.task.prepare(self.data, **{"is_large": self.is_large, "target": self.target})

    def _reset_model_weights(self):
        self.model.reset_parameters()
        for name, param in self.model.named_parameters():
            if "bias" in name:
                torch.nn.init.zeros_(param)
            elif param.dim() < 2:
                # Catch 1D weights (like LayerNorm or BatchNorm)
                torch.nn.init.ones_(param)
            else:
                if self.init_scheme == "xavier":
                    torch.nn.init.xavier_uniform_(param)
                elif self.init_scheme == "he":
                    torch.nn.init.kaiming_uniform_(param, nonlinearity="relu")

    def _reset_optimizer(self):
        # Maybe in the future we will have a cool reset function for optimizers
        # https://github.com/pytorch/pytorch/issues/37410
        # Caveat: If optimizer has any state that is not in defaults, it will not be restored.
        # But for most standard optimizers, defaults covers everything
        # self.optimizer = self.optimizer.__class__(
        #     self.model.parameters(), **self.optimizer.defaults
        # )
        # New Method
        self.optimizer = self.optimizer_factory.build(self.model.parameters())
        if self.scheduler_factory is not None:
            self.scheduler = self.scheduler_factory.build(self.optimizer)
        else:
            self.scheduler = None

    def _create_h5_dir(self):
        self.h5data_path.mkdir(exist_ok=True, parents=True)

    def create_h5files(self, run_name, model_name):
        self._create_h5_dir()

        f_outputs = f_weights = f_emb = f_metrics = None
        preds_ = out_ = wei_ = emb_ = None

        f_metrics = h5py.File(
            self.h5data_path / f"metrics_{run_name}_{model_name}.hdf5", "w"
        )
        out_ = f_metrics.create_group("metrics")

        if self.save_predictions:
            f_outputs = h5py.File(
                self.h5data_path / f"predictions_{run_name}_{model_name}.hdf5", "w"
            )
            preds_ = f_outputs.create_group("predictions")

        if self.save_weights:
            f_weights = h5py.File(
                self.h5data_path / f"weights_{run_name}_{model_name}.hdf5", "w"
            )
            wei_ = f_weights.create_group("weights")

        if self.save_embeddings:
            f_emb = h5py.File(
                self.h5data_path / f"embeddings_{run_name}_{model_name}.hdf5", "w"
            )
            emb_ = f_emb.create_group("embeddings")

        return f_outputs, f_weights, f_emb, f_metrics, preds_, out_, wei_, emb_

    def clean_h5files(self):
        self._create_h5_dir()

        for f in os.listdir(self.h5data_path):
            if f.endswith(".hdf5"):
                os.remove(self.h5data_path / f)

    def train(
        self,
        epochs,
        model_name,
        run_name,
        snapshot_frequency,
        validate_frequency,
        only_print_end_result=False,
        validation_logic="loss",
    ):
        best_metric = np.inf
        best_state_dict = None
        best_epoch = -1
        best_metrics_scalars = {}

        stopper = EarlyStopping(patience=self.patience, grace_period=self.grace_period)
        history = {"loss": []}

        # torch.cuda.empty_cache()
        try:
            f_outputs, f_weights, f_emb, f_metrics, preds_, out_, wei_, emb_ = (
                self.create_h5files(run_name, model_name)
            )

            epoch_print_mod_value = epochs // 10 
            pbar = tqdm(range(1, epochs + 1))
            for epoch in pbar:

                # _start_time = datetime.datetime.now()
                # --- STRATEGY: Train Step ---
                loss = self.task.train_epoch(
                    self.model, self.optimizer, self.criterion, self.data
                )
                # _total_time = datetime.datetime.now() - _start_time
                # train_epoch_times.append(_total_time.total_seconds())

                # --- STRATEGY: Validation Step ---
                step_metrics = self.task.val_epoch(
                    self.model, self.criterion, self.data
                )

                # Extract reserved keys that shouldn't be logged as scalar metrics
                _ = step_metrics.pop("embeddings", None)
                _ = step_metrics.pop("outputs", None)

                history["loss"].append(loss)
                for k, v in step_metrics.items():
                    if k not in history:
                        history[k] = []
                    history[k].append(v)

                val_variable = loss
                if validation_logic != "loss":
                    val_variable = step_metrics["validation_loss"]

                if self.scheduler is not None:
                    self.scheduler_factory.step(self.scheduler, metric=val_variable)

                # --- IN-MEMORY SAVING LOGIC FOR BEST EPOCH ---
                if val_variable < best_metric:
                    best_metric = val_variable
                    best_epoch = epoch
                    best_state_dict = copy.deepcopy(self.model.state_dict())

                    # Store scalar metrics to save them properly at the end
                    best_metrics_scalars = {k: v for k, v in step_metrics.items()}
                    best_metrics_scalars["loss"] = loss

                postfix = {
                    k: f"{v:.4f}"
                    for k, v in step_metrics.items()
                }
                if self.scheduler is not None:
                    postfix["LR"] = f"{self.optimizer.param_groups[0]['lr']:.5f}"

                pbar.set_postfix(postfix)

                if (
                    not only_print_end_result and ((epoch % epoch_print_mod_value) == 0 or epoch == 1)
                ) or (only_print_end_result and epoch == epochs):
                    tqdm.write(
                        " | ".join([f"{k}: {v}" for k, v in postfix.items()])
                    )

                if stopper(val_variable):
                    print(f"Early stopped triggered at epoch {epoch}/{epochs}")
                    break

            if best_state_dict is not None:
                print(f"Evaluating and saving best model from epoch {best_epoch}...")

                self.model.load_state_dict(best_state_dict)
                best_step_metrics = self.task.val_epoch(
                    self.model, self.criterion, self.data
                )

                best_embeddings = best_step_metrics.pop("embeddings", None)
                best_outputs = best_step_metrics.pop("outputs", None)

                _number = str(best_epoch).zfill(len(str(epochs)))
                if self.save_weights:
                    for k, v in best_state_dict.items():
                        wei_.create_dataset(
                            f"{_number}/{k}",
                            data=v.detach().cpu().numpy(),
                            dtype=np.float64,
                        )

                if self.save_embeddings and best_embeddings is not None:
                    for layer_num, h in enumerate(best_embeddings):
                        emb_.create_dataset(
                            f"{_number}/L{layer_num}",
                            data=h.cpu().numpy(),
                            dtype=np.float64,
                            compression="gzip",
                            compression_opts=9,
                        )

                if self.save_predictions and best_outputs is not None:
                    preds_.create_dataset(
                        f"{_number}",
                        data=best_outputs.cpu().numpy(),
                        dtype=np.float64,
                    )

                out_.create_dataset(
                    f"{_number}/loss", data=best_metrics_scalars["loss"]
                )
                for k, v in best_metrics_scalars.items():
                    if k != "loss":
                        out_.create_dataset(f"{_number}/{k}", data=v, dtype=np.float32)

            return history

        except Exception as e:
            print(traceback.format_exc())
            raise e
        finally:
            for f in (f_outputs, f_weights, f_emb, f_metrics):
                if f is not None:
                    f.close()
