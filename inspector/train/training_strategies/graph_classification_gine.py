import torch

from torch_geometric.loader import DataLoader

from ..task_strategy import DeterministicTrainingTask
from ...metrics.metrics import roc_auc

from inspector.seeder import make_generator, seed_worker


def reconstruct_full_embeddings_dataset(
    concatenated_reps_train,
    concatenated_reps_val,
    concatenated_reps_test,
    train_mask,
    val_mask,
    test_mask,
):
    """
    Reconstructs full-dataset embeddings per layer, placing train/val/test
    embeddings back into their original positions using boolean masks.

    Args:
        concatenated_reps_*: list of [N_split, D] tensors, one per layer
        *_mask: boolean tensors of shape [N_total]

    Returns:
        list of [N_total, D] tensors, one per layer
    """
    num_layers = len(concatenated_reps_train)
    N_total = train_mask.shape[0]
    full_reps = []
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == N_total, (
        "Masks overlap or don't cover the full dataset"
    )

    for layer_idx in range(num_layers):
        t_train = concatenated_reps_train[layer_idx]  # [N_train, D]
        t_val = concatenated_reps_val[layer_idx]  # [N_val, D]
        t_test = concatenated_reps_test[layer_idx]  # [N_test, D]

        D = t_train.shape[1]
        full = torch.zeros(N_total, D, dtype=t_train.dtype, device=torch.device("cpu"))

        full[train_mask] = t_train
        full[val_mask] = t_val
        full[test_mask] = t_test

        full_reps.append(full)

    return full_reps


def reconstruct_full_predictions_dataset(
    concatenated_preds_train,
    concatenated_preds_val,
    concatenated_preds_test,
    train_mask,
    val_mask,
    test_mask,
):
    """
    Reconstructs full-dataset of predictions, placing train/val/test
    embeddings back into their original positions using boolean masks.

    Args:
        concatenated_preds_*: list of [N_split, D] tensors contained predictions
        *_mask: boolean tensors of shape [N_total]

    Returns:
        list of [N_total, D] tensors
    """
    N_total = train_mask.shape[0]
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == N_total, (
        "Masks overlap or don't cover the full dataset"
    )

    t_train = concatenated_preds_train  # [N_train, D]
    t_val = concatenated_preds_val  # [N_val, D]
    t_test = concatenated_preds_test  # [N_test, D]

    D = t_train.shape[1]
    full = torch.zeros(N_total, D, dtype=t_train.dtype, device=torch.device("cpu"))

    full[train_mask] = t_train
    full[val_mask] = t_val
    full[test_mask] = t_test

    return full


class GraphClassificationGINETask(DeterministicTrainingTask):
    def __init__(self, seed, device, batch_size):
        super().__init__(int(seed), device)
        self.batch_size = batch_size
        # TODO: The .squeeze() on batch.y is no longer necessary on the new version of the dataset
        # Still, it does not harm to have it.

    def prepare(self, data, **kwargs):
        """
        Called ONCE at the start of a model run (e.g., inside trainer.reset())
        Note about DataLoader memory footprint:
        https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html#:~:text=Dataset%20stores%20the%20samples%20and,easy%20access%20to%20the%20samples
        """
        self.output_size = data.num_classes

        g = make_generator(self.seed, self.device)

        self.train_loader = DataLoader(
            data[data.train_mask],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Increasing this, makes recreate per epoch prohibitive
            generator=g,
            worker_init_fn=seed_worker,
        )
        self.validation_loader = DataLoader(
            data[data.val_mask],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            generator=g,
            worker_init_fn=seed_worker,
        )
        self.test_loader = DataLoader(
            data[data.test_mask],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            generator=g,
            worker_init_fn=seed_worker,
        )
        self.train_eval_loader = DataLoader(
            data[data.train_mask],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            generator=g,
            worker_init_fn=seed_worker,
        )

    def train_epoch(self, model, optimizer, criterion, data, scheduler):
        model.train()
        running_loss = 0.0
        total_seen_examples = 0

        for batch in self.train_loader:
            batch = batch.to(self.device)

            optimizer.zero_grad()
            outputs, _ = model(
                batch.x,
                batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch,
            )
            outputs.squeeze_()

            assert outputs.shape == batch.y.squeeze().shape, (
                f"Model out: {outputs.shape}, batch.y: {batch.y.squeeze().shape}"
            )
            loss = criterion(outputs, batch.y.squeeze())

            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            running_loss += loss.detach().cpu().item() * len(batch)
            total_seen_examples += len(batch)

        return running_loss / total_seen_examples

    def val_epoch(self, model, criterion, data):
        model.eval()
        test_loss = 0.0
        val_loss = 0.0
        train_loss = 0.0
        total_test_seen_examples = 0.0
        total_val_seen_examples = 0.0
        total_train_seen_examples = 0.0
        reconstruct_embeddings = True

        roc_auc_metrics = {"auc_test": 0, "auc_val": 0, "auc_train": 0}
        all_test_preds = torch.zeros(
            (data.test_mask.sum().item(), self.output_size),
            dtype=data.y.dtype,
        ).cpu()
        all_val_preds = torch.zeros(
            (data.val_mask.sum().item(), self.output_size),
            dtype=data.y.dtype,
        ).cpu()
        all_train_preds = torch.zeros(
            (data.train_mask.sum().item(), self.output_size),
            dtype=data.y.dtype,
        ).cpu()

        all_test_true = torch.zeros(
            (data.test_mask.sum().item(), self.output_size),
            dtype=data.y.dtype,
        ).cpu()
        all_val_true = torch.zeros(
            (data.val_mask.sum().item(), self.output_size),
            dtype=data.y.dtype,
        ).cpu()
        all_train_true = torch.zeros(
            (data.train_mask.sum().item(), self.output_size),
            dtype=data.y.dtype,
        ).cpu()

        all_test_embeddings = []
        all_val_embeddings = []
        all_train_embeddings = []

        with torch.no_grad():
            start_idx = 0
            for batch in self.test_loader:
                batch = batch.to(self.device)

                updated_outputs_test, embeddings = model(
                    batch.x,
                    batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                )
                updated_outputs_test.squeeze_()
                n = len(batch)

                assert updated_outputs_test.shape == batch.y.squeeze().shape, (
                    f"Model out: {updated_outputs_test.shape}, batch.y: {batch.y.squeeze().shape}"
                )

                loss = criterion(updated_outputs_test, batch.y.squeeze())
                predictions_test = updated_outputs_test

                test_loss += loss.cpu().item() * n

                end_idx = start_idx + n
                _s = slice(start_idx, end_idx)
                all_test_preds[_s, :] = predictions_test.cpu().unsqueeze(-1)
                all_test_true[_s, :] = batch.y.unsqueeze(-1).cpu()
                start_idx = end_idx

                if embeddings is not None:
                    all_test_embeddings.append(embeddings)

                total_test_seen_examples += n

            roc_auc_metrics["auc_test"] += roc_auc(
                all_test_preds,
                all_test_true,
            )

            start_idx = 0
            for batch in self.validation_loader:
                batch = batch.to(self.device)

                updated_outputs_val, embeddings_val = model(
                    batch.x,
                    batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                )
                updated_outputs_val.squeeze_()
                n = len(batch)

                assert updated_outputs_val.shape == batch.y.squeeze().shape, (
                    f"Model out: {updated_outputs_val.shape}, batch.y: {batch.y.squeeze().shape}"
                )

                loss = criterion(updated_outputs_val, batch.y.squeeze())
                predictions_val = updated_outputs_val

                val_loss += loss.cpu().item() * n

                end_idx = start_idx + n
                _s = slice(start_idx, end_idx)
                all_val_preds[_s, :] = predictions_val.cpu().unsqueeze(-1)
                all_val_true[_s, :] = batch.y.unsqueeze(-1).cpu()
                start_idx = end_idx

                if embeddings_val is not None:
                    all_val_embeddings.append(embeddings_val)

                total_val_seen_examples += n

            roc_auc_metrics["auc_val"] += roc_auc(all_val_preds, all_val_true)

            start_idx = 0
            for batch in self.train_eval_loader:
                batch = batch.to(self.device)

                updated_outputs_train, embeddings_train = model(
                    batch.x,
                    batch.edge_index,
                    edge_attr=batch.edge_attr,
                    batch=batch.batch,
                )
                updated_outputs_train.squeeze_()
                n = len(batch)

                assert updated_outputs_train.shape == batch.y.squeeze().shape, (
                    f"Model out: {updated_outputs_train.shape}, batch.y: {batch.y.squeeze().shape}"
                )

                loss = criterion(updated_outputs_train, batch.y.squeeze())
                predictions_train = updated_outputs_train

                train_loss += loss.cpu().item() * n

                end_idx = start_idx + n
                _s = slice(start_idx, end_idx)
                all_train_preds[_s, :] = predictions_train.cpu().unsqueeze(-1)
                all_train_true[_s, :] = batch.y.unsqueeze(-1).cpu()
                start_idx = end_idx

                if embeddings_train is not None:
                    all_train_embeddings.append(embeddings_train)

                total_train_seen_examples += n

            roc_auc_metrics["auc_train"] += roc_auc(all_train_preds, all_train_true)

        metrics = {}
        for k in roc_auc_metrics.keys():
            metrics[k] = roc_auc_metrics[k]

        if len(all_test_embeddings) != 0:
            concatenated_reps_test = [
                torch.cat([t.detach().cpu() for t in tensors if t is not None], dim=0)
                for tensors in zip(*all_test_embeddings)
            ]
        else:
            concatenated_reps_test = None
            reconstruct_embeddings = False

        if len(all_val_embeddings) != 0:
            concatenated_reps_val = [
                torch.cat([t.detach().cpu() for t in tensors if t is not None], dim=0)
                for tensors in zip(*all_val_embeddings)
            ]
        else:
            concatenated_reps_val = None
            reconstruct_embeddings = False

        if len(all_train_embeddings) != 0:
            concatenated_reps_train = [
                torch.cat([t.detach().cpu() for t in tensors if t is not None], dim=0)
                for tensors in zip(*all_train_embeddings)
            ]
        else:
            concatenated_reps_train = None
            reconstruct_embeddings = False

        full_embbeddings = None
        if reconstruct_embeddings:
            full_embbeddings = reconstruct_full_embeddings_dataset(
                concatenated_reps_train,
                concatenated_reps_val,
                concatenated_reps_test,
                data.train_mask.cpu(),
                data.val_mask.cpu(),
                data.test_mask.cpu(),
            )

        all_preds = reconstruct_full_predictions_dataset(
            all_train_preds,
            all_val_preds,
            all_test_preds,
            data.train_mask.cpu(),
            data.val_mask.cpu(),
            data.test_mask.cpu(),
        )

        metrics["test_loss"] = test_loss / total_test_seen_examples
        metrics["validation_loss"] = val_loss / total_val_seen_examples
        metrics["train_loss"] = train_loss / total_train_seen_examples
        metrics["outputs"] = all_preds
        metrics["embeddings"] = full_embbeddings

        return metrics
