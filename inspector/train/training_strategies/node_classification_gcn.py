import torch

from ..task_strategy import DeterministicTrainingTask
from ...metrics.metrics import compute_accuracy, average_precision

_METRIC_FN_MAP = {
    "Accuracy": (compute_accuracy, False),
    "AP": (average_precision, False),
}


class NodeClassificationGCNTask(DeterministicTrainingTask):
    def prepare(self, data, **kwargs):
        self.is_large = kwargs.pop("is_large", False)

    def train_epoch(self, model, optimizer, criterion, data):
        data = data[0].to(self.device)
        model.train()
        optimizer.zero_grad()

        if self.is_large:
            out, _ = model(data.x, data.adj.t())
        else:
            out, _ = model(data.x, data.edge_index)
        out.squeeze_()  # Squeeze for BCELL

        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        return loss.detach().cpu().item()

    def val_epoch(self, model, criterion, data):
        data = data[0].to(self.device)
        model.eval()

        return_vals = {}
        with torch.no_grad():
            if self.is_large:
                updated_outputs, embeddings = model(data.x, data.adj.t())
            else:
                updated_outputs, embeddings = model(data.x, data.edge_index)
            updated_outputs.squeeze_()

            is_multiclass = updated_outputs.ndim > 1
            for metric_name in _METRIC_FN_MAP:
                metric_fn, _ = _METRIC_FN_MAP[metric_name]
                if metric_name == "Accuracy" and is_multiclass:
                    return_vals[f"train_{metric_name}"] = metric_fn(
                        updated_outputs.argmax(dim=1)[data.train_mask].cpu(),
                        data.y[data.train_mask].cpu(),
                    )
                    return_vals[f"test_{metric_name}"] = metric_fn(
                        updated_outputs.argmax(dim=1)[data.test_mask].cpu(),
                        data.y[data.test_mask].cpu(),
                    )
                elif metric_name == "Accuracy":
                    return_vals[f"train_{metric_name}"] = metric_fn(
                        (updated_outputs > 0.5).type(torch.long)[data.train_mask].cpu(),
                        data.y[data.train_mask].cpu(),
                    )
                    return_vals[f"test_{metric_name}"] = metric_fn(
                        (updated_outputs > 0.5).type(torch.long)[data.test_mask].cpu(),
                        data.y[data.test_mask].cpu(),
                    )

                if metric_name == "AP" and not is_multiclass:
                    return_vals[f"train_{metric_name}"] = metric_fn(
                        updated_outputs[data.train_mask].cpu(),
                        data.y[data.train_mask].cpu(),
                    )
                    return_vals[f"test_{metric_name}"] = metric_fn(
                        updated_outputs[data.test_mask].cpu(),
                        data.y[data.test_mask].cpu(),
                    )

                if is_multiclass:
                    return_vals["validation_loss"] = criterion(
                        updated_outputs.squeeze()[data.val_mask],
                        data.y.squeeze()[data.val_mask],
                    ).cpu()
                else:
                    return_vals["validation_loss"] = criterion(
                        updated_outputs[data.val_mask], data.y[data.val_mask]
                    ).cpu()

        return_vals["embeddings"] = embeddings
        return_vals["outputs"] = updated_outputs
        return return_vals
