import torch

from ..task_strategy import DeterministicTrainingTask
from ...metrics.metrics import rmse, r2

_METRIC_FN_MAP = {
    "RMSE": (rmse, True),
    "R2": (r2, False),
}


class NodeRegressionGCNTask(DeterministicTrainingTask):
    def prepare(self, data, **kwargs):
        self.is_large = kwargs.pop("is_large", False)
        self.target = kwargs.pop("target", None)
        if self.target is not None and self.target == "distribution":
            print("Estimating mean+variance of distribution, not direct predictions!")

    def train_epoch(self, model, optimizer, criterion, data):
        data = data[0].to(self.device)
        model.train()
        optimizer.zero_grad()  # set_to_none=True

        if self.is_large:
            outputs, _ = model(data.x, data.adj.t(), data.edge_weight)
        else:
            outputs, _ = model(data.x, data.edge_index, data.edge_weight)
        outputs = outputs.squeeze()

        new_mask = data.train_mask
        if self.target is not None and self.target == "distribution":
            mu = outputs[:, 0]
            var = torch.nn.functional.softplus(outputs[:, 1]) + 1e-6
            loss = criterion(mu[new_mask], data.y.squeeze()[new_mask], var[new_mask])
        elif self.target is None:
            loss = criterion(
                outputs[new_mask],
                data.y.squeeze()[new_mask],
            )  # Compute the loss solely based on the training nodes.

        loss.backward()
        optimizer.step()
        return loss.detach().cpu().item()

    def val_epoch(self, model, criterion, data):
        data = data[0].to(self.device)
        model.eval()

        return_vals = {}
        with torch.no_grad():
            if self.is_large:
                updated_outputs, embeddings = model(
                    data.x, data.adj.t(), data.edge_weight
                )
            else:
                updated_outputs, embeddings = model(
                    data.x, data.edge_index, data.edge_weight
                )
            updated_outputs = updated_outputs.squeeze()

            if self.target is not None and self.target == "distribution":
                mu = updated_outputs[:, 0]
                var = torch.nn.functional.softplus(updated_outputs[:, 1]) + 1e-6
                predictions = mu
            elif self.target is None:
                predictions = updated_outputs

            for metric_name in _METRIC_FN_MAP:
                metric_fn, _ = _METRIC_FN_MAP[metric_name]
                if metric_name == "RMSE":
                    return_vals[f"test_{metric_name}"] = metric_fn(
                        predictions[data.test_mask].cpu(),
                        observation=data.y[data.test_mask].cpu(),
                        original_std=data.original_std.cpu(),
                    ).item()
                    return_vals[f"train_{metric_name}"] = rmse(
                        predictions[data.train_mask].cpu(),
                        observation=data.y[data.train_mask].cpu(),
                        original_std=data.original_std.cpu(),
                    ).item()
                elif metric_name == "R2":
                    return_vals[f"test_{metric_name}"] = r2(
                        predictions[data.test_mask].cpu(),
                        data.y.squeeze()[data.test_mask].cpu(),
                    )
                    return_vals[f"train_{metric_name}"] = r2(
                        predictions[data.train_mask].cpu(),
                        data.y.squeeze()[data.train_mask].cpu(),
                    )

                return_vals["validation_loss"] = 0
                if data.val_mask.sum() != 0:
                    if self.target is not None and self.target == "distribution":
                        return_vals["validation_loss"] = criterion(
                            mu[data.val_mask],
                            data.y.squeeze()[data.val_mask],
                            var[data.val_mask],
                        ).cpu()
                    elif self.target is None:
                        return_vals["validation_loss"] = criterion(
                            predictions[data.val_mask], data.y.squeeze()[data.val_mask]
                        ).cpu()

        return_vals["embeddings"] = embeddings
        return_vals["outputs"] = updated_outputs
        return return_vals
