import torch
import numpy as np

from scipy.integrate import simpson

from inspector.metrics.metrics import (
    average_set_size,
    classification_nll,
    compute_accuracy,
    confidence_set_coverage,
    multiclass_calibration_error,
    average_precision,
)

from inspector.metrics.store_and_print import (
    plot_calibration_curve,
    plot_histogram,
)

from inspector.metrics.calculate_metrics_base import BaseMetricsComputation

_METRIC_FN_MAP = {
    "Accuracy": (compute_accuracy, False),
    "AP": (average_precision, False),
}


class ClassificationMetrics(BaseMetricsComputation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _process_predictions(self, raw_preds, test_mask, metric):
        # Raw: (Models, Samples, Classes) -> Softmax -> Mean over models -> (Samples, Classes)
        epsilon = 1e-6
        if raw_preds.shape[-1] == 1:
            probabilities = torch.sigmoid(raw_preds)
        else:
            probabilities = torch.nn.functional.softmax(raw_preds, dim=2)

        probabilities = torch.clamp(probabilities, epsilon, 1.0 - epsilon)
        probabilities_out = probabilities  # (Models, Samples, Classes)
        if not self.baseline and not self.best_baseline:
            probabilities_out = probabilities.mean(dim=0)  # (Samples, Classes)

        if self.save_uq_decomposition:
            total_uncertainty = -torch.sum(
                probabilities_out * torch.log(probabilities_out), dim=-1
            )
            aleatoric_uncertainty = 0.0
            aleatoric_uncertainty = -torch.sum(
                probabilities * torch.log(probabilities), dim=-1
            ).mean(dim=0)
            epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

            print("=-------------------------------------=")
            print("Epistemic: ", epistemic_uncertainty.mean())
            print("Aleatoric: ", aleatoric_uncertainty.mean())
            print("Total: ", total_uncertainty.mean())
            print("=-------------------------------------=")
            self.uq_decomposition["aleatoric"] = aleatoric_uncertainty
            self.uq_decomposition["epistemic"] = epistemic_uncertainty
            self.uq_decomposition["total"] = total_uncertainty

        return probabilities_out

    def _compute_nll_for_repetition(
        self, pred, y_true, test_mask, train_mask, val_mask
    ):
        test_metric, train_metric = 0, 0
        if not self.baseline:
            test_metric = self._compute_metric(
                self.main_metric, pred[test_mask], y_true[test_mask]
            )
            train_metric = self._compute_metric(
                self.main_metric, pred[train_mask], y_true[train_mask]
            )
            # squeeze() has no effect if last dim > 1 so it does not affect multiclass
            nll_test = classification_nll(
                pred[test_mask].squeeze(), y_true[test_mask]
            )
        else:
            nll_test = classification_nll(
                pred[:, test_mask, :].squeeze(), y_true[test_mask]
            )

        return {
            f"{self.main_metric} Test": test_metric,
            f"{self.main_metric} Train": train_metric,
            "NLL": nll_test,
        }

    def _compute_uq_for_repetition(self, rep_idx, preds, y_true, test_mask, stats_path):
        preds = preds[test_mask, :]
        y_true = y_true[test_mask]

        N_CLASSES = preds.size(1)
        results = {}

        # ECE and Calibration Curves
        # multiclass_calibration_error returns global scalar AND per-class curve data
        if preds.shape[-1] == 1:
            ecme, raw_coverage_per_class = multiclass_calibration_error(
                1 - preds, y_true
            )
        else:
            ecme, raw_coverage_per_class = multiclass_calibration_error(preds, y_true)
        results["ECE"] = ecme

        # Process per-class calibration curves and coverage
        scaled_cal_list = []
        raw_cal_list = []
        specific_coverage = {90: [], 95: [], 99: []}

        for jj in range(N_CLASSES):
            class_data = np.array(raw_coverage_per_class[jj])

            observed = class_data[:, 0]
            expected = class_data[:, 1]
            bin_end_value = class_data[:, 2]

            r_max, r_min = 0.5, 0
            deviation = np.abs(observed - expected)
            area_under_deviation = simpson(y=deviation, x=expected)
            scaled_area = (area_under_deviation - r_min) / (r_max - r_min)

            scaled_cal_list.append(scaled_area)
            raw_cal_list.append(area_under_deviation)

            plot_calibration_curve(
                observed,
                expected,
                stats_path,
                f"calibration_curve_prob_model-{rep_idx}_class-{jj}_{self.uq_type.lower()}_{self.actual_num_models}x{self.actual_num_repetitions}",
                "Calibration Curve - Probability",
            )

            # Specific Coverage for relevant quantiles
            for k in specific_coverage.keys():
                covered_bins_index = bin_end_value <= k / 100
                if np.all(covered_bins_index):
                    _valid_index = covered_bins_index.shape[0]
                else:
                    _valid_index = np.argmin(covered_bins_index)
                cumulative_observed = observed[:_valid_index].mean()
                cumulative_expected = expected[:_valid_index].mean()
                specific_coverage[k].append(cumulative_expected - cumulative_observed)

        results["Confidence Set Coverage"] = np.array(
            confidence_set_coverage(preds.numpy(), y_true.numpy())
        )

        results["Calibration"] = np.array(scaled_cal_list)
        results["Raw Calibration"] = np.array(raw_cal_list)

        # Sharpness
        # Was (Max Prob - 1/K). Is 1 - Max Prob so 0 good 1 bad.
        pred_proba_flat = 1 - torch.flatten(torch.max(preds, dim=1).values)
        results["Mean Sharpness"] = torch.mean(pred_proba_flat).item()
        results["Std Sharpness"] = torch.std(pred_proba_flat).item()

        plot_histogram(
            pred_proba_flat.numpy(),
            stats_path,
            f"sharpness_model-{rep_idx}_{self.uq_type.lower()}_{self.actual_num_models}x{self.actual_num_repetitions}",
            "Distribution of Predicted Uncertainty",
        )

        # Intervals (Set Size)
        set_size = average_set_size(preds)
        # avg_set_size returns list corresponding to [[q90_m, q90_s], [q95_m, q95_s], ...]
        quantiles = [90, 95, 99]
        for i, q in enumerate(quantiles):
            results[f"Q{q} Mean Interval Width"] = np.array(set_size[i][0])
            results[f"Q{q} Std Interval Width"] = np.array(set_size[i][1])
            results[f"Q{q} Coverage Error"] = np.array(specific_coverage[q])

        return results

    def _compute_extra_for_repetition(self, pred, y_true, test_mask, train_mask):
        """
        Computes every metric listed in other_metrics for this dataset.
        Returns a dict: {
            "<MetricName> Test":  np.array of shape (output_dim,),
            "<MetricName> Train": np.array of shape (output_dim,),
        }
        """
        results = {}

        for metric_name in self.other_metrics:
            train_result = self._compute_metric(
                metric_name, pred[train_mask], y_true[train_mask]
            )
            test_result = self._compute_metric(
                metric_name, pred[test_mask], y_true[test_mask]
            )

            if train_result is not None:
                results[f"{metric_name} Train"] = train_result
            if test_result is not None:
                results[f"{metric_name} Test"] = test_result

        return results

    def _compute_metric(self, metric_name: str, pred, y_true) -> np.ndarray:
        """
        Computes a single metric for given predictions and ground truth.

        Args:
            metric_name: Name of the metric to compute (must be in _METRIC_FN_MAP).
            pred:        Model predictions tensor.
            y_true:      Ground truth tensor.

        Returns:
            np.ndarray of shape (output_dim,) with the computed metric value(s).
        """
        if metric_name not in _METRIC_FN_MAP:
            raise NotImplementedError(
                f"Extra metric '{metric_name}' is not in _METRIC_FN_MAP. "
                f"Available: {list(_METRIC_FN_MAP.keys())}"
            )

        metric_fn, _ = _METRIC_FN_MAP[metric_name]
        is_multiclass = pred.shape[-1] > 1

        if metric_name == "Accuracy":
            preds_transformed = (
                pred.argmax(dim=1) if is_multiclass else (pred > 0.5).type(torch.long).squeeze(-1)
            )
            return metric_fn(preds_transformed, y_true)

        if metric_name == "AP" and not is_multiclass:
            return metric_fn(pred.squeeze(), y_true)

        return None

    def _forge_trivial_data(self, y_true, train_mask, test_mask, metric, **kwargs):
        if y_true.dtype != torch.long:
            print("Y true was not long, converting ....")
            y_true = y_true.to(torch.long)

        class_counts = torch.bincount(y_true[train_mask])
        class_priors = class_counts.float() / class_counts.sum()

        # Expand to a "full dataset" (N_samples, N_classes)
        trivial_probs = class_priors.unsqueeze(0).repeat(y_true.size(0), 1)
        if trivial_probs.shape[-1] == 2:
            trivial_probs = trivial_probs[:, 1].unsqueeze(-1)

        # Th _compute_nll_for_repetition uses .argmax(dim=1).
        # Since trivial_probs is constant, argmax will correctly pick the majority class. I hope!

        return trivial_probs
