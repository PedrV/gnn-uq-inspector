import torch
import numpy as np

from scipy.integrate import simpson

from sklearn.covariance import LedoitWolf

from inspector.metrics.metrics import (
    average_interval_width,
    central_quantile_coverage_error,
    cov_per_dim,
    diag_cov_nll,
    full_cov_nll,
    r2,
    rmse,
)

from inspector.metrics.store_and_print import (
    plot_calibration_curve,
    plot_histogram,
)
from inspector.metrics.calculate_metrics_base import BaseMetricsComputation


_METRIC_FN_MAP = {
    "RMSE": (rmse, True),
    "R2": (r2, False),
}


class RegressionMetrics(BaseMetricsComputation):
    def __init__(self, *args, original_std, original_mean, stable=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.original_std = original_std
        self.original_mean = original_mean
        self.stable = stable

    def _process_predictions(self, raw_preds, test_mask, metric):
        # raw_preds: List of tensors [N_models, N_samples, Dim]
        # But here 'raw_preds' passed from loop is ONE repetition: (N_models, N_samples, Dim)

        # Ensure correct shape manipulation: permute to (Samples, Models, Dim)
        preds = raw_preds.permute(1, 0, 2)
        test_examples = test_mask.sum()

        if not self.distribution_output:
            mean_pred = preds.mean(dim=1)
            std_pred = preds.std(dim=1)
        else:
            mus = preds[:, :, 0]  # (N_samples, N_models)
            s2 = preds[:, :, 1]  # (N_samples, N_models)
            s2 = torch.nn.functional.softplus(s2) + 1e-6

            mean_pred = mus.mean(dim=1).unsqueeze(-1)  # (N_samples,1)

            total_var = (s2 + mus**2).mean(dim=1) - mean_pred.squeeze() ** 2
            std_pred_total = total_var.clamp(min=1e-10).sqrt().unsqueeze(-1)
            if self.baseline:
                self.baseline_regression_single_models_std = s2.sqrt()

            epistemic_var = (mus**2).mean(dim=1) - mean_pred.squeeze() ** 2
            std_pred_epis = epistemic_var.clamp(min=1e-10).sqrt().unsqueeze(-1)

            aleatoric_var = s2.mean(dim=1)
            std_pred_alea = aleatoric_var.clamp(min=1e-10).sqrt().unsqueeze(-1)

            if self.save_uq_decomposition:
                print("=-------------------------------------=")
                print("Epistemic: ", epistemic_var.mean())
                print("Aleatoric: ", aleatoric_var.mean())
                print("Total: ", total_var.mean())
                print("=-------------------------------------=")
                self.uq_decomposition["aleatoric"] = std_pred_alea**2
                self.uq_decomposition["epistemic"] = std_pred_epis**2
                self.uq_decomposition["total"] = std_pred_total**2

            if self.uq_type.lower() == "total":
                std_pred = std_pred_total
            elif self.uq_type.lower() == "aleatoric":
                std_pred = std_pred_alea
            elif self.uq_type.lower() == "epistemic":
                std_pred = std_pred_epis
            else:
                raise NotImplementedError(
                    f"Do not know what {self.uq_type} uncertainty is."
                )

            test_cov = torch.zeros((test_examples, test_examples, self.output_dim))
            for jj in range(self.output_dim):
                test_cov[:, :, jj] = torch.diag(std_pred[test_mask] ** 2)

        if metric in ("uq", "extra"):
            # UQ metrics and extra usually just need mean and std (diagonal approximation)
            return {"mean": mean_pred, "std": std_pred}

        # For NLL, we calculate Full Covariance
        if self.stable:
            test_cov = torch.zeros((test_examples, test_examples, self.output_dim))
            for jj in range(self.output_dim):
                # LedoitWolf estimation for numerical stability
                _lw_cov = LedoitWolf().fit(preds[test_mask, :, jj])
                test_cov[:, :, jj] = torch.from_numpy(_lw_cov.covariance_)
        else:
            test_cov = cov_per_dim(preds[test_mask, :, :])
            jitter = (
                torch.eye(test_cov.shape[0]).unsqueeze(-1).repeat(1, 1, self.output_dim)
                * 1e-4
            )
            test_cov += jitter

        return {"mean": mean_pred, "std": std_pred, "cov": test_cov}

    def _compute_nll_for_repetition(
        self, preds, y_true, test_mask, train_mask, val_mask
    ):
        pred = preds["mean"]
        std = preds["std"]
        test_cov = preds["cov"]
        if self.best_baseline:
            std.fill_(
                torch.sqrt(torch.mean((pred[test_mask] - y_true[test_mask]) ** 2))
            )
        elif self.baseline:
            # std.fill_(torch.sqrt(torch.mean((pred[val_mask] - y_true[val_mask]) ** 2)))
            std = self.baseline_regression_single_models_std[:, self.baseline_single_model_id].unsqueeze(-1)

        metrics = {
            self.main_metric + " Test": np.zeros(self.output_dim),
            self.main_metric + " Train": np.zeros(self.output_dim),
            "Diagonal NLL": np.zeros(self.output_dim),
            "Full NLL": np.zeros(self.output_dim),
        }
        fn, needs_std = _METRIC_FN_MAP[self.main_metric]

        for jj in range(self.output_dim):
            if needs_std:
                metrics[f"{self.main_metric} Test"][jj] = fn(
                    pred[test_mask, jj], y_true[test_mask, jj], self.original_std[jj]
                )
                metrics[f"{self.main_metric} Train"][jj] = fn(
                    pred[train_mask, jj], y_true[train_mask, jj], self.original_std[jj]
                )
            else:
                metrics[f"{self.main_metric} Test"][jj] = fn(
                    pred[test_mask, jj], y_true[test_mask, jj]
                )
                metrics[f"{self.main_metric} Train"][jj] = fn(
                    pred[train_mask, jj], y_true[train_mask, jj]
                )

            try:
                metrics["Diagonal NLL"][jj] = diag_cov_nll(
                    pred[test_mask, jj], std[test_mask, jj], y_true[test_mask, jj]
                )
            except ValueError:
                metrics["Diagonal NLL"][jj] = np.inf

            try:
                metrics["Full NLL"][jj] = full_cov_nll(
                    pred[test_mask, jj], test_cov[:, :, jj], y_true[test_mask, jj]
                )
            except ValueError:
                metrics["Full NLL"][jj] = np.inf

        return metrics

    def _compute_extra_for_repetition(self, preds, y_true, test_mask, train_mask):
        """
        Computes every metric listed in other_metrics for this dataset.
        Returns a dict: {
            "<MetricName> Test":  np.array of shape (output_dim,),
            "<MetricName> Train": np.array of shape (output_dim,),
        }
        """
        pred = preds["mean"]  # (N_samples, Dim)
        results = {}

        for metric_name in self.other_metrics:
            if metric_name not in _METRIC_FN_MAP:
                raise NotImplementedError(
                    f"Extra metric '{metric_name}' is not in _METRIC_FN_MAP. "
                    f"Available: {list(_METRIC_FN_MAP.keys())}"
                )

            fn, needs_std = _METRIC_FN_MAP[metric_name]

            test_vals = np.zeros(self.output_dim)
            train_vals = np.zeros(self.output_dim)

            for jj in range(self.output_dim):
                pred_test = pred[test_mask, jj]
                pred_train = pred[train_mask, jj]
                y_test = y_true[test_mask, jj]
                y_train = y_true[train_mask, jj]

                if needs_std:
                    # rmse signature: (prediction, observation, original_std)
                    std_jj = self.original_std[jj]
                    test_val = fn(pred_test, y_test, std_jj)
                    train_val = fn(pred_train, y_train, std_jj)
                else:
                    test_val = fn(pred_test, y_test)
                    train_val = fn(pred_train, y_train)

                # Unpack tensor/scalar to a plain float
                test_vals[jj] = (
                    float(test_val) if not isinstance(test_val, float) else test_val
                )
                train_vals[jj] = (
                    float(train_val) if not isinstance(train_val, float) else train_val
                )

            results[f"{metric_name} Test"] = test_vals
            results[f"{metric_name} Train"] = train_vals

        return results

    def _compute_uq_for_repetition(self, rep_idx, preds, y_true, test_mask, stats_path):
        mean_pred = preds["mean"][test_mask, :]
        std_pred = preds["std"][test_mask, :]
        y_true = y_true[test_mask, :]

        results = {}

        ece_list = []
        raw_cal_list = []

        # Interval Width Containers
        q_widths = {90: [], 95: [], 99: []}
        q_stds = {90: [], 95: [], 99: []}
        specific_coverage = {90: [], 95: [], 99: []}

        expected_quantiles = np.arange(0, 1 + 0.01, 0.01)

        for jj in range(self.output_dim):
            # Create Distribution ONCE per dimension
            m = mean_pred[:, jj]
            s = std_pred[:, jj]

            pred_dist = torch.distributions.multivariate_normal.MultivariateNormal(
                m, torch.diag(s**2)
            )

            # Calibration Curve & ECE
            curve_results = []
            for q in expected_quantiles:
                curve_results.append(
                    central_quantile_coverage_error(
                        pred_dist, y_true[:, jj].reshape(-1, 1), quantile=q
                    ).item()
                )
            observed_quantiles = np.array(curve_results)

            # Area Calculation
            r_max, r_min = 0.5, 0
            deviation = np.abs(observed_quantiles - expected_quantiles)
            area_under_deviation = simpson(y=deviation, x=expected_quantiles)
            scaled_area = (area_under_deviation - r_min) / (r_max - r_min)

            ece_list.append(scaled_area)  # For regression, scaled area is ECE
            raw_cal_list.append(area_under_deviation)

            plot_calibration_curve(
                observed_quantiles,
                expected_quantiles,
                stats_path,
                f"calibration_curve_centred_model-{rep_idx}_dim_{jj}_{self.uq_type.lower()}_{self.actual_num_models}x{self.actual_num_repetitions}",
                "Calibration Curve - Centred Interval",
            )

            plot_histogram(
                s.numpy(),
                stats_path,
                f"sharpness_model-{rep_idx}_dim_{jj}",
                "Distribution of Predicted Uncertainty (STD**2)",
            )

            # Specific Coverage for relevant quantiles
            for e, o in zip(expected_quantiles, observed_quantiles):
                if int(e * 100) in specific_coverage.keys():
                    specific_coverage[int(e * 100)].append(e - o)

            # Intervals
            # average_interval_width returns [(q90_m, q90_s), (q95_m, q95_s), ...]
            width_stats = average_interval_width(pred_dist)

            q_widths[90].append(width_stats[0][0])
            q_stds[90].append(width_stats[0][1])
            q_widths[95].append(width_stats[1][0])
            q_stds[95].append(width_stats[1][1])
            q_widths[99].append(width_stats[2][0])
            q_stds[99].append(width_stats[2][1])

        results["ECE"] = np.array(ece_list)
        results["Raw Calibration"] = np.array(raw_cal_list)

        results["Mean Sharpness"] = std_pred.mean(dim=0).numpy()
        results["Std Sharpness"] = std_pred.std(dim=0).numpy()

        for q in [90, 95, 99]:
            results[f"Q{q} Mean Interval Width"] = np.array(q_widths[q])
            results[f"Q{q} Std Interval Width"] = np.array(q_stds[q])
            results[f"Q{q} Coverage Error"] = np.array(specific_coverage[q])

        return results

    def _forge_trivial_data(self, y_true, train_mask, test_mask, metric, **kwargs):
        # We need shapes covering the full dataset (to allow train/test masking later)
        # NLL needs Train metrics too, so we can't just slice Test here.
        # Notice that we can't use self.original_std and original_mean because they
        # translate back to the original location and spread, we need location and spread that
        # match the ones present in the data.

        use_this_std = kwargs.get("use_this_std", 1)
        use_this_mean = kwargs.get("use_this_mean", 0)

        # Standard Normal Baseline or Normal based given mean and std
        trivial_mean = torch.zeros_like(y_true) + use_this_mean
        trivial_std = torch.ones_like(y_true) * use_this_std

        data = {"mean": trivial_mean, "std": trivial_std}

        if metric == "nll":
            num_samples = test_mask.sum()
            # shape (N_samples, Dim, Dim)
            trivial_cov = torch.eye(num_samples) * use_this_std**2
            trivial_cov_expanded = trivial_cov.unsqueeze(2).expand(
                num_samples, num_samples, self.output_dim
            )
            data["cov"] = trivial_cov_expanded

        return data

