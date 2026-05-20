import torch
import numpy as np

from sklearn.metrics import average_precision_score, r2_score


def cov_per_dim(x, dim: int = 2, out_dim: int = 0):
    assert x.dim() == 3, f"Expected 3D tensor, got {x.dim()}D"
    assert dim != out_dim, f"dim and out_dim must be different, got {dim} and {out_dim}"

    result = torch.zeros(x.size(out_dim), x.size(out_dim), x.size(dim))
    for i in range(x.size(dim)):
        _x = torch.index_select(x, dim=dim, index=torch.tensor([i])).squeeze(dim)
        _x = _x.movedim(out_dim if out_dim < dim else out_dim - 1, 0).reshape(
            x.size(out_dim), -1
        )
        result[:, :, i] = torch.cov(_x)

    return result


def average_precision(pred_y, y):
    if isinstance(pred_y, torch.Tensor):
        pred_y = pred_y.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    assert y.shape == pred_y.shape

    return average_precision_score(y_true=y, y_score=pred_y)


def r2(pred_y, y):
    if isinstance(pred_y, torch.Tensor):
        pred_y = pred_y.numpy()
    if isinstance(y, torch.Tensor):
        y = y.numpy()

    assert y.shape == pred_y.shape

    return r2_score(y_true=y, y_pred=pred_y)


def rmse(prediction, observation, original_std):
    if len(prediction.shape) == 1:
        prediction = prediction.unsqueeze(1)
    if len(observation.shape) == 1:
        observation = observation.unsqueeze(1)

    assert prediction.shape == observation.shape

    result = torch.linalg.norm(observation - prediction, dim=0)
    result = original_std * result / np.sqrt(len(observation))
    return result


def mse(prediction, observation, original_std):
    if len(prediction.shape) == 1:
        prediction = prediction.unsqueeze(1)
    if len(observation.shape) == 1:
        observation = observation.unsqueeze(1)

    assert prediction.shape == observation.shape

    result = torch.sum((observation - prediction) ** 2, dim=0)
    result = (original_std**2) * result / len(observation)
    return result


def compute_accuracy(pred_y, y):
    return (pred_y == y).sum() / y.size(0)


def full_cov_nll(mean_vector, cov_matrix, observation):
    """
    Say we have (double check the shapes though, might as well `.squeeze()` all
    the tensors)
    > f_preds = model(xs)  # prediction without observation noise
    > y_preds = likelihood(model(xs))  # prediction with observation noise
    > mean_vector = y_preds.mean
    > cov_matrix = y_preds.covariance_matrix

    Then `full_cov_nll(mean_vector, cov_matrix, observation)` should be equal to
    > mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    > -mll(f_preds, observation)*len(observation)  # note: f_preds, not y_preds
    It should also be equal to
    > -y_preds.log_prob(observation)
    """
    mean_vector = mean_vector.squeeze()
    cov_matrix = cov_matrix.squeeze()
    observation = observation.squeeze()

    assert len(mean_vector.shape) == 1
    assert len(cov_matrix.shape) == 2
    assert len(observation.shape) == 1
    assert mean_vector.shape == observation.shape
    assert cov_matrix.shape[0] == cov_matrix.shape[1]
    assert cov_matrix.shape[0] == mean_vector.shape[0]

    distribution = torch.distributions.multivariate_normal.MultivariateNormal(
        mean_vector, cov_matrix
    )
    return -distribution.log_prob(observation)


def diag_cov_nll(mean_vector, stds_vector, observation):
    """
    Say we have (double check the shapes though, might as well `.squeeze()` all
    the tensors)
    > f_preds = model(xs)  # prediction without observation noise
    > y_preds = likelihood(model(xs))  # prediction with observation noise
    > mean_vector = y_preds.mean
    > cov_matrix = y_preds.covariance_matrix

    Then `diag_cov_nll(mean_vector, stds_vector, observation)` should be equal
    to
    > torch.nn.functional.gaussian_nll_loss(
          mean_vector,
          observation,
          stds_vector**2, ## notice the power 2
          reduction="sum",
          full=True # this adds the "len(observation)/2*np.log(2*np.pi)" const
      ))

    """
    mean_vector = mean_vector.squeeze()
    stds_vector = stds_vector.squeeze()
    observation = observation.squeeze()

    assert len(mean_vector.shape) == 1
    assert len(stds_vector.shape) == 1
    assert len(observation.shape) == 1
    assert mean_vector.shape == stds_vector.shape
    assert stds_vector.shape == observation.shape

    distribution = torch.distributions.multivariate_normal.MultivariateNormal(
        mean_vector, torch.diag(stds_vector**2)
    )
    return -distribution.log_prob(observation)


def classification_nll(mean_prob, observation, epsilon=1e-12):
    """
    Unified NLL for Binary/Multiclass and Single/Ensemble inputs.

    Shapes supported:
    - Binary Single:   prob (N,),       obs (N,)
    - Binary Ensemble: prob (M, N),    obs (N,)
    - Multi Single:    prob (N, C),    obs (N,)
    - Multi Ensemble:  prob (M, N, C), obs (N,)
    Note: The baselines are assumed (Multi/Binary) Ensemble since each ensemble is made of a single model.
          As for the "actual ensembles", when baseline=False, they are considered (Binary/Multi) Single.
          In all cases the std etc. result purely from the call in `for j, preds_processed in enumerate(ensemble_data)`.
          In the case of (Multi/Binary) Ensemble maybe we could have used the M dimension to calculate distribution statistics.
          Regardless, I think this is just a detail.
    """
    mean_prob = torch.clamp(mean_prob, epsilon, 1.0 - epsilon)

    # Multiclass happens if:
    # - Single Model: prob (N, C) vs obs (N) -> ndim diff is 1 AND last dim != obs size
    # - Ensemble: prob (M, N, C) vs obs (N) -> ndim diff is 2
    is_multiclass = (
        mean_prob.ndim == observation.ndim + 1
        and mean_prob.shape[-1] != observation.shape[-1]
    ) or (mean_prob.ndim == observation.ndim + 2)

    if not is_multiclass:
        # Binary Cross Entropy Case
        # Standard: -(y*log(p) + (1-y)*log(1-p))
        nlls = -(
            observation * torch.log(mean_prob)
            + (1 - observation) * torch.log(1 - mean_prob)
        )
        return nlls.mean()

    else:
        # We need to extract the probability of the correct class
        # observation shape is (N,), we need to index the last dimension of mean_prob
        # We use gather to handle any number of leading dimensions (Models, Examples, etc.)
        # Reshape observation to match mean_prob's rank for gathering
        target = (
            observation.view(1, -1, 1)
            if mean_prob.ndim == 3
            else observation.view(-1, 1)
        )

        # If 3D (M, N, C), we need to broadcast target to (M, N, 1)
        if mean_prob.ndim == 3:
            target = target.expand(mean_prob.shape[0], -1, -1)

        # Gather the probabilities of the observed classes
        correct_class_probs = torch.gather(mean_prob, dim=-1, index=target)
        return -torch.log(correct_class_probs).mean()


def _bin_calibration_error(confidences, correctness, n_bins=10, nu=2):
    """
    confidences: array of shape (N,)
    correctness: array of shape (N,), 1 if correct, 0 otherwise
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(confidences, bins, right=True) - 1

    errors = []
    weights = []
    raw_values = []

    # print(confidences, correctness)
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        bin_conf = confidences[mask].mean()  # beta
        bin_acc = correctness[mask].mean()  # alpha
        raw_values.append([bin_acc, bin_conf, bins[b + 1]])
        # print(bins[b], confidences[mask], correctness[mask])
        errors.append(np.abs(bin_acc - bin_conf) ** nu)
        weights.append(mask.mean())

    return (np.sum(np.array(weights) * np.array(errors))) ** (1.0 / nu), raw_values


def multiclass_calibration_error(probs, labels, n_bins=100, nu=2):
    """
    N: number of examples; K: number of classes

    probs: shape (N, K)
    labels: shape (N,), integer class labels
    """
    N, K = probs.shape
    mce_terms = []
    raw_values_per_class = []

    for k in range(K):
        p_k = probs[:, k]
        y_k = (labels == k).to(torch.float32)

        ce_k, raw_values_k = _bin_calibration_error(p_k, y_k, n_bins=n_bins, nu=nu)
        raw_values_per_class.append(raw_values_k)
        mce_terms.append(ce_k**nu)

    mce_terms = np.array(mce_terms)
    weights = np.ones(K) / K

    return (np.sum(weights * mce_terms)) ** (1.0 / nu), raw_values_per_class


def average_set_size(preds):
    # Handle binary classification with shape (N, 1)
    if preds.shape[1] == 1:
        preds = np.concatenate([1 - preds, preds], axis=1)

    N, N_CLASSES = preds.shape[0], preds.shape[1]

    desired_ranges = [0.9, 0.95, 0.99]
    _set_sizes = [[]] * len(desired_ranges)

    y_pred_proba_sorted = np.sort(preds, axis=1)[:, ::-1]
    cum_probas = np.cumsum(y_pred_proba_sorted, axis=1)

    for j in range(N):
        for k, target_range in enumerate(desired_ranges):
            if (cum_probas[j] >= target_range).any():
                set_size = np.argmax(cum_probas[j] >= target_range) + 1
            else:
                # Should only happen if target_range > 1.0, but safety margin is N_CLASSES
                set_size = N_CLASSES

            _set_sizes[k].append(set_size)

    return [[np.mean(s), np.std(s)] for s in _set_sizes]


def central_quantile_coverage_error(pred_dist, y_true, quantile: float = 0.95):
    """
    pred_dist : Predictive Distribution. Typically torch.distributions.MultivariateNormal.
    y_true : True values. NxC where N is the number of examples and C the number of dimensions.
    quantile : Quantile to use.

    Caveat, this is only well defined for C = 1

    Adapted from https://github.com/cornellius-gp/gpytorch/blob/73a98a5798b1627b4ebd67ddf8305f714722087c/gpytorch/metrics/metrics.py
    If the input is 90, the query for ICDF is 0.95.
    We are thus asking for the quantity that covers 95% of the interval 0 through 95.
    By using this deviation, because it is from a central approach (computing +- the mean),
    this will give a lower value that covers up to 5% (lower tail) and a upper value that covers
    everything except the last 5%. Effectively making a symmetric interval around the mean.
    TODO: Use <= >= instead of < >?
    """
    if quantile < 0 or quantile > 1:
        raise NotImplementedError("Quantile must be between 0 and 1")

    if y_true.shape[1] != 1:
        raise NotImplementedError(
            f"Input number of dimensions must be 1, got {y_true.shape[1]}"
        )

    combine_dim = 0
    N = y_true.shape[combine_dim]

    standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    deviation = standard_normal.icdf(torch.as_tensor(0.5 + 0.5 * quantile))

    # preds_dist.mean is of shape (N,)
    lower = pred_dist.mean - deviation * pred_dist.stddev
    upper = pred_dist.mean + deviation * pred_dist.stddev

    n_samples_within_bounds = (
        (y_true > lower.reshape(N, -1)) & (y_true < upper.reshape(N, -1))
    ).sum(combine_dim)
    fraction = n_samples_within_bounds / y_true.shape[combine_dim]

    return fraction


def one_side_quantile_coverage_error(pred_dist, y_true, quantile: float = 0.95):
    """
    pred_dist : Predictive Distribution. Typically torch.distributions.MultivariateNormal.
    y_true : True values. NxC where N is the number of examples and C the number of dimensions.
    quantile : Quantile to use.

    Caveat, this is only well defined for C = 1
    """
    if quantile < 0 or quantile > 1:
        raise NotImplementedError("Quantile must be between 0 and 1")

    if y_true.shape[1] != 1:
        raise NotImplementedError(
            f"Input number of dimensions must be 1, got {y_true.shape[1]}"
        )

    combine_dim = 0
    N = y_true.shape[combine_dim]

    standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)
    deviation = standard_normal.icdf(torch.as_tensor(quantile))

    # preds_dist.mean is of shape (N,)
    upper = pred_dist.mean + deviation * pred_dist.stddev

    n_samples_within_bounds = ((y_true < upper.reshape(N, -1))).sum(combine_dim)
    fraction = n_samples_within_bounds / y_true.shape[combine_dim]

    return fraction


def average_interval_width(pred_dist):
    desired_ranges = [0.9, 0.95, 0.99]
    standard_normal = torch.distributions.Normal(loc=0.0, scale=1.0)

    interval_width = []
    for q in desired_ranges:
        deviation = standard_normal.icdf(torch.as_tensor(0.5 + 0.5 * q))

        lower = pred_dist.mean - deviation * pred_dist.stddev
        upper = pred_dist.mean + deviation * pred_dist.stddev

        _m = torch.mean(upper - lower).item()
        _s = torch.std(upper - lower).item()
        interval_width.append([_m, _s])

    return interval_width


def confidence_set_coverage(preds, y_true):
    N, N_CLASSES = preds.shape[0], preds.shape[1]

    desired_ranges = [0.9, 0.95, 0.99]
    _sets = [0] * len(desired_ranges)

    y_pred_proba_sorted = np.sort(preds, axis=1)[:, ::-1]
    y_pred_proba_index_sorted = np.argsort(preds, axis=1)[:, ::-1]
    cum_probas = np.cumsum(y_pred_proba_sorted, axis=1)

    for j in range(N):
        for k, target_range in enumerate(desired_ranges):
            if (cum_probas[j] >= target_range).any():
                set_size = np.argmax(cum_probas[j] >= target_range) + 1
            else:
                # Should only happen if target_range > 1.0, but safety margin is N_CLASSES
                set_size = N_CLASSES

            _confidence_set = y_pred_proba_index_sorted[j, :set_size]
            # print(_confidence_set, y_true[j])
            if np.any(_confidence_set == y_true[j]):
                _sets[k] += 1

    return [s / N - k for s, k in zip(_sets, desired_ranges)]
