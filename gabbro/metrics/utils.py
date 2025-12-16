"""Utility functions for metrics."""

import numpy as np
import scipy
import torch


def quantiled_kl_divergence(
    sample_ref: np.ndarray,
    sample_approx: np.ndarray,
    n_bins: int = 30,
    return_bin_edges=False,
    return_zero_if_nan_or_inf=False,
):
    """Calculate the KL divergence using quantiles on sample_ref to define the bounds.

    Parameters
    ----------
    sample_ref : np.ndarray
        The first sample to compare (this is the reference, so in the context of
        jet generation, those are the real jets).
    sample_approx : np.ndarray
        The second sample to compare (this is the model/approximation, so in the
        context of jet generation, those are the generated jets).
    n_bins : int
        The number of bins to use for the histogram. Those bins are defined by
        equiprobably quantiles of sample_ref.
    return_bin_edges : bool, optional
        If True, return the bins used to calculate the KL divergence.
    return_zero_if_nan_or_inf : bool, optional
        If True, return 0 if the KL divergence is NaN or inf. Default is False.
    """
    bin_edges = np.quantile(sample_ref, np.linspace(0.0, 1.0, n_bins + 1))
    bin_edges[0] = float("-inf")
    bin_edges[-1] = float("inf")
    pk = np.histogram(sample_ref, bin_edges)[0] / len(sample_ref)
    qk = np.histogram(sample_approx, bin_edges)[0] / len(sample_approx)
    kl = scipy.stats.entropy(pk, qk)
    # check if kl is nan or inf, if it is, return 0
    if return_zero_if_nan_or_inf and (np.isnan(kl) or np.isinf(kl)):
        kl = 0
    if return_bin_edges:
        return kl, bin_edges
    return kl


def calc_quantiled_kl_divergence_for_dict(
    dict_reference: dict,
    dict_approx: dict,
    names: list,
    n_bins: int = 30,
    return_zero_if_nan_or_inf=False,
):
    """Calculate the quantiled KL divergence for two dictionaries of samples.

    Parameters
    ----------
    dict_reference : dict
        The first dictionary of samples.
    dict_approx : dict
        The second dictionary of samples.
    names : list
        The names of the samples to compare. All names must be included in both dicts.
    return_zero_if_nan_or_inf : bool, optional
        If True, return 0 if the KL divergence is NaN or inf. Default is False
    """
    # loop over the names and calculate the quantiled kld for each name
    klds = {}
    for name in names:
        klds[name] = quantiled_kl_divergence(
            sample_ref=np.array(dict_reference[name]),
            sample_approx=np.array(dict_approx[name]),
            n_bins=n_bins,
            return_zero_if_nan_or_inf=return_zero_if_nan_or_inf,
        )
    return klds


def calc_accuracy(preds, labels, verbose=False):
    """Calculates accuracy and AUC.

    Parameters
    ----------
    preds : array-like
        Classifier scores. Tensor of shape (n_samples, n_classes).
    labels : array-like
        Array with the true labels (one-hot encoded). Tensor of shape (n_samples, n_classes).

    Returns
    -------
    accuracy : float
        Accuracy.
    """
    accuracy = (np.argmax(preds, axis=1) == np.argmax(labels, axis=1)).mean()

    return accuracy


def calc_rejection(scores, labels, verbose=False, sig_eff=0.3):
    """Calculates the R30 metric.

    Parameters
    ----------
    scores : array-like
        Classifier scores (probability of being signal). Array of shape (n_samples,).
    labels : array-like
        Array with the true labels (0 or 1). Array of shape (n_samples,).
    sig_eff : float, optional
        Signal efficiency at which to calculate the rejection.

    Returns
    -------
    rejection : float
        Rejection metric value.
    cut_value : float
        Cut value for this rejection.
    """
    is_signal = labels == 1
    cut_value = np.percentile(scores[is_signal], 100 - sig_eff * 100)
    background_efficiency = np.sum(scores[~is_signal] > cut_value) / np.sum(~is_signal)
    if verbose:
        print(f"cut_value = {cut_value}")
        print(f"background_efficiency = {background_efficiency}")
    rejection = 1 / background_efficiency
    return rejection, cut_value


def calc_acc_from_logits(
    logits, targets, mask, token_indices_to_calc=[0, 1, 2, 3, 10, 20, 30, 40, 50, 100]
):
    """Calculate the accuracy for each token in the logits.

    Parameters
    ----------
    logits : torch.Tensor
        The logits from the model. Shape (batch_size, n_tokens, n_classes).
    targets : torch.Tensor
        The targets. Shape (batch_size, n_tokens).
    mask : torch.Tensor
        The mask. Shape (batch_size, n_tokens).
    token_indices_to_calc : list, optional
        The token indices to calculate the accuracy for.
        Default is [0, 1, 2, 3, 10, 20, 30, 40, 50, 100].

    Returns
    -------
    accs : dict
        Dictionary with the accuracy for each token position that we
        calculate the accuracy for.
    """
    if len(targets.size()) == 3:
        targets = targets.squeeze(-1)
    accs = {}
    for i in token_indices_to_calc:
        accs[f"acc_token_{i}"] = (
            (torch.argmax(logits[:, i, :], dim=-1) == targets[:, i]) * mask[:, i]
        ).sum() / mask[:, i].sum()
    accs["acc_all_tokens"] = ((torch.argmax(logits, dim=-1) == targets) * mask).sum() / mask.sum()
    stop_token_value = logits.shape[-1] - 1
    accs["frac_stop_token_included_argmax_logit"] = (
        torch.sum(torch.argmax(logits, dim=-1) == stop_token_value) / logits.shape[0]
    )
    accs["frac_stop_token_included_argmax_target"] = (
        torch.sum(targets == stop_token_value) / logits.shape[0]
    )

    return accs
