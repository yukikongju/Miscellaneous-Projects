"""Utilities for fitting and evaluating logistic saturation curves."""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def logistic_curve(x, L, k, x0):
    """Compute a logistic curve value.

    Args:
        x: Spend values.
        L: Curve ceiling (maximum attainable outcome).
        k: Growth steepness.
        x0: Midpoint spend where growth starts to slow.

    Returns:
        Logistic outcome values.
    """
    return L / (1 + np.exp(-k * (x - x0)))


def logistic_marginal_return(x, L, k, x0):
    """Compute the derivative of the logistic curve.

    Args:
        x: Spend values.
        L: Curve ceiling (maximum attainable outcome).
        k: Growth steepness.
        x0: Midpoint spend.

    Returns:
        Marginal outcome values.
    """
    exp_term = np.exp(-k * (x - x0))
    return (L * k * exp_term) / (1 + exp_term) ** 2


def get_logistic_params(
    x: np.ndarray, y: np.ndarray, func=logistic_curve
) -> Tuple[float, float, float]:
    """Fit a logistic-like function and return estimated parameters.
    Args:
        x: Input spend values.
        y: Observed outcome values.
        func: Curve function passed to `scipy.optimize.curve_fit`.

    Returns:
        Fitted parameters as `(L, k, x0)`.
    """
    # if x.size() == 0 or y.size() == 0:
    # if len(x) == 0 or len(y) == 0:
    # return np.nan, np.nan, np.nan

    p0 = [
        y.max(),
        0.0035,  # FIXME
        np.median(x),
    ]  # note: [L, k, x0] are the initial guess (ceiling, slope, midpoint)
    # FIXME: add bounds
    bounds = ([0.0, 1e-9, x.min() * 0.5], [np.inf, np.inf, x.max() * 1.5])
    # params, _ = curve_fit(func, x, y, p0=p0, maxfev=20000)  # 40000
    params, _ = curve_fit(func, x, y, p0=p0, bounds=bounds, maxfev=20000)  # 40000
    L, k, x0 = params
    return (L, k, x0)


def get_logistic_asymptote_bound(x0, k, perc: float) -> float:
    """Convert a target logistic percentile into the corresponding spend value.

    Args:
        x0: Logistic midpoint parameter.
        k: Logistic steepness parameter.
        perc: Target percentile on the asymptote, between 0 and 1.

    Returns:
        Spend value at the requested percentile.
    """
    return round(x0 + (1.0 / k) * np.log(perc / (1.0 - perc)), 2)
