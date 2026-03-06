"""Utilities for segment filtering and marginal cutoff tabulation."""

from collections import defaultdict
from itertools import product
from typing import List, Tuple

import pandas as pd
import numpy as np
from streamlit import metric

from curves import get_logistic_params, get_logistic_asymptote_bound, logistic_curve


def filter_segment(
    df: pd.DataFrame,
    network: str,
    platform: str,
    country: str,
    x_col: str = "spend",
    y_col: str = "paid",
    remove_outliers: bool = False,
) -> pd.DataFrame:
    """Filter a DataFrame to one segment and optionally remove outcome outliers.

    Args:
        df: Source data containing segmentation and metric columns.
        network: Network selector.
        platform: Platform selector.
        country: Country selector.
        x_col: Spend column name.
        y_col: Metric column name.
        remove_outliers: Whether to remove `y_col` outliers by IQR bounds.

    Returns:
        Filtered DataFrame with only `x_col` and `y_col`.
    """
    mask = (
        (df["network"] == network)
        & (df["platform"] == platform)
        & (df["country"] == country)
        & (df[x_col] > 0)
        & (df[y_col] > 0)
    )
    dff = df.loc[mask, [x_col, y_col]].dropna().copy()

    if remove_outliers and not dff.empty:
        q1 = dff[y_col].quantile(0.25)
        q3 = dff[y_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        dff = dff[(dff[y_col] >= lower) & (dff[y_col] <= upper)]

    return dff


def get_marginal_spend_metric_table(
    df: pd.DataFrame,
    spend_col: str,
    metric_col: str,
    percentiles: List[float] = [0.5, 0.75, 0.95],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build a table of spend cutoffs at multiple asymptote percentiles.

    Args:
        df: Input weekly dataset with segment identifiers.
        spend_col: Spend column name used for curve fitting.
        metric_col: Metric column name used for curve fitting.

    Returns:
        DataFrames containing segment keys and percentile columns (`p50`, `p75`,
        `p95`), excluding rows with missing cutoffs for percentile spend
        and ratio (cac/roas)
    """
    rows = df[["network", "platform", "country"]].drop_duplicates()

    # TODO: compute logistic regression param for each (network-platform-country) pair
    dct_logistic_param = defaultdict()
    for _, row in rows.iterrows():
        network, platform, country = (
            str(row["network"]),
            str(row["platform"]),
            str(row["country"]),
        )
        dff = filter_segment(
            df=df,
            network=network,
            platform=platform,
            country=country,
            x_col=spend_col,
            y_col=metric_col,
            remove_outliers=True,
        )
        if dff.empty:
            continue

        x = dff[spend_col].to_numpy(dtype=float)
        y = dff[metric_col].to_numpy(dtype=float)

        min_points = 12
        if len(x) < min_points:
            continue

        dct_logistic_param[(network, platform, country)] = get_logistic_params(
            x=x, y=y, func=logistic_curve
        )

    # --- compute marginal spend
    marginal_data, predictions_data = [], []
    for _, row in rows.iterrows():
        network, platform, country = row["network"], row["platform"], row["country"]
        values = [network, platform, country]
        predictions = [network, platform, country]
        if (network, platform, country) not in dct_logistic_param:
            values += len(percentiles) * [np.nan]
            continue

        L, k, x0 = dct_logistic_param[(network, platform, country)]
        for percentile in percentiles:
            spend = get_logistic_asymptote_bound(x0, k, percentile)
            pred = logistic_curve(x=spend, L=L, k=k, x0=x0)
            values.append(spend)
            predictions.append(pred)
        marginal_data.append(values)
        predictions_data.append(predictions)
    percentile_col_names = [f"p{int(round(p * 100))}" for p in percentiles]

    # Dataset for spend cutoff
    df_spend_cutoff = pd.DataFrame(
        marginal_data, columns=["network", "platform", "country"] + percentile_col_names
    )
    df_spend_cutoff = df_spend_cutoff.dropna()

    # FIXME: Dataset for percentile ratio
    df_predictions = pd.DataFrame(
        predictions_data,
        columns=["network", "platform", "country"] + percentile_col_names,
    )
    df_predictions = df_predictions.dropna()
    if metric_col == "revenue":
        df_ratio = df_predictions.copy()
        for c in percentile_col_names:
            df_ratio[c] = df_ratio[c] / df_spend_cutoff[c]
    else:
        df_ratio = df_spend_cutoff.copy()
        for c in percentile_col_names:
            df_ratio[c] = df_ratio[c] / df_predictions[c]

    return df_spend_cutoff, df_ratio
