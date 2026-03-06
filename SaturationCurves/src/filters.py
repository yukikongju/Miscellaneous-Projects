"""Utilities for segment filtering and marginal cutoff tabulation."""

from collections import defaultdict
from itertools import product
from typing import List

import pandas as pd
import numpy as np
from streamlit import metric

# from curves import get_spend_metric_cutoff
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


# def get_ratio_percentile(
# df: pd.DataFrame, x_col: str, y_col: str, percentiles: List[float]
# ):
# """Compute ratio percentiles for a spend-to-metric relationship.

# Args:
# df: Input DataFrame.
# x_col: Numerator column for ratio construction.
# y_col: Denominator column for ratio construction.
# percentiles: Percentiles to compute.

# Returns:
# Placeholder return until implemented.
# """
# pass


def get_marginal_spend_metric_table(
    df: pd.DataFrame,
    spend_col: str,
    metric_col: str,
    percentiles: List[float] = [0.5, 0.75, 0.95],
) -> pd.DataFrame:
    """Build a table of spend cutoffs at multiple asymptote percentiles.

    Args:
        df: Input weekly dataset with segment identifiers.
        spend_col: Spend column name used for curve fitting.
        metric_col: Metric column name used for curve fitting.

    Returns:
        DataFrame containing segment keys and cutoff columns (`p50`, `p75`,
        `p95`), excluding rows with missing cutoffs.
    """
    # networks = list(df["network"].unique())
    # platforms = list(df["platform"].unique())
    # countries = list(df["country"].unique())
    rows = df[["network", "platform", "country"]].drop_duplicates()

    # rows = list(product(networks, platforms, countries))
    # dff = pd.DataFrame(rows, columns=["network", "platform", "country"])

    # def _get_spend_metric_cutoff(
    # df, network: str, platform: str, country: str, percentile: float
    # ) -> float:
    # dff = filter_segment(
    # df=df,
    # network=network,
    # platform=platform,
    # country=country,
    # x_col="spend",
    # y_col=metric_col,
    # )
    # return get_spend_metric_cutoff(
    # df=dff, spend_col=spend_col, metric_col=metric_col, p_asymptote=percentile
    # )

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
    data = []
    for _, row in rows.iterrows():
        network, platform, country = row["network"], row["platform"], row["country"]
        values = [network, platform, country]
        if (network, platform, country) not in dct_logistic_param:
            values += len(percentiles) * [np.nan]
            continue

        _, k, x0 = dct_logistic_param[(network, platform, country)]
        for percentile in percentiles:
            spend = get_logistic_asymptote_bound(x0, k, percentile)
            values.append(spend)
        data.append(values)
    percentile_col_names = [f"p{int(round(p * 100))}" for p in percentiles]
    df_spend_cutoff = pd.DataFrame(
        data, columns=["network", "platform", "country"] + percentile_col_names
    )
    df_spend_cutoff = df_spend_cutoff.dropna()

    # TODO: compute ratio

    # for percentile in [0.5, 0.75, 0.95]:
    # col_name = f"p{int(round(percentile * 100))}"
    # dff[col_name] = dff.apply(
    # lambda row: _get_spend_metric_cutoff(
    # df=df,
    # network=row["network"],
    # platform=row["platform"],
    # country=row["country"],
    # percentile=percentile,
    # ),
    # axis=1,
    # )

    # TODO: compute ratio (cac/roas)
    # 1. compute logistic regression params for each (network-platform) pair
    # 2. compute spend cutoff for

    # filter out nan rows
    # dff = dff.dropna()

    return df_spend_cutoff
