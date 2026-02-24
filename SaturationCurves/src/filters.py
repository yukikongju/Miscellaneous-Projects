from numpy import nanmax
import pandas as pd
from itertools import product
from curves import get_spend_metric_cutoff


def filter_segment(
    df: pd.DataFrame,
    network: str,
    platform: str,
    country: str,
    x_col: str = "spend",
    y_col: str = "paid",
    remove_outliers: bool = False,
) -> pd.DataFrame:
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
    df: pd.DataFrame, spend_col: str, metric_col: str
) -> pd.DataFrame:
    networks = list(df["network"].unique())
    platforms = list(df["platform"].unique())
    countries = list(df["country"].unique())

    rows = list(product(networks, platforms, countries))
    dff = pd.DataFrame(rows, columns=["network", "platform", "country"])

    def _get_spend_metric_cutoff(
        df, network: str, platform: str, country: str, percentile: float
    ) -> float:
        dff = filter_segment(
            df=df,
            network=network,
            platform=platform,
            country=country,
            x_col="spend",
            y_col=metric_col,
        )
        return get_spend_metric_cutoff(
            df=dff, spend_col=spend_col, metric_col=metric_col, p_asymptote=percentile
        )

    for percentile in [0.5, 0.75, 0.95]:
        col_name = f"p{int(round(percentile * 100))}"
        dff[col_name] = dff.apply(
            lambda row: _get_spend_metric_cutoff(
                df=df,
                network=row["network"],
                platform=row["platform"],
                country=row["country"],
                percentile=percentile,
            ),
            axis=1,
        )
    # filter out nan rows
    dff = dff.dropna()

    return dff
