# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "google-cloud>=0.34.0",
#     "marimo>=0.19.10",
#     "matplotlib>=3.10.8",
#     "numpy>=2.4.2",
#     "pandas>=3.0.1",
#     "protobuf>=6.33.5",
#     "pyzmq>=27.1.0",
#     "scikit-learn>=1.8.0",
#     "scipy>=1.17.0",
# ]
# ///

import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", layout_file="layouts/gui.slides.json")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt

    return mo, plt


@app.cell
def _():
    from google.cloud import bigquery
    import pandas as pd

    def run_query(client: bigquery.Client, query: str) -> pd.DataFrame:
        df = client.query(query).to_dataframe()
        return df

    return bigquery, pd, run_query


@app.cell
def _(bigquery, run_query):
    from query import weekly_conversions_query

    client = bigquery.Client()
    df = run_query(client=client, query=weekly_conversions_query)
    return (df,)


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(pd, plt):
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    from scipy.optimize import curve_fit

    def logistic_curve(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def logistic_marginal_return(x, L, k, x0):
        exp_term = np.exp(-k * (x - x0))
        return (L * k * exp_term) / (1 + exp_term) ** 2

    def scatterplot(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        country_code: str,
        network: str,
        platform: str,
        remove_outliers: bool = False,
        min_perc_asymptote: float = 0.00325,
        max_perc_asymptote: float = 0.95,
    ):
        # filter
        country_mask = df["country"] == country_code
        platform_mask = df["platform"] == platform
        network_mask = df["network"] == network
        spend_mask = df["spend"] > 0
        dff = (
            df.loc[country_mask & spend_mask & platform_mask & network_mask, [x_col, y_col]]
            .dropna()
            .copy()
        )

        # remove outliers IQR
        if remove_outliers:
            q1, q3 = dff[y_col].quantile(0.25), dff[y_col].quantile(0.75)
            IQR = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * IQR, q3 + 1.5 * IQR
            outlier_mask = (dff[y_col] < lower_bound) | (dff[y_col] > upper_bound)
            dff = dff[~outlier_mask]

        # estimate logistic
        x, y = dff[x_col].to_numpy(dtype=float), dff[y_col].to_numpy(dtype=float)
        p0 = [y.max(), 0.0035, np.median(x)]
        params, _ = curve_fit(logistic_curve, x, y, p0=p0, maxfev=20000)
        L, k, x0 = params

        # plot logistic
        x_grid = np.linspace(x.min(), x.max(), 100)
        y_pred = logistic_curve(x_grid, L, k, x0)

        marginal_returns = logistic_marginal_return(x, L, k, x0)

        # compute asymptote
        max_spend_asymptote = round(x0 + (1.0 / k) * np.log(max_perc_asymptote / 0.05), 2)
        min_spend_asymptote = round(x0 + (1.0 / k) * np.log(min_perc_asymptote / 0.05), 2)
        # max_spend_asymptote = round(x0 + (1.0 / k) * np.log(max_perc_asymptote / (1.0 - max_perc_asymptote)), 2)
        # min_spend_asymptote = round(x0 + (1.0 / k) * np.log(min_perc_asymptote / (1.0 - min_perc_asymptote)), 2)

        # TODO: compute marginal
        mCPI = 1 / marginal_returns
        current_marginal_return = logistic_marginal_return(np.mean(x), L, k, x0)
        incremental_per_1K = 1000 * current_marginal_return
        # MR(x) - 1/target_cpi = 0
        # TODO: compute target cpi: maximum cost per install we are willing to pay
        # target cpi = allowed cac per paying user * paid
        # allowed cac per paying user = ltv * margin_guardrail
        # margin guarrail =

        # print(mCPI)
        # print(current_marginal_return)
        print(f"Incremental {y_col} per 1K {x_col}: {incremental_per_1K}")
        # print(flattened_asymptote)

        # scatterplot
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.scatter(dff[x_col], dff[y_col], color="blue", alpha=0.2)
        # plt.plot([], [], ' ', label=f"Incremental {y_col} per 1K {x_col}: {incremental_per_1K}")
        plt.plot(x_grid, y_pred, color="green")
        plt.axvline(
            x=min_spend_asymptote,
            color="m",
            linestyle="--",
            label=f"Asymptote {min_perc_asymptote * 100}%: {min_spend_asymptote}",
        )
        plt.axvline(
            x=max_spend_asymptote,
            color="r",
            linestyle="--",
            label=f"Asymptote {max_perc_asymptote * 100}%: {max_spend_asymptote}",
        )
        plt.title(f"{network} {country_code} {platform} - {y_col} vs {x_col}")
        plt.legend(loc="upper right")
        plt.show()

    return curve_fit, logistic_curve, logistic_marginal_return, np, scatterplot


@app.cell
def _(curve_fit, logistic_curve, np, pd, y_col):
    # --- TODO: compute marginal increase for clicks, installs, trials, paid, revenue by (network, platform, geo)
    # rows: network, platform, geo
    # columns (spend flatten cutoff): clicks, installs, trials, paid, revenue

    def _fit_cutoff_for_metric(
        df: pd.DataFrame,
        spend_col: str,
        metric_col: str,
        p_asymptote: float = 0.95,
        min_points: int = 12,
    ):
        s = df[[spend_col, metric_col]].dropna()
        s = s[(s[spend_col] > 0)]  # & (s[metric_col] >= 0)

        if len(s) < min_points or s[spend_col].nunique() < 5:
            return np.nan

        x = s[spend_col].to_numpy(dtype=float)
        y = s[metric_col].to_numpy(dtype=float)

        # Reasonable init + bounds for stability
        p0 = [max(y.max(), 1.0), 0.001, np.median(x)]
        bounds = ([0.0, 1e-9, x.min() * 0.5], [np.inf, np.inf, x.max() * 1.5])

        try:
            params, _ = curve_fit(logistic_curve, x, y, p0=p0, bounds=bounds, maxfev=40000)
            L, k, x0 = params
            if k <= 0:
                return np.nan
        except:
            raise Exception()

        # Spend cutoff at chosen asymptote level (e.g., 95%)
        # cutoff = x0 + (1.0 / k) * np.log(p_asymptote / (1.0 - p_asymptote))
        cutoff = round(x0 + (1.0 / k) * np.log(p_asymptote / 0.05), 2)

        return float(cutoff)

    def get_spend_metric_cutoff(
        df: pd.DataFrame,
        spend_col: str,
        metric_col: str,
        country_code: str,
        network: str,
        platform: str,
        p_asymptote: float,
        remove_outliers: bool = False,
    ):
        country_mask = df["country"] == country_code
        platform_mask = df["platform"] == platform
        network_mask = df["network"] == network
        spend_mask = df[spend_col] > 0
        dff = (
            df.loc[
                country_mask & spend_mask & platform_mask & network_mask, [spend_col, metric_col]
            ]
            .dropna()
            .copy()
        )

        # remove outliers IQR
        if remove_outliers:
            q1, q3 = dff[y_col].quantile(0.25), dff[y_col].quantile(0.75)
            IQR = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * IQR, q3 + 1.5 * IQR
            outlier_mask = (dff[y_col] < lower_bound) | (dff[y_col] > upper_bound)
            dff = dff[~outlier_mask]

        return _fit_cutoff_for_metric(
            df=dff, spend_col=spend_col, metric_col=metric_col, p_asymptote=p_asymptote
        )

    return (get_spend_metric_cutoff,)


@app.cell(hide_code=True)
def _(mo):
    _df = mo.sql(
        f"""
        -- select distinct network, platform, country from df
        """
    )
    return


@app.cell
def _(df, get_spend_metric_cutoff, pd):
    from itertools import product

    networks = [
        "Facebook Ads",
        "Apple Search Ads",
        "googleadwords_int",
        "tiktokglobal_int",
        "tatari_streaming",
        "tatari_linear",
        "snapchat_int",
    ]  # TODO: snapchat_int, tatari
    platforms = ["ios", "android"]  # no web because not enough data
    countries = [
        "US",
    ]  # TODO: add 'ROW'
    metrics = ["clicks", "installs", "trials", "paid", "revenue"]

    rows = list(product(networks, platforms, countries))
    df_max_spend = pd.DataFrame(rows, columns=["network", "platform", "country"])
    df_min_spend = pd.DataFrame(rows, columns=["network", "platform", "country"])

    MAX_PERC_ASYMPTOTE = 0.95
    MIN_PERC_ASYMPTOTE = 0.00325

    for metric in metrics:
        df_max_spend[metric] = df_max_spend.apply(
            lambda row: get_spend_metric_cutoff(
                df=df,
                spend_col="spend",
                metric_col=metric,
                country_code=row["country"],
                platform=row["platform"],
                network=row["network"],
                p_asymptote=MAX_PERC_ASYMPTOTE,
            ),
            axis=1,
        )
        # FIXME:
        df_min_spend[metric] = df_min_spend.apply(
            lambda row: get_spend_metric_cutoff(
                df=df,
                spend_col="spend",
                metric_col=metric,
                country_code=row["country"],
                platform=row["platform"],
                network=row["network"],
                p_asymptote=MIN_PERC_ASYMPTOTE,
            ),
            axis=1,
        )
    return (
        MAX_PERC_ASYMPTOTE,
        MIN_PERC_ASYMPTOTE,
        countries,
        df_max_spend,
        df_min_spend,
        metrics,
        networks,
        platforms,
        product,
    )


@app.cell
def _():
    return


@app.cell
def _():
    from datetime import datetime

    def get_date_from_year_isoweek(y, w):
        return datetime.strptime(f"{y} {w} 1", "%G %V %u")

    return (get_date_from_year_isoweek,)


@app.cell
def _(get_date_from_year_isoweek, pd, plt):
    # --- plot timeline vs spend and add
    from matplotlib.lines import Line2D

    def plot_spend_marginal_timeline(
        df: pd.DataFrame,
        df_max_spend: pd.DataFrame,
        df_min_spend: pd.DataFrame,
        country: str,
        network: str,
        platform: str,
    ):
        # filter
        network_mask = df["network"] == network
        country_mask = df["country"] == country
        platform_mask = df["platform"] == platform
        spend_mask = df["spend"] > 0
        dff = df[network_mask & country_mask & platform_mask & spend_mask].dropna().copy()

        # year format
        dff["date"] = dff.apply(
            lambda row: get_date_from_year_isoweek(row.year, row.isoweek), axis=1
        )
        dff = dff.sort_values(by=["date"])

        # plot
        segment_mask = (
            (df_max_spend["country"] == country)
            & (df_max_spend["network"] == network)
            & (df_max_spend["platform"] == platform)
        )

        # draw min and max hlines
        metric_colors = {"installs": "r", "trials": "g", "paid": "c", "revenue": "y"}
        for metric, color in metric_colors.items():
            max_val = df_max_spend.loc[segment_mask, metric].squeeze()
            min_val = df_min_spend.loc[segment_mask, metric].squeeze()

            plt.axhline(max_val, color=color, linestyle="--", alpha=0.9, label="_nolegend_")
            plt.axhline(min_val, color=color, linestyle=":", alpha=0.9, label="_nolegend_")

        # legend shows only metric colors
        metric_legend = [
            Line2D([0], [0], color=color, lw=2, label=metric)
            for metric, color in metric_colors.items()
        ]
        plt.legend(handles=metric_legend, loc="upper right")

        plt.plot(dff["date"], dff["spend"])
        plt.title(f"{network} {platform} {country} - Spend over time")
        plt.show()

    return (plot_spend_marginal_timeline,)


@app.cell
def _(countries, metrics, mo, networks, platforms):
    network_dropdown = mo.ui.dropdown(options=networks, value=networks[0])
    platform_dropdown = mo.ui.dropdown(options=platforms, value=platforms[0])
    country_dropdown = mo.ui.dropdown(options=countries, value=countries[0])
    metrics_dropdown = mo.ui.dropdown(options=metrics, value=metrics[0])
    outlier_dropdown = mo.ui.dropdown(options=[True, False], value=False, label="remove outlier")
    side_by_side = mo.hstack(
        [network_dropdown, platform_dropdown, country_dropdown, metrics_dropdown, outlier_dropdown],
        justify="start",
        gap=1,
    )
    return (
        country_dropdown,
        metrics_dropdown,
        network_dropdown,
        outlier_dropdown,
        platform_dropdown,
        side_by_side,
    )


@app.cell
def _(side_by_side):
    side_by_side
    return


@app.cell(hide_code=True)
def _(
    MAX_PERC_ASYMPTOTE,
    MIN_PERC_ASYMPTOTE,
    country_dropdown,
    df,
    df_max_spend,
    df_min_spend,
    metrics_dropdown,
    network_dropdown,
    outlier_dropdown,
    platform_dropdown,
    plot_spend_marginal_timeline,
    scatterplot,
):
    scatterplot(
        df=df,
        x_col="spend",
        y_col=metrics_dropdown.value,
        country_code=country_dropdown.value,
        platform=platform_dropdown.value,
        network=network_dropdown.value,
        remove_outliers=outlier_dropdown.value,
        min_perc_asymptote=MIN_PERC_ASYMPTOTE,
        max_perc_asymptote=MAX_PERC_ASYMPTOTE,
    )
    plot_spend_marginal_timeline(
        df=df,
        df_max_spend=df_max_spend,
        df_min_spend=df_min_spend,
        country=country_dropdown.value,
        network=network_dropdown.value,
        platform=platform_dropdown.value,
    )
    return


@app.cell
def _():
    # --- TODO: greedy budget allocator
    return


@app.cell
def _(curve_fit, logistic_curve, logistic_marginal_return, np):
    import math
    from pydantic import BaseModel
    from typing import List

    class MarginalReturn(BaseModel):
        increment: int
        marginal_return: float

    def get_logistic_marginal_returns(
        x, y, max_spend: float = 250000, jump: int = 100
    ) -> List[MarginalReturn]:
        p0 = [y.max(), 0.001, np.median(x)]
        params, _ = curve_fit(logistic_curve, x, y, p0=p0, maxfev=20000)
        L, k, x0 = params
        increments = range(0, math.ceil(x.max() / 1000) * 1000, jump)
        marginal_returns = logistic_marginal_return(increments, L, k, x0)

        # dataframe
        # data = {
        #     "increments": increments,
        #     "marginal_return": marginal_returns
        # }
        # df = pd.DataFrame(data)

        # Marginal Return
        data = [
            MarginalReturn(**{"increment": increment, "marginal_return": mreturn})
            for increment, mreturn in zip(increments, marginal_returns)
        ]

        return data

    return List, MarginalReturn, get_logistic_marginal_returns


@app.cell
def _(pd):
    def filter_segment(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        country_name: str,
        network_name: str,
        platform_name: str,
        remove_outliers: bool = False,
    ):
        # filter
        country_mask = df["country"] == country_name
        platform_mask = df["platform"] == platform_name
        network_mask = df["network"] == network_name
        spend_mask = df["spend"] > 0
        dff = (
            df.loc[country_mask & spend_mask & platform_mask & network_mask, [x_col, y_col]]
            .dropna()
            .copy()
        )

        # remove outliers IQR
        if remove_outliers:
            q1, q3 = dff[y_col].quantile(0.25), dff[y_col].quantile(0.75)
            IQR = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * IQR, q3 + 1.5 * IQR
            outlier_mask = (dff[y_col] < lower_bound) | (dff[y_col] > upper_bound)
            dff = dff[~outlier_mask]
        return dff

    return (filter_segment,)


@app.cell
def _():
    # dct_marginal_returns[('Facebook Ads', 'ios', 'US')]
    # dct_marginal_returns[('Apple Search Ads', 'android', 'US')]
    return


@app.cell
def _(
    List,
    countries,
    df,
    filter_segment,
    get_logistic_marginal_returns,
    networks,
    pd,
    platforms,
    product,
):
    from collections import defaultdict

    metric_col = "revenue"
    spend_col = "spend"
    MARGINAL_STEP = 100
    MAX_SPEND = 250000

    def get_marginal_returns_dct(
        df: pd.DataFrame,
        networks: List[str],
        platforms: List[str],
        countries: List[str],
        marginal_step: int,
        max_spend: float,
    ):
        dct = defaultdict()
        for network, platform, country in list(product(networks, platforms, countries)):
            dff = filter_segment(
                df=df,
                country_name=country,
                platform_name=platform,
                network_name=network,
                x_col=spend_col,
                y_col=metric_col,
            )
            if dff.empty:
                continue
            dct[(network, platform, country)] = get_logistic_marginal_returns(
                x=dff[spend_col].to_numpy(dtype=float),
                y=dff[metric_col].to_numpy(dtype=float),
                jump=marginal_step,
                max_spend=max_spend,
            )
        return dct

    dct_marginal_returns = get_marginal_returns_dct(
        df=df,
        networks=networks,
        platforms=platforms,
        countries=countries,
        marginal_step=MARGINAL_STEP,
        max_spend=MAX_SPEND,
    )
    return MARGINAL_STEP, dct_marginal_returns, defaultdict, metric_col


@app.cell
def _(
    MARGINAL_STEP,
    countries,
    df_min_spend,
    metric_col,
    networks,
    np,
    pd,
    platforms,
    product,
):
    df_budget = pd.DataFrame(
        list(product(networks, platforms, countries)), columns=["network", "platform", "country"]
    )
    df_budget["budget"] = 0

    BUDGET = 150_000  # note: need to match aggregation (currrent: weekly)
    MAX_INIT_SPEND = 10_000

    key_cols = ["network", "platform", "country"]
    # initialization - set minimum for all segments and floor to marginal step
    # NOTE: very bad when one network minimal spend is too high or weekly spend is too low; ASA is exponential :o
    df_budget = (
        pd.merge(df_budget, df_min_spend, on=key_cols, how="left")[key_cols + [metric_col]]
        .fillna(0)
        .rename(columns={metric_col: "budget"})
    )
    df_budget["budget"] = np.ceil(df_budget["budget"] / MARGINAL_STEP) * MARGINAL_STEP

    # fallback
    negative_mask = df_budget["budget"] < 0
    max_init_spend_mask = df_budget["budget"] > MAX_INIT_SPEND
    df_budget.loc[negative_mask, "budget"] = 0.0
    df_budget.loc[max_init_spend_mask, "budget"] = MAX_INIT_SPEND

    return BUDGET, df_budget


@app.cell
def _(
    BUDGET,
    List,
    MARGINAL_STEP,
    MarginalReturn,
    dct_marginal_returns,
    defaultdict,
    df_budget,
    pd,
):
    # initialize marginal return from budget init
    import bisect
    import heapq
    from typing import Dict, Tuple

    def _init_marginal_budget_returns(
        df_budget: pd.DataFrame,
        dct_marginal_returns: Dict[Tuple[str, str, str], List[MarginalReturn]],
    ):
        margins = dct_marginal_returns.copy()
        for k in margins.keys():
            network, platform, country = k
            network_mask = df_budget["network"] == network
            platform_mask = df_budget["platform"] == platform
            country_mask = df_budget["country"] == country
            threshold = df_budget.loc[network_mask & platform_mask & country_mask, "budget"].item()
            idx = bisect.bisect_left([m.increment for m in margins[k]], threshold)
            margins[k] = margins[k][idx:]
        return margins

    def get_budget_allocation(budget: float, jump: int):
        margins = _init_marginal_budget_returns(df_budget, dct_marginal_returns)

        # print(margins)

        dct_budget_allocations = defaultdict(float)
        for k in margins.keys():
            dct_budget_allocations[k] = margins[k][0].increment

        # init max heap
        heap = [(-margins[k].pop(0).marginal_return, k) for k in margins.keys()]
        # heap = [(-m.marginal_return, k) for k in margins.keys() for m in margins[k]]
        heapq.heapify(heap)
        # print(heap)

        # greedy allocation -
        budget -= df_budget["budget"].sum()
        expected_total = 0
        while heap and budget > MARGINAL_STEP:
            v, k = heapq.heappop(heap)
            # expected_total += -v * jump
            expected_total += -v
            budget -= jump
            dct_budget_allocations[k] += jump

            if margins[k]:
                next_margin = margins[k].pop(0)
                heapq.heappush(heap, (-next_margin.marginal_return, k))

        df_allocations = pd.DataFrame(
            [
                {"network": k[0], "platform": k[1], "country": k[2], "allocated_budget": v}
                for k, v in dct_budget_allocations.items()
            ]
        )

        print(df_allocations)

        # print(dct_budget_allocations)
        print(expected_total)

    get_budget_allocation(budget=BUDGET, jump=MARGINAL_STEP)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(df_max_spend):
    df_max_spend
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
