import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", layout_file="layouts/gui.slides.json")


@app.cell
def _():
    import marimo as mo
    import sklearn
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
def _():
    from query import weekly_conversions_query

    weekly_conversions_query
    return (weekly_conversions_query,)


@app.cell
def _(bigquery, run_query, weekly_conversions_query):
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
        p0 = [y.max(), 0.001, np.median(x)]
        params, _ = curve_fit(logistic_curve, x, y, p0=p0, maxfev=20000)
        L, k, x0 = params

        # plot logistic
        x_grid = np.linspace(x.min(), x.max(), 100)
        y_pred = logistic_curve(x_grid, L, k, x0)

        marginal_returns = logistic_marginal_return(x, L, k, x0)

        # compute asymptote
        p_asymptote = 0.95
        flattened_asymptote = round(x0 + (1.0 / k) * np.log(p_asymptote / 0.05), 2)

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
            x=flattened_asymptote,
            color="r",
            linestyle="--",
            label=f"Asymptote {p_asymptote * 100}%: {flattened_asymptote}",
        )
        plt.title(f"{network} {country_code} {platform} - {y_col} vs {x_col}")
        plt.legend(loc="upper right")
        plt.show()

    return curve_fit, logistic_curve, np, scatterplot


@app.cell
def _(curve_fit, logistic_curve, np, pd, y_col):
    # --- TODO: compute marginal increase for clicks, installs, trials, paid, revenue by (network, platform, geo)
    # rows: network, platform, geo
    # columns (spend flatten cutoff): clicks, installs, trials, paid, revenue
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
    metrics = ["clicks", "installs", "trials", "paid"]  # TODO: add 'revenue'

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
        cutoff = x0 + (1.0 / k) * np.log(p_asymptote / (1.0 - p_asymptote))
        return float(cutoff)

    def get_spend_metric_cutoff(
        df: pd.DataFrame,
        spend_col: str,
        metric_col: str,
        country_code: str,
        network: str,
        platform: str,
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

        return _fit_cutoff_for_metric(df=dff, spend_col=spend_col, metric_col=metric_col)

    return countries, get_spend_metric_cutoff, metrics, networks, platforms


@app.cell
def _(
    countries,
    df,
    get_spend_metric_cutoff,
    metrics,
    networks,
    pd,
    platforms,
):
    from itertools import product

    rows = list(product(networks, platforms, countries))
    df_marginal = pd.DataFrame(rows, columns=["network", "platform", "country"])

    for metric in metrics:
        df_marginal[metric] = df_marginal.apply(
            lambda row: get_spend_metric_cutoff(
                df=df,
                spend_col="spend",
                metric_col=metric,
                country_code=row["country"],
                platform=row["platform"],
                network=row["network"],
            ),
            axis=1,
        )
    return (df_marginal,)


@app.cell
def _(df_marginal):
    df_marginal
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
    def plot_spend_marginal_timeline(
        df: pd.DataFrame, df_marginal: pd.DataFrame, country: str, network: str, platform: str
    ):
        # filter
        network_mask = df["network"] == network
        country_mask = df["country"] == country
        platform_mask = df["platform"] == platform
        dff = df[network_mask & country_mask & platform_mask].dropna().copy()

        # year format
        dff["date"] = dff.apply(
            lambda row: get_date_from_year_isoweek(row.year, row.isoweek), axis=1
        )
        dff = dff.sort_values(by=["date"])

        # plot
        segment_mask = (
            (df_marginal["country"] == country)
            & (df_marginal["network"] == network)
            & (df_marginal["platform"] == platform)
        )
        # plt.axhline(df_marginal.loc[segment_mask, 'clicks'].values, label='clicks', color='m', linestyle='--')
        plt.axhline(
            df_marginal.loc[segment_mask, "installs"].values,
            label="installs",
            color="r",
            linestyle="--",
        )
        plt.axhline(
            df_marginal.loc[segment_mask, "trials"].values,
            label="trials",
            color="g",
            linestyle="--",
        )
        plt.axhline(
            df_marginal.loc[segment_mask, "paid"].values, label="paid", color="c", linestyle="--"
        )
        # plt.axhline(df_marginal.loc[segment_mask, 'revenue'].values, label='revenue', color='y', linestyle='--')
        plt.plot(dff["date"], dff["spend"])
        plt.title(f"{network} {platform} {country} - Spend over time")
        plt.legend(loc="upper right")
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


@app.cell
def _(
    country_dropdown,
    df,
    df_marginal,
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
    )
    plot_spend_marginal_timeline(
        df=df,
        df_marginal=df_marginal,
        country=country_dropdown.value,
        network=network_dropdown.value,
        platform=platform_dropdown.value,
    )

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
