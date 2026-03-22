import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    from google.cloud import bigquery

    def run_query(client: bigquery.Client, query: str) -> pd.DataFrame:
        df = client.query(query).to_dataframe()
        return df

    return bigquery, run_query


@app.cell
def _(bigquery, run_query):
    from src.queries import organics_monthly_query

    client = bigquery.Client()
    df = run_query(client=client, query=organics_monthly_query)
    return (df,)


@app.cell
def _(pd):
    df_influencer = pd.read_csv("influencer_spend.csv")
    df_influencer.head()
    return (df_influencer,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df, df_influencer, pd):
    key_cols = ["year", "month"]
    metric_cols = ["installs", "trials", "paid", "revenue"]
    df_organics = pd.merge(
        df_influencer[key_cols + ["spend"]],
        df[key_cols + metric_cols],
        on=["year", "month"],
        how="left",
    )
    return (df_organics,)


@app.cell
def _(df_organics):
    df_organics
    return


@app.cell
def _():
    # remove organics conversions normally => we didn't spend anything on April 2025, so it should be our organics baseline
    # mask = (df_organics['year'] == 2025) & (df_organics['month'] == 4)
    # april_values = df_organics.loc[mask, metric_cols].iloc[0]

    # df_organics_cleaned = df_organics.copy()
    # df_organics_cleaned[metric_cols] = df_organics_cleaned[metric_cols] - april_values
    # df_organics_cleaned
    return


@app.cell
def _():
    import numpy as np
    from scipy.optimize import curve_fit

    def logistic_curve(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def linear_func(x, m, b):
        return m * x + b

    def logistic_marginal_return(x, L, k, x0):
        exp_term = np.exp(-k * (x - x0))
        return (L * k * exp_term) / (1 + exp_term) ** 2

    return curve_fit, linear_func, np


@app.cell
def _(curve_fit, df_organics, linear_func, np):
    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    dff = df_organics

    fig = go.Figure()

    x_col = "spend"
    y_col = "paid"

    # x_col = 'paid'
    # y_col = 'spend'

    x, y = dff[x_col].to_numpy(dtype=float), dff[y_col].to_numpy(dtype=float)

    # y = y - y.min()

    x_grid = np.linspace(0, x.max(), 1000)

    # logistic curve
    # p0 = [y.max(), 0.05, np.median(x)]
    # params, _ = curve_fit(logistic_curve, x, y, p0=p0, maxfev=200000)
    # L, k, x0 = params
    # y_pred = logistic_curve(x_grid, L, k, x0)

    # linear regression
    params, _ = curve_fit(linear_func, x, y)
    m, b = params
    y_pred = linear_func(x_grid, m, b)

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=dff[x_col],
            y=dff[y_col],
            mode="markers",
            marker=dict(color="blue", opacity=0.2),
            name="Data",
        )
    )

    # Fitted curve
    fig.add_trace(
        go.Scatter(x=x_grid, y=y_pred, mode="lines", line=dict(color="green"), name="Fit")
    )

    fig.update_layout(
        # title=f"{network} {country_code} {platform} - {y_col} vs {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend=dict(x=1, xanchor="right"),
        template="plotly_white",
    )

    fig.show()

    return b, dff, m


@app.cell
def _(b):
    print(f"Intercept: {b.item()}")
    return


@app.cell
def _():
    #
    # 1 = b + m × Spend
    # Spend = (1 - b) / m
    return


@app.cell
def _(b, m):
    min_conversions = 100
    print(f"Min Spend: {abs((min_conversions - b) / m)}")
    return


@app.cell
def _(df_organics, dff, np, pd):
    # using CPA: CPA = Total Spend / Total Conversions
    def get_cpa(df: pd.DataFrame, metric_col: str):
        dff["cpa"] = df["spend"] / (df[metric_col] - df[metric_col].min())
        mask = (dff["cpa"] >= 1.0) & (np.isfinite(dff["cpa"]))  # & (~np.isnan(dff))
        # print(list(dff['cpa']))
        return dff.loc[mask, "cpa"].mean().item()

    print(f"Paid CPA: {get_cpa(df=df_organics, metric_col='paid')}")
    print(f"Trial CPA: {get_cpa(df=df_organics, metric_col='trials')}")
    return


@app.cell
def _(df_organics):

    # using conversion rate: Conversion Rate = Conversions / Spend
    (df_organics["paid"] - df_organics["paid"].min()) / df_organics["spend"]
    return


if __name__ == "__main__":
    app.run()
