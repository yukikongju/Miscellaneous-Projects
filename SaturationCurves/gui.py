import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium", layout_file="layouts/gui.slides.json")


@app.cell
def _():
    import marimo as mo
    import sklearn
    import matplotlib.pyplot as plt

    return (plt,)


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
def _(df, pd, plt):
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    from scipy.optimize import curve_fit

    def logistic_curve(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    def logistic_marginal_return(x, L, k, x0):
        exp_term = np.exp(-k * (x - x0))
        return (L * k * exp_term) / (1 + exp_term) ** 2

    def scatterplot(df: pd.DataFrame, x_col: str, y_col: str, country_code: str):
        # filter
        country_mask = df["country"] == country_code
        dff = df.loc[country_mask, [x_col, y_col]].dropna().copy()

        # remove outliers IQR
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

        marginal_returns_grid = logistic_marginal_return(x, L, k, x0)

        # TODO: compute marginal

        # scatterplot
        plt.scatter(dff[x_col], dff[y_col], color="blue", alpha=0.2)
        plt.plot(x_grid, y_pred, color="green")
        plt.title(f"{country_code} - {y_col} vs {x_col}")
        plt.show()

    scatterplot(df=df, x_col="spend", y_col="installs", country_code="US")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
