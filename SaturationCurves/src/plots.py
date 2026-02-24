import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.dataframe_util import _XARRAY_DATASET_TYPE_STR
from curves import get_logistic_params, logistic_curve


def get_intervals(data, jump: int = 1):
    """Groups a sorted list of numbers into continuous intervals."""
    if not data:
        return []

    # data = sorted(data)
    intervals = []
    start = data[0]
    end = data[0]

    for i in range(1, len(data)):
        if data[i] == end + jump:
            end = data[i]
        else:
            intervals.append([start, end + jump])
            start = data[i]
            end = data[i]

    # Append the last interval
    intervals.append([start, end + jump])
    return intervals


def plot_saturation_curve(df: pd.DataFrame, x_col: str, y_col: str, threshold: float):
    # TODO: check if x, y in df

    if df.empty:
        st.write("No data")
        return

    # estimate logistic
    x, y = df[x_col].to_numpy(dtype=float), df[y_col].to_numpy(dtype=float)
    L, k, x0 = get_logistic_params(logistic_curve, x, y)

    # compute asymptotes
    asymptote_percentiles = [0.50, 0.75, 0.95]
    asymptote_colors = ["cyan", "orange", "red"]
    asymptotes = []
    for percentile in asymptote_percentiles:
        spend_asymptote = round(x0 + (1.0 / k) * np.log(percentile / (1.0 - percentile)), 2)
        asymptotes.append((percentile, spend_asymptote))

    # plot logistic
    max_spend_asymptote = max(spend for _, spend in asymptotes)
    #  x_grid = np.linspace( 0, max(x.max(), max_spend_asymptote), 100)
    x_grid = np.arange(0, max(x.max(), max_spend_asymptote), 100)
    y_pred = logistic_curve(x_grid, L, k, x0)

    # TODO: compute range where roas between x and y
    if y_col == "revenue":
        #  threshold = 0.45
        ratios = y_pred / x_grid
        ratio_col_name = "ROAS"
        best_ratio_mask = np.isfinite(ratios) & (ratios != 0.0)
        best_ratio = ratios[np.where(best_ratio_mask)][10:].max()

        threshold_mask = np.isfinite(ratios) & (ratios > threshold)
    else:
        #  threshold = 70
        ratios = x_grid / y_pred
        ratio_col_name = "CAC"
        best_ratio_mask = np.isfinite(ratios) & (ratios != 0.0)
        best_ratio = ratios[np.where(best_ratio_mask)][10:].min()
        threshold_mask = np.isfinite(ratios) & (ratios < threshold)

    # compute intervals
    threshold_idx = np.where(threshold_mask)[0]
    threshold_range = x_grid[threshold_idx]
    threshold_intervals = get_intervals(threshold_range.tolist(), jump=100)
    intervals_string = ";".join([f"{s:,.0f}-{e:,.0f}" for s, e in threshold_intervals])
    st.write(f"Valid Intervals: {intervals_string}")

    # TODO: compute marginal
    #  current_marginal_return = logistic_marginal_return(np.mean(x), L, k, x0)
    #  incremental_per_1K = 1000 * current_marginal_return
    #  print(f"Incremental {y_col} per 1K {x_col}: {incremental_per_1K}")

    fig = go.Figure()

    # Scatter points
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            #  x=dff[x_col],
            #  y=dff[y_col],
            mode="markers",
            marker=dict(color="blue", opacity=0.2),
            name="Data",
        )
    )

    # Fitted curve - TODO: add template
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=y_pred,
            mode="lines",
            line=dict(color="green"),
            name="Fit",
            customdata=np.column_stack([ratios]),
            hovertemplate=(
                f"{x_col}: %{{x:,.0f}}<br>"
                f"{y_col}: %{{y:,.0f}}<br>"
                f"{ratio_col_name}: %{{customdata[0]:.3f}}<br>"
                "<extra>Fitted curve</extra>"
            ),
        )
    )

    for i, ((percentile, spend_asymptote), color) in enumerate(zip(asymptotes, asymptote_colors)):
        fig.add_vline(
            x=spend_asymptote,
            line=dict(color=color, dash="dash"),
        )
        fig.add_annotation(
            x=spend_asymptote,
            y=1.02 - (i * 0.08),
            xref="x",
            yref="paper",
            text=f"Asymptote {percentile * 100:.0f}%: {spend_asymptote}",
            showarrow=False,
            font=dict(color=color),
            xanchor="left",
            align="left",
        )

    fig.update_layout(
        #  title=f"{network} {country_code} {platform} - {y_col} vs {x_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        legend=dict(x=1, xanchor="right"),
        template="plotly_white",
    )

    st.write(f"Best {ratio_col_name}: {best_ratio}")

    #  fig.show()
    st.plotly_chart(fig, use_container_width=True)


def plot_spend_marginal_timeline():
    pass
