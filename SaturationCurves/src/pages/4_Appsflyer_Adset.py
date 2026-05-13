"""
Build a streamlit page which shows day-over-day (DoD), week-over-week (WoW)

What the script does:
1. Getting the data
    a. Fetch the data from Appsflyer Master API
    b. Rename dataframe columns
    c. Create dataframes: groups the data at different dates aggregation at the (install_date, network, platform, country, campaign, adset) granularity
        - df_dod: day-over-day (DoD)
        - df_wow: week-over-week (WoW)
        - df_mom: month-over-month (MoM)
    d. Compute the following rates
        - CPM — Cost per 1,000 impressions → cost / impressions * 1000
        - CTR — Click-through rate → clicks / impressions
        - CPI — Cost per install → cost / installs
        - CPT — Cost per trial → cost / subscription_process_succeed
        - CAC — Cost to acquire a paying user → cost / af_subscribe_unique_users
        - I2T - Install-to-trial, the share of installs starters who converted to trials: subscription_process_succeed / installs
        - I2P - Install-to-paid, the share of installs startes who converted to trials: af_subscribe_unique_users / installs
        - T2P — Trial-to-paid rate, the share of trial starters who converted to a paid sub → af_subscribe_unique_users / subscription_process_succeed
        - RPP — Revenue per paying user → af_subscribe_sales_in_usd / af_subscribe_unique_users
        - RPP Net - Revenue Net per paying user → (af_subscribe_sales_in_usd - af_refund_sales_in_usd) / af_subscribe_unique_users
        - ROAS — Return on ad spend, net of refunds → (af_subscribe_sales_in_usd - af_refund_sales_in_usd) / cost
        - Refund rate — Share of subscribers who refunded → af_refund_unique_users / af_subscribe_unique_users
2. Create the streamlit UI
    a. Create dropdown for the following fields: network, platform, country
        - networks filters should have only: "Apple Search Ads", "Facebook Ads", "googleadwords_int", "tiktokglobal_int", "snapchat_int"
        - platform filters should only be: "ios", "android", "web"
        - if no filter are applied, then we select all values to compute kpis/metrics
    b. Create side-by-side 3 plotly graphs which shows T2P, ROAS, RPP
        - DoD view for the past 2 weeks (default)
        - WoW view for the past 2 months
    c. When we click on the plotly graph for a given day, we show a table at the campaign level
       around that given day with N_PREVIOUS_DAYS (apply the same filters as the dropdown). Breakdown on 2 level: campaign, date
       Toggle for 2 modes:
        - Campaign-Date:
            > campaign 1: overall rates
                - date 1: rates per day
                - date 2
            > campaign 2
        - Date-Campaign

Ensure that the UI output ressemble the following image: `assets/images/dod_wow_mom_view.png`


Notes:
- streamlit should cache the fetched data with ttl of 2 hours
- Order campaign by decreasing order
- ROAS should be percentage
- Add red color for days with bad ROAS
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from queries import adset_master_api_query
from utils import get_bigquery_client, run_query

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

N_PREVIOUS_DAYS = 3
DATA_MATURITY_LAG = 8  # Appsflyer attribution matures after ~8 days; exclude recent immature data

BASE_METRIC_COLS = [
    "cost",
    "impressions",
    "clicks",
    "installs",
    "subscription_process_succeed",
    "af_subscribe_unique_users",
    "af_subscribe_sales_in_usd",
    "af_refund_unique_users",
    "af_refund_sales_in_usd",
]

RATE_COLS = [
    "T2P",
    "I2T",
    "I2P",
    "ROAS",
    "RPP",
    "RPP_net",
    "CAC",
    "CPI",
    "CPT",
    "CPM",
    "CTR",
    "refund_rate",
]

ALL_DISPLAY_COLS = RATE_COLS + [
    "cost",
    "installs",
    "impressions",
    "clicks",
    "subscription_process_succeed",
    "af_subscribe_unique_users",
    "af_subscribe_sales_in_usd",
    "af_refund_unique_users",
    "af_refund_sales_in_usd",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=7200)
def load_data(start_date: str, end_date: str) -> pd.DataFrame:
    logging.info("Fetching from BigQuery adset_master_api_query (%s → %s)", start_date, end_date)
    query = adset_master_api_query.replace(
        "declare start_date date default date_sub(current_date('UTC'), interval 90 day);",
        f"declare start_date date default date '{start_date}';",
    ).replace(
        "declare end_date date default current_date('UTC');",
        f"declare end_date date default date '{end_date}';",
    )
    client = get_bigquery_client()
    df = run_query(client, query)
    if df.empty:
        return df

    df["install_date"] = pd.to_datetime(df["install_date"])
    for col in BASE_METRIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df_iso = df["install_date"].dt.isocalendar()
    df["iso_week"] = df_iso["week"]
    df["iso_week_start"] = df["install_date"] - pd.to_timedelta(
        df["install_date"].dt.weekday, unit="D"
    )
    df["year_month"] = df["install_date"].dt.to_period("M").dt.to_timestamp()
    return df


def compute_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived KPIs. Uses np.where so zero denominators yield NaN, not 0."""
    df = df.copy()
    imp = df["impressions"]
    trials = df["subscription_process_succeed"]
    subs = df["af_subscribe_unique_users"]
    rev = df["af_subscribe_sales_in_usd"]
    refunds = df["af_refund_sales_in_usd"]
    refund_users = df["af_refund_unique_users"]
    cost = df["cost"]
    installs = df["installs"]

    df["CPM"] = np.where(imp > 0, cost / imp * 1000, np.nan)
    df["CTR"] = np.where(imp > 0, df["clicks"] / imp, np.nan)
    df["CPI"] = np.where(installs > 0, cost / installs, np.nan)
    df["CPT"] = np.where(trials > 0, cost / trials, np.nan)
    df["CAC"] = np.where(subs > 0, cost / subs, np.nan)
    df["I2T"] = np.where(installs > 0, trials / installs, np.nan)
    df["I2P"] = np.where(installs > 0, subs / installs, np.nan)
    df["T2P"] = np.where(trials > 0, subs / trials, np.nan)
    df["RPP"] = np.where(subs > 0, rev / subs, np.nan)
    df["RPP_net"] = np.where(subs > 0, (rev - refunds) / subs, np.nan)
    df["ROAS"] = np.where(cost > 0, (rev - refunds) / cost * 100, np.nan)
    df["refund_rate"] = np.where(subs > 0, refund_users / subs, np.nan)
    return df


def _aggregate(df: pd.DataFrame, date_col: str, extra_dims: list[str]) -> pd.DataFrame:
    group_cols = [date_col] + extra_dims
    agg = df.groupby(group_cols, as_index=False)[BASE_METRIC_COLS].sum()
    return compute_rates(agg)


def make_dod(df: pd.DataFrame, extra_dims: list[str] = []) -> pd.DataFrame:
    return _aggregate(df, "install_date", extra_dims)


def make_wow(df: pd.DataFrame, extra_dims: list[str] = []) -> pd.DataFrame:
    return _aggregate(df, "iso_week_start", extra_dims)


def make_mom(df: pd.DataFrame, extra_dims: list[str] = []) -> pd.DataFrame:
    return _aggregate(df, "year_month", extra_dims)


def apply_filters(
    df: pd.DataFrame, networks: list, platforms: list, countries: list
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if networks:
        mask &= df["network"].isin(networks)
    if platforms:
        mask &= df["platform"].isin(platforms)
    if countries:
        mask &= df["country"].isin(countries)
    return df[mask]


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _delta_pct(df_sorted: pd.DataFrame, metric: str) -> float | None:
    """% change from second-to-last to last value in the visible window."""
    vals = df_sorted[metric].dropna().values
    if len(vals) < 2 or vals[-2] == 0:
        return None
    return float((vals[-1] - vals[-2]) / abs(vals[-2]) * 100)


def _badge_html(pct: float | None) -> str:
    if pct is None:
        return ""
    green = pct >= 0
    color = "#1a6e2e" if green else "#9a1c1c"
    bg = "#d4edda" if green else "#f8d7da"
    sign = "+" if green else ""
    return (
        f'<span style="background:{bg};color:{color};padding:2px 9px;'
        f'border-radius:12px;font-size:0.8em;font-weight:600;vertical-align:middle">'
        f"{sign}{pct:.1f}%</span>"
    )


def _mini_chart(df_view: pd.DataFrame, x_col: str, metric: str, label: str, chart_key: str):
    df_sorted = df_view.sort_values(x_col)
    delta = _delta_pct(df_sorted, metric)
    st.markdown(
        f'<p style="font-size:1.05em;font-weight:700;margin-bottom:2px">'
        f"{label} &nbsp; {_badge_html(delta)}</p>",
        unsafe_allow_html=True,
    )
    if df_view.empty or metric not in df_view.columns or df_view[metric].isna().all():
        st.warning(f"No data for {metric}")
        return None
    fig = px.line(df_sorted, x=x_col, y=metric, markers=True)
    fig.update_traces(
        line_color="#2563eb",
        marker=dict(size=7, color="#2563eb"),
    )
    fig.update_layout(
        margin=dict(t=4, b=4, l=4, r=4),
        height=200,
        xaxis_title=None,
        yaxis_title=None,
        showlegend=False,
        plot_bgcolor="white",
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0"),
    )
    ev = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key=chart_key)
    st.caption("click a day to drill down ↓")
    return ev


_TABLE_COLS = [
    ("cost", "${:,.0f}", "SPEND"),
    ("impressions", "{:,.0f}", "IMP"),
    ("clicks", "{:,.0f}", "CLICKS"),
    ("installs", "{:,.0f}", "INSTALLS"),
    ("subscription_process_succeed", "{:,.0f}", "TRIALS"),
    ("af_subscribe_unique_users", "{:,.0f}", "SUBS"),
    ("af_subscribe_sales_in_usd", "${:,.2f}", "REV"),
    #  ("af_refund_unique_users", "{:,.0f}", "RFND U"),
    #  ("af_refund_sales_in_usd", "${:,.2f}", "RFND $"),
    ("CTR", "{:.2%}", "CTR"),
    ("CPM", "${:,.2f}", "CPM"),
    ("CPI", "${:.2f}", "CPI"),
    ("CPT", "${:.2f}", "CPT"),
    ("I2T", "{:.1%}", "I2T"),
    ("I2P", "{:.1%}", "I2P"),
    ("T2P", "{:.1%}", "T2P"),
    ("ROAS", "{:.1f}%", "ROAS"),
    ("RPP", "${:.2f}", "RPP"),
    ("CAC", "${:.2f}", "CAC"),
]

_BREAKDOWN_STYLE = """<style>
.bkd details {{ margin: 0; }}
.bkd details summary {{ list-style: none; display: block; cursor: pointer; padding: 0; margin: 0; }}
.bkd details summary::-webkit-details-marker {{ display: none; }}
.bkd .arrow::before {{ content: "▶"; font-size: 0.7em; color: #555; margin-right: 6px; }}
.bkd details[open] .arrow::before {{ content: "▼"; }}
.bkd .brow {{ display: grid; grid-template-columns: 3fr {cols}; }}
.bkd .bc   {{ padding: 10px 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.bkd .bcr  {{ padding: 10px 8px; text-align: right; white-space: nowrap; }}
.bkd .bcs  {{ padding: 8px 8px 8px 28px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.bkd .bcsr {{ padding: 8px 8px; text-align: right; white-space: nowrap; }}
</style>""".format(
    cols=" ".join(["1fr"] * len(_TABLE_COLS))
)


def _fv(val, fmt: str) -> str:
    try:
        return "—" if pd.isna(val) else fmt.format(float(val))
    except Exception:
        return "—"


def _metric_cells(
    series: pd.Series, bold: bool = False, blue: bool = False, sub: bool = False
) -> str:
    cls = "bcsr" if sub else "bcr"
    extra = ("font-weight:700;" if bold else "") + ("color:#2563eb;" if blue else "")
    style = f' style="{extra}"' if extra else ""
    out = ""
    for key, fmt, _ in _TABLE_COLS:
        v = series.get(key, np.nan)
        out += f'<div class="{cls}"{style}>{_fv(v, fmt)}</div>'
    return out


def _build_breakdown_html(
    agg: pd.DataFrame,
    agg_adset: pd.DataFrame,
    center,
    campaign_order: list,
    breakdown_mode: str,
) -> str:
    center_str = str(center)

    first_col_label = {
        "Campaign → Adset": "CAMPAIGN / ADSET",
        "Campaign → Date": "CAMPAIGN / DATE",
        "Date → Campaign": "DATE / CAMPAIGN",
    }.get(breakdown_mode, "CAMPAIGN / DATE")

    hdr = (
        '<div class="brow" style="border-bottom:2px solid #ddd;background:#fafafa;">'
        f'<div class="bc" style="color:#888;font-size:0.78em;font-weight:700;text-transform:uppercase">{first_col_label}</div>'
        + "".join(
            f'<div class="bcr" style="color:#888;font-size:0.78em;font-weight:700;text-transform:uppercase">{lbl}</div>'
            for _, _, lbl in _TABLE_COLS
        )
        + "</div>"
    )

    body = ""

    if breakdown_mode == "Campaign → Adset":
        for i, campaign in enumerate(campaign_order):
            grp = agg_adset[agg_adset["campaign"] == campaign]
            if grp.empty:
                continue
            totals = compute_rates(grp[BASE_METRIC_COLS].sum().to_frame().T).iloc[0]

            summary = (
                '<div class="brow" style="background:#f5f5f5;border-bottom:1px solid #ddd;">'
                f'<div class="bc" style="font-weight:700"><span class="arrow"></span>{campaign}</div>'
                + _metric_cells(totals, bold=True)
                + "</div>"
            )

            adsets_html = ""
            for _, row in grp.sort_values("cost", ascending=False).iterrows():
                adsets_html += (
                    '<div class="brow" style="border-bottom:1px solid #f0f0f0;">'
                    f'<div class="bcs">{row["ad"]}</div>' + _metric_cells(row, sub=True) + "</div>"
                )

            open_attr = "open" if i == 0 else ""
            body += f"<details {open_attr}><summary>{summary}</summary>{adsets_html}</details>"

    elif breakdown_mode == "Campaign → Date":
        for i, campaign in enumerate(campaign_order):
            grp = agg[agg["campaign"] == campaign]
            totals = compute_rates(grp[BASE_METRIC_COLS].sum().to_frame().T).iloc[0]

            summary = (
                '<div class="brow" style="background:#f5f5f5;border-bottom:1px solid #ddd;">'
                f'<div class="bc" style="font-weight:700"><span class="arrow"></span>{campaign}</div>'
                + _metric_cells(totals, bold=True)
                + "</div>"
            )

            dates_html = ""
            for _, row in grp.sort_values("install_date").iterrows():
                ds = str(row["install_date"])
                sel = ds == center_str
                label = f"{ds} ●" if sel else ds
                extra = 'style="color:#2563eb;font-weight:600;"' if sel else ""
                dates_html += (
                    '<div class="brow" style="border-bottom:1px solid #f0f0f0;">'
                    f'<div class="bcs" {extra}>{label}</div>'
                    + _metric_cells(row, blue=sel, sub=True)
                    + "</div>"
                )

            open_attr = "open" if i == 0 else ""
            body += f"<details {open_attr}><summary>{summary}</summary>{dates_html}</details>"

    else:  # Date → Campaign
        for date, grp in agg.groupby("install_date"):
            ds = str(date)
            sel = ds == center_str
            totals = compute_rates(grp[BASE_METRIC_COLS].sum().to_frame().T).iloc[0]
            label = f"{ds} ●" if sel else ds
            extra = 'style="color:#2563eb;"' if sel else ""

            summary = (
                '<div class="brow" style="background:#f5f5f5;border-bottom:1px solid #ddd;">'
                f'<div class="bc" style="font-weight:700;" {extra}><span class="arrow"></span>{label}</div>'
                + _metric_cells(totals, bold=True, blue=sel)
                + "</div>"
            )

            campaigns_html = ""
            rank = {c: i for i, c in enumerate(campaign_order)}
            grp = grp.copy()
            grp["_rank"] = grp["campaign"].map(rank).fillna(len(rank))
            grp = grp.sort_values("_rank").drop(columns="_rank")
            for _, row in grp.iterrows():
                campaigns_html += (
                    '<div class="brow" style="border-bottom:1px solid #f0f0f0;">'
                    f'<div class="bcs">{row["campaign"]}</div>'
                    + _metric_cells(row, sub=True)
                    + "</div>"
                )

            open_attr = "open" if sel else ""
            body += f"<details {open_attr}><summary>{summary}</summary>{campaigns_html}</details>"

    return (
        _BREAKDOWN_STYLE
        + '<div class="bkd" style="border:1px solid #e0e0e0;border-radius:8px;overflow:hidden;margin-top:4px;">'
        + hdr
        + body
        + "</div>"
    )


def _show_campaign_breakdown(
    df_raw: pd.DataFrame,
    lo: pd.Timestamp,
    hi: pd.Timestamp,
    networks: list,
    platforms: list,
    countries: list,
    center=None,  # set when triggered by a chart click (used for label + highlight)
) -> None:
    window_df = apply_filters(df_raw, networks, platforms, countries)
    window_df = window_df[(window_df["install_date"] >= lo) & (window_df["install_date"] <= hi)]

    if window_df.empty:
        st.info("No campaign data in this window.")
        return

    agg = window_df.groupby(["campaign", "install_date"], as_index=False)[BASE_METRIC_COLS].sum()
    agg = compute_rates(agg)
    agg["install_date"] = agg["install_date"].dt.date

    agg_adset = window_df.groupby(["campaign", "ad"], as_index=False)[BASE_METRIC_COLS].sum()
    agg_adset = compute_rates(agg_adset)

    campaign_order = (
        agg.groupby("campaign")["cost"].sum().sort_values(ascending=False).index.tolist()
    )

    if center is not None:
        date_badge = f"📅 {center} ± {N_PREVIOUS_DAYS} days"
        caption_suffix = f"±{N_PREVIOUS_DAYS} days around clicked point"
    else:
        date_badge = f"📅 {lo.date()} → {hi.date()}"
        caption_suffix = "display date range"

    hdr_col, toggle_col = st.columns([3, 2])
    with hdr_col:
        st.markdown(
            f'<span style="font-size:1.05em;font-weight:700">Campaign breakdown</span> &nbsp;'
            f'<span style="background:#dbeafe;color:#1d4ed8;padding:3px 10px;'
            f'border-radius:12px;font-size:0.85em">{date_badge}</span>',
            unsafe_allow_html=True,
        )
    with toggle_col:
        breakdown_mode = st.radio(
            "mode",
            ["Campaign → Adset", "Campaign → Date", "Date → Campaign"],
            horizontal=True,
            key="breakdown_mode",
            label_visibility="collapsed",
        )

    st.markdown(
        _build_breakdown_html(agg, agg_adset, center, campaign_order, breakdown_mode),
        unsafe_allow_html=True,
    )

    net_str = " · ".join(networks) if networks else "all networks"
    plat_str = " · ".join(platforms) if platforms else "all platforms"
    ctry_str = " · ".join(countries) if countries else "all countries"
    st.caption(
        f"● selected day | window: {caption_suffix} | "
        f"filters: {net_str} · {plat_str} · {ctry_str}"
    )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Appsflyer Adset", page_icon="📊", layout="wide")
st.markdown("# Appsflyer Adset Performance")

today = datetime.today().date()
fetch_start = today - timedelta(days=90)
fetch_end = today - timedelta(days=1)

# Always fetch the full 90-day window; date range below only controls what is displayed.
df_raw = load_data(fetch_start.strftime("%Y-%m-%d"), fetch_end.strftime("%Y-%m-%d"))

if df_raw.empty:
    st.error("No data returned from BigQuery. Check your credentials and date range.")
    st.stop()

st.info(
    f"Data loaded for the last 90 days ({fetch_start} → {fetch_end}). "
    "Use the date range below to narrow what is shown in the charts and table."
)

# --- Date range (display filter, not fetch) ---
col_dates, col_dl = st.columns([3, 1], vertical_alignment="bottom")
with col_dates:
    date_range = st.date_input(
        "Display date range",
        value=(fetch_start, fetch_end),
        min_value=fetch_start,
        max_value=fetch_end,
        key="date_range",
    )
with col_dl:
    st.download_button(
        label="⬇ Download raw data",
        data=df_raw.to_csv(index=False),
        file_name=f"adset_{fetch_start}_{fetch_end}.csv",
        mime="text/csv",
    )

if not (isinstance(date_range, (list, tuple)) and len(date_range) == 2):
    st.warning("Please select a complete date range.")
    st.stop()

display_start = pd.Timestamp(date_range[0])
display_end = pd.Timestamp(date_range[1])

# --- Horizontal filter bar ---
all_networks = sorted(df_raw["network"].dropna().unique().tolist())
all_platforms = sorted(df_raw["platform"].dropna().unique().tolist())
all_countries = sorted(df_raw["country"].dropna().unique().tolist())

col_net, col_plat, col_ctr, col_data, col_view = st.columns([2, 2, 2, 1.5, 2])
with col_net:
    sel_networks = st.multiselect("Network", options=all_networks, placeholder="All networks")
with col_plat:
    sel_platforms = st.multiselect("Platform", options=all_platforms, placeholder="All platforms")
with col_ctr:
    sel_countries = st.multiselect("Country", options=all_countries, placeholder="All countries")
with col_data:
    data_mode = st.radio("Data", ["Mature", "Unmature"], horizontal=True)
with col_view:
    view = st.radio("View", ["DoD – 2 wks", "WoW – 2 mo"], horizontal=True)

st.divider()

# --- Filter & aggregate ---
df_filtered = apply_filters(df_raw, sel_networks, sel_platforms, sel_countries)
df_filtered = df_filtered[
    (df_filtered["install_date"] >= display_start) & (df_filtered["install_date"] <= display_end)
]
if data_mode == "Mature":
    maturity_end = pd.Timestamp(today - timedelta(days=DATA_MATURITY_LAG))
    df_filtered = df_filtered[df_filtered["install_date"] <= maturity_end]

if view.startswith("DoD"):
    df_view = make_dod(df_filtered)
    x_col = "install_date"
else:
    df_view = make_wow(df_filtered)
    x_col = "iso_week_start"

if df_view.empty:
    st.warning("No data for the selected filters. Adjust your selections.")
    st.stop()

# --- Three side-by-side charts ---
col1, col2, col3 = st.columns(3)
with col1:
    ev_t2p = _mini_chart(df_view, x_col, "T2P", "T2P", "chart_t2p")
with col2:
    ev_roas = _mini_chart(df_view, x_col, "ROAS", "ROAS", "chart_roas")
with col3:
    ev_rpp = _mini_chart(df_view, x_col, "RPP", "RPP", "chart_rpp")

st.divider()

# --- Click → campaign breakdown ---
# Detect a fresh chart click this rerun (from live events, not session state).
fresh_click = None
for ev in [ev_t2p, ev_roas, ev_rpp]:
    if ev and hasattr(ev, "selection") and ev.selection and ev.selection.points:
        raw_x = ev.selection.points[0].get("x", "")
        if raw_x:
            fresh_click = str(raw_x)
            break

# Detect a date range change this rerun.
prev_date_range = st.session_state.get("prev_date_range")
date_range_changed = prev_date_range != date_range
st.session_state["prev_date_range"] = date_range

# Update last_trigger based on what changed this rerun.
if fresh_click:
    st.session_state["selected_date"] = fresh_click
    st.session_state["last_trigger"] = "chart_click"
elif date_range_changed:
    st.session_state["last_trigger"] = "date_range"

last_trigger = st.session_state.get("last_trigger")
selected_date = st.session_state.get("selected_date")

if last_trigger == "chart_click" and selected_date:
    try:
        center = pd.to_datetime(selected_date).date()
    except Exception:
        center = None
    if center:
        _show_campaign_breakdown(
            df_raw,
            lo=pd.Timestamp(center - timedelta(days=N_PREVIOUS_DAYS)),
            hi=pd.Timestamp(center + timedelta(days=N_PREVIOUS_DAYS)),
            networks=sel_networks,
            platforms=sel_platforms,
            countries=sel_countries,
            center=center,
        )
elif last_trigger == "date_range":
    _show_campaign_breakdown(
        df_raw,
        lo=display_start,
        hi=display_end,
        networks=sel_networks,
        platforms=sel_platforms,
        countries=sel_countries,
    )
else:
    st.info("Click a point on any chart above to see the campaign breakdown.")
