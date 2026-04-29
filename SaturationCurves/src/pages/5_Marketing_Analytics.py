"""

Create a streamlit marketing app with the following tabs:
- Performance Summary => `assets/images/ua_performance_sumary.png`
    * We want to show the overall performance summary: revenue, total_spend, impressions, clicks, installs, trials, paid, blended cac, roas
    * we want to show spend & revenue trend
    * we want to show spend allocation in a pie chart with their respective spend percentage
    * we want to have a channel performance summary with the following metrics: spend, clicks, CTR, installs, CPI, trials, I2T, paid, T2P, I2P, CPA, CAC, ROAS
    * we want to have install volume by channel, which is a stacked bar graph
- Network Deep Dive => `assets/images/ua_network_deep_dive.png`
    * we want to show the channel overall performance for (Facebook Ads, Google, tiktok, tatari, Apple Search Ads) for the metrics: installs, trials, paid, CPI, CAC, Trial Rate, T2P, ROAS
    * we want to have a cac and roas comparison by channel
    * we want to have a conversion funnel by channel
- Campaign Drill Down => `assets/images/ua_campaign_drilldown.png`

- Funnel Activity => `assets/images/ua_funnel_activity.png`

Steps:
1. Pre-computing the data
    a. Query the data from BigQuery using `queries.yearly_final_table_data`. The data should be cached with a TTL of 5 hours => df_raw
    b. Compute dataframes at several grain: daily (df_daily), weekly (df_weekly), monthly (df_monthly), WTD. All dates should be isodates
2. UI
    a. There should be several tabs:
        - Performance Summary => `assets/images/ua_performance_sumary.png`
        - Network Deep Dive => `assets/images/ua_network_deep_dive.png`
        - Campaign Drill Down => `assets/images/ua_campaign_drilldown.png`
        - Funnel Activity => `assets/images/ua_funnel_activity.png`
        Important: Look at the images to get a feel for the output. aim to create something similar
    b. There should be a Daily,Weekly,Monthly,YTD,Custom selector. The date
       grain should match what the user see in all tabs

IMPORTANT Notes:
- Keep the instructions when modifying the code
- Use plotly to create the graph. Make sure that we can see important information when hovering over the graphs
- When showing columns, show the rows sorted descending based on spend except if there are dates involved


"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import date, timedelta

from queries import yearly_final_table_data_aggregated
from utils import get_bigquery_client, run_query

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NETWORK_DISPLAY = {
    "Facebook Ads": "Meta",
    "tiktokglobal_int": "TikTok",
    "googleadwords_int": "Google",
    "tatari": "Tatari",
    "snapchat_int": "Snapchat",
    "Apple Search Ads": "Apple Search Ads",
}

NETWORK_COLORS = {
    "Meta": "#407076",
    "Google UAC": "#698996",
    "TikTok": "#97B1A6",
    "Apple Search Ads": "#C9C5BA",
    "Tatari": "#EBBAB9",
    "Snapchat": "#2A4D52",
}
#  NETWORK_COLORS = {
#  "Meta": "#335C67",
#  "Google UAC": "#E09F3E",
#  "TikTok": "#9E2A2B",
#  "Apple Search Ads": "#FFF3B0",
#  "Tatari": "#540B0E",
#  "Snapchat": "#4A7C87",
#  }

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------


@st.cache_data(ttl=18000)
def load_data() -> pd.DataFrame:
    client = get_bigquery_client()
    df = run_query(client=client, query=yearly_final_table_data_aggregated)
    df["date"] = pd.to_datetime(df["date"])
    df["network_display"] = df["network"].map(NETWORK_DISPLAY).fillna(df["network"])
    df["spend"] = df["cost_usd"].fillna(0)
    # paid/revenue already resolved in SQL (no need_modeling flag needed)
    df["paid_eff"] = df["paid"].clip(lower=0)
    df["revenue_eff"] = df["revenue"].clip(lower=0)
    for col in ["impressions", "clicks", "installs", "trials"]:
        df[col] = df[col].fillna(0)
    return df


@st.cache_data(ttl=18000)
def get_grain_data(grain: str, custom_start=None, custom_end=None) -> pd.DataFrame:
    df = load_data()
    filtered = filter_by_grain(df, grain, custom_start, custom_end)
    return add_period_col(filtered, grain)


def aggregate_metrics(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    agg = df.groupby(group_cols, as_index=False).agg(
        spend=("spend", "sum"),
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        installs=("installs", "sum"),
        trials=("trials", "sum"),
        paid=("paid_eff", "sum"),
        revenue=("revenue_eff", "sum"),
    )
    agg["CTR"] = np.where(agg["impressions"] > 0, agg["clicks"] / agg["impressions"], np.nan)
    agg["CPI"] = np.where(agg["installs"] > 0, agg["spend"] / agg["installs"], np.nan)
    agg["CPT"] = np.where(agg["trials"] > 0, agg["spend"] / agg["trials"], np.nan)
    agg["I2T"] = np.where(agg["installs"] > 0, agg["trials"] / agg["installs"], np.nan)
    agg["T2P"] = np.where(agg["trials"] > 0, agg["paid"] / agg["trials"], np.nan)
    agg["I2P"] = np.where(agg["installs"] > 0, agg["paid"] / agg["installs"], np.nan)
    agg["CAC"] = np.where(agg["paid"] > 0, agg["spend"] / agg["paid"], np.nan)
    agg["CPA"] = agg["CAC"]
    agg["ROAS"] = np.where(agg["spend"] > 0, agg["revenue"] / agg["spend"], np.nan)
    return agg


# ---------------------------------------------------------------------------
# Date filtering helpers
# ---------------------------------------------------------------------------


def filter_by_grain(
    df: pd.DataFrame,
    grain: str,
    custom_start=None,
    custom_end=None,
) -> pd.DataFrame:
    today = pd.Timestamp.today().normalize()
    if grain == "Daily":
        return df[df["date"] >= today - pd.Timedelta(days=30)]
    if grain == "Weekly":
        return df[df["date"] >= today - pd.Timedelta(weeks=12)]
    if grain == "Monthly":
        return df[df["date"] >= today - pd.DateOffset(months=12)]
    if grain == "YTD":
        return df[df["date"] >= pd.Timestamp(today.year, 1, 1)]
    if grain == "Custom" and custom_start and custom_end:
        return df[
            (df["date"] >= pd.Timestamp(custom_start)) & (df["date"] <= pd.Timestamp(custom_end))
        ]
    return df


def add_period_col(df: pd.DataFrame, grain: str) -> pd.DataFrame:
    df = df.copy()
    if grain == "Weekly":
        df["period"] = df["date"].dt.to_period("W").apply(lambda p: p.start_time)
    elif grain == "Monthly":
        df["period"] = df["date"].dt.to_period("M").apply(lambda p: p.start_time)
    else:
        df["period"] = df["date"]
    return df


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------


def fmt_currency(v, decimals: int = 0) -> str:
    if pd.isna(v):
        return "—"
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:.0f}K"
    return f"${v:.{decimals}f}"


def fmt_pct(v) -> str:
    if pd.isna(v):
        return "—"
    return f"{v * 100:.1f}%"


def fmt_num(v) -> str:
    if pd.isna(v):
        return "—"
    if abs(v) >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"{v / 1_000:.0f}K"
    return f"{v:.0f}"


def fmt_x(v) -> str:
    return f"{v:.2f}x" if not pd.isna(v) else "—"


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Marketing Analytics", page_icon="📊", layout="wide")
st.markdown("## 📊 Marketing Analytics")

# Date grain selector
col_grain, col_custom = st.columns([3, 4])
with col_grain:
    grain = st.radio(
        "Date Grain",
        ["Daily", "Weekly", "Monthly", "YTD", "Custom"],
        horizontal=True,
    )

custom_start, custom_end = None, None
if grain == "Custom":
    with col_custom:
        c1, c2 = st.columns(2)
        custom_start = c1.date_input("Start date", value=date.today() - timedelta(days=30))
        custom_end = c2.date_input("End date", value=date.today())

# df_period has both filtered rows and a `period` column for trend charts;
# df_filtered is the same slice without the period grouping column.
df_period = get_grain_data(grain, custom_start, custom_end)
df_filtered = df_period

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Performance Summary",
        "🌐 Network Deep Dive",
        "🎯 Campaign Drill Down",
        "🔻 Funnel Activity",
    ]
)


# ===========================================================================
# TAB 1 — Performance Summary
# ===========================================================================
with tab1:
    totals = {
        "spend": df_filtered["spend"].sum(),
        "impressions": df_filtered["impressions"].sum(),
        "clicks": df_filtered["clicks"].sum(),
        "installs": df_filtered["installs"].sum(),
        "trials": df_filtered["trials"].sum(),
        "paid": df_filtered["paid_eff"].sum(),
        "revenue": df_filtered["revenue_eff"].sum(),
    }

    blended_cac = totals["spend"] / totals["paid"] if totals["paid"] > 0 else np.nan
    roas = totals["revenue"] / totals["spend"] if totals["spend"] > 0 else np.nan

    # KPI cards
    kpi_cols = st.columns(8)
    kpis = [
        ("Revenue", fmt_currency(totals["revenue"])),
        ("Total Spend", fmt_currency(totals["spend"])),
        ("Impressions", fmt_num(totals["impressions"])),
        ("Clicks", fmt_num(totals["clicks"])),
        ("Installs", fmt_num(totals["installs"])),
        ("Trials", fmt_num(totals["trials"])),
        ("Blended CAC", fmt_currency(blended_cac, 2)),
        ("ROAS", fmt_x(roas)),
    ]
    for col, (label, val) in zip(kpi_cols, kpis):
        col.metric(label, val)

    st.divider()

    # Spend & Revenue Trend  |  Spend Allocation pie
    col_trend, col_pie = st.columns([2, 1])

    with col_trend:
        trend = df_period.groupby("period", as_index=False).agg(
            spend=("spend", "sum"),
            revenue=("revenue_eff", "sum"),
        )
        fig_trend = go.Figure()
        fig_trend.add_trace(
            go.Scatter(
                x=trend["period"],
                y=trend["spend"],
                name="Spend",
                mode="lines+markers",
                line=dict(color="#636EFA"),
                hovertemplate="%{x|%Y-%m-%d}<br>Spend: $%{y:,.0f}<extra></extra>",
            )
        )
        fig_trend.add_trace(
            go.Scatter(
                x=trend["period"],
                y=trend["revenue"],
                name="Revenue",
                mode="lines+markers",
                line=dict(color="#00CC96"),
                yaxis="y2",
                hovertemplate="%{x|%Y-%m-%d}<br>Revenue: $%{y:,.0f}<extra></extra>",
            )
        )
        fig_trend.update_layout(
            title="Spend & Revenue Trend",
            yaxis=dict(title="Spend ($)", showgrid=False),
            yaxis2=dict(title="Revenue ($)", overlaying="y", side="right"),
            legend=dict(orientation="h", y=1.12),
            hovermode="x unified",
            height=320,
            margin=dict(t=50),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with col_pie:
        alloc = (
            df_filtered.groupby("network_display", as_index=False)
            .agg(spend=("spend", "sum"))
            .sort_values("spend", ascending=False)
        )
        fig_pie = px.pie(
            alloc,
            values="spend",
            names="network_display",
            title="Spend Allocation",
            hole=0.5,
            color="network_display",
            color_discrete_map=NETWORK_COLORS,
        )
        fig_pie.update_traces(
            textposition="outside",
            textinfo="label+percent",
            hovertemplate="%{label}<br>Spend: $%{value:,.0f}<br>Share: %{percent}<extra></extra>",
        )
        fig_pie.update_layout(showlegend=False, height=320, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # Channel Performance Summary table
    st.markdown("#### Channel Performance Summary")
    ch = aggregate_metrics(df_filtered, ["network_display"]).sort_values("spend", ascending=False)
    st.dataframe(
        pd.DataFrame(
            {
                "Channel": ch["network_display"],
                "Spend": ch["spend"].map(fmt_currency),
                "Clicks": ch["clicks"].map(fmt_num),
                "CTR": ch["CTR"].map(fmt_pct),
                "Installs": ch["installs"].map(fmt_num),
                "CPI": ch["CPI"].map(lambda v: fmt_currency(v, 2)),
                "Trials": ch["trials"].map(fmt_num),
                "I2T": ch["I2T"].map(fmt_pct),
                "Paid": ch["paid"].map(fmt_num),
                "T2P": ch["T2P"].map(fmt_pct),
                "I2P": ch["I2P"].map(fmt_pct),
                "CPA": ch["CPA"].map(lambda v: fmt_currency(v, 2)),
                "CAC": ch["CAC"].map(lambda v: fmt_currency(v, 2)),
                "ROAS": ch["ROAS"].map(fmt_x),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Install Volume stacked bar  |  CAC Trend by Channel
    col_bar, col_cac = st.columns(2)

    with col_bar:
        inst_trend = df_period.groupby(["period", "network_display"], as_index=False).agg(
            installs=("installs", "sum")
        )
        fig_bar = px.bar(
            inst_trend,
            x="period",
            y="installs",
            color="network_display",
            title="Install Volume by Channel (Stacked)",
            barmode="stack",
            color_discrete_map=NETWORK_COLORS,
            labels={
                "installs": "Installs",
                "period": "Date",
                "network_display": "Channel",
            },
        )
        fig_bar.update_traces(
            hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: %{y:,.0f}<extra></extra>"
        )
        fig_bar.update_layout(height=340, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_cac:
        cac_trend = df_period.groupby(["period", "network_display"], as_index=False).agg(
            spend=("spend", "sum"), paid=("paid_eff", "sum")
        )
        cac_trend["CAC"] = np.where(
            cac_trend["paid"] > 0, cac_trend["spend"] / cac_trend["paid"], np.nan
        )
        fig_cac = px.line(
            cac_trend,
            x="period",
            y="CAC",
            color="network_display",
            title="CAC Trend by Channel",
            color_discrete_map=NETWORK_COLORS,
            labels={"CAC": "CAC ($)", "period": "Date", "network_display": "Channel"},
        )
        fig_cac.update_traces(
            hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name} CAC: $%{y:,.2f}<extra></extra>"
        )
        fig_cac.update_layout(height=340, legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig_cac, use_container_width=True)


# ===========================================================================
# TAB 2 — Network Deep Dive
# ===========================================================================
with tab2:
    st.markdown("#### Channel Overall Performance")
    ch2 = aggregate_metrics(df_filtered, ["network_display"]).sort_values("spend", ascending=False)

    # Channel cards — 3 per row
    rows = [ch2.iloc[i : i + 3] for i in range(0, len(ch2), 3)]
    for row_df in rows:
        cols = st.columns(3)
        for col, (_, r) in zip(cols, row_df.iterrows()):
            with col:
                with st.container(border=True):
                    st.markdown(f"**{r['network_display']}**")
                    st.markdown(f"### {fmt_currency(r['spend'])}")
                    m1, m2 = st.columns(2)
                    m1.metric("Installs", fmt_num(r["installs"]))
                    m2.metric("CPI", fmt_currency(r["CPI"], 2))
                    m1.metric("Trials", fmt_num(r["trials"]))
                    m2.metric("Trial Rate", fmt_pct(r["I2T"]))
                    m1.metric("Paid", fmt_num(r["paid"]))
                    m2.metric("T2P", fmt_pct(r["T2P"]))
                    m1.metric("CAC", fmt_currency(r["CAC"], 2))
                    m2.metric("ROAS", fmt_x(r["ROAS"]))

    st.divider()

    # CAC and ROAS comparison bars
    col_cac2, col_roas2 = st.columns(2)

    with col_cac2:
        cac_df = ch2[["network_display", "CAC"]].dropna().sort_values("CAC")
        fig_cac_bar = px.bar(
            cac_df,
            x="CAC",
            y="network_display",
            orientation="h",
            title="CAC Comparison by Channel",
            color="network_display",
            color_discrete_map=NETWORK_COLORS,
            labels={"CAC": "CAC ($)", "network_display": "Channel"},
            text=cac_df["CAC"].map(lambda v: fmt_currency(v, 2)),
        )
        fig_cac_bar.update_traces(
            textposition="outside",
            hovertemplate="%{y}: $%{x:,.2f}<extra></extra>",
        )
        fig_cac_bar.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_cac_bar, use_container_width=True)

    with col_roas2:
        roas_df = ch2[["network_display", "ROAS"]].dropna().sort_values("ROAS")
        fig_roas_bar = px.bar(
            roas_df,
            x="ROAS",
            y="network_display",
            orientation="h",
            title="ROAS Comparison by Channel",
            color="network_display",
            color_discrete_map=NETWORK_COLORS,
            labels={"ROAS": "ROAS", "network_display": "Channel"},
            text=roas_df["ROAS"].map(fmt_x),
        )
        fig_roas_bar.update_traces(
            textposition="outside",
            hovertemplate="%{y}: %{x:.2f}x<extra></extra>",
        )
        fig_roas_bar.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig_roas_bar, use_container_width=True)

    st.divider()

    # Conversion Funnel by Channel
    st.markdown("#### Conversion Funnel by Channel")
    funnel_ch = ch2.sort_values("spend", ascending=False).reset_index(drop=True)
    ir = np.where(funnel_ch["clicks"] > 0, funnel_ch["installs"] / funnel_ch["clicks"], np.nan)
    st.dataframe(
        pd.DataFrame(
            {
                "Channel": funnel_ch["network_display"],
                "Impressions": funnel_ch["impressions"].map(fmt_num),
                "Clicks": funnel_ch["clicks"].map(fmt_num),
                "CTR": funnel_ch["CTR"].map(fmt_pct),
                "Installs": funnel_ch["installs"].map(fmt_num),
                "Install Rate": pd.Series(ir).map(fmt_pct),
                "Trials": funnel_ch["trials"].map(fmt_num),
                "Trial Rate (I2T)": funnel_ch["I2T"].map(fmt_pct),
                "Paid": funnel_ch["paid"].map(fmt_num),
                "T2P": funnel_ch["T2P"].map(fmt_pct),
                "I2P": funnel_ch["I2P"].map(fmt_pct),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Channel Performance History (period × channel)
    st.markdown("#### Channel Performance History")
    hist = df_period.groupby(["period", "network_display"], as_index=False).agg(
        spend=("spend", "sum"),
        installs=("installs", "sum"),
        trials=("trials", "sum"),
        paid=("paid_eff", "sum"),
        revenue=("revenue_eff", "sum"),
    )
    hist["CPI"] = np.where(hist["installs"] > 0, hist["spend"] / hist["installs"], np.nan)
    hist["T2P"] = np.where(hist["trials"] > 0, hist["paid"] / hist["trials"], np.nan)
    hist["CAC"] = np.where(hist["paid"] > 0, hist["spend"] / hist["paid"], np.nan)
    hist["ROAS"] = np.where(hist["spend"] > 0, hist["revenue"] / hist["spend"], np.nan)
    hist = hist.sort_values(["period", "spend"], ascending=[False, False])
    st.dataframe(
        pd.DataFrame(
            {
                "Period": hist["period"].dt.strftime("%Y-%m-%d"),
                "Channel": hist["network_display"],
                "Spend": hist["spend"].map(fmt_currency),
                "Installs": hist["installs"].map(fmt_num),
                "CPI": hist["CPI"].map(lambda v: fmt_currency(v, 2)),
                "Trials": hist["trials"].map(fmt_num),
                "Paid": hist["paid"].map(fmt_num),
                "T2P": hist["T2P"].map(fmt_pct),
                "CAC": hist["CAC"].map(lambda v: fmt_currency(v, 2)),
                "ROAS": hist["ROAS"].map(fmt_x),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ===========================================================================
# TAB 3 — Campaign Drill Down
# ===========================================================================
with tab3:
    st.markdown("#### Campaign Drill Down")
    networks_avail = sorted(df_filtered["network_display"].dropna().unique().tolist())
    selected_network = st.radio("Channel", networks_avail, horizontal=True)

    df_net = df_filtered[df_filtered["network_display"] == selected_network]

    st.markdown(f"**{selected_network}** — All Campaigns")

    camp_agg = df_net.groupby("campaign_name", as_index=False).agg(
        spend=("spend", "sum"),
        installs=("installs", "sum"),
        trials=("trials", "sum"),
        paid=("paid_eff", "sum"),
        revenue=("revenue_eff", "sum"),
    )
    camp_agg["CPI"] = np.where(
        camp_agg["installs"] > 0, camp_agg["spend"] / camp_agg["installs"], np.nan
    )
    camp_agg["CAC"] = np.where(camp_agg["paid"] > 0, camp_agg["spend"] / camp_agg["paid"], np.nan)
    camp_agg["ROAS"] = np.where(
        camp_agg["spend"] > 0, camp_agg["revenue"] / camp_agg["spend"], np.nan
    )
    camp_agg = camp_agg.sort_values("spend", ascending=False)

    # Derive campaign status from last-7-day spend
    today_ts = pd.Timestamp.today().normalize()
    recent_cutoff = today_ts - pd.Timedelta(days=7)
    recent_spend = df_net[df_net["date"] >= recent_cutoff].groupby("campaign_name")["spend"].sum()
    camp_agg["Status"] = camp_agg["campaign_name"].apply(
        lambda c: (
            "🟢 Active" if (c in recent_spend.index and recent_spend[c] > 0) else "🔴 Paused"
        )
    )

    st.dataframe(
        pd.DataFrame(
            {
                "Campaign": camp_agg["campaign_name"],
                "Status": camp_agg["Status"],
                "Spend": camp_agg["spend"].map(fmt_currency),
                "Installs": camp_agg["installs"].map(fmt_num),
                "CPI": camp_agg["CPI"].map(lambda v: fmt_currency(v, 2)),
                "Trials": camp_agg["trials"].map(fmt_num),
                "Subs": camp_agg["paid"].map(fmt_num),
                "CAC": camp_agg["CAC"].map(lambda v: fmt_currency(v, 2)),
                "ROAS": camp_agg["ROAS"].map(fmt_x),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Spend trend for selected network's top campaigns
    st.markdown("#### Spend Trend — Top Campaigns")
    top_campaigns = camp_agg.head(6)["campaign_name"].tolist()
    df_top_camp = df_period[
        (df_period["network_display"] == selected_network)
        & (df_period["campaign_name"].isin(top_campaigns))
    ]
    camp_trend = df_top_camp.groupby(["period", "campaign_name"], as_index=False).agg(
        spend=("spend", "sum"),
        installs=("installs", "sum"),
    )
    fig_camp_trend = px.line(
        camp_trend,
        x="period",
        y="spend",
        color="campaign_name",
        title=f"Spend Trend — {selected_network} Top Campaigns",
        labels={"spend": "Spend ($)", "period": "Date", "campaign_name": "Campaign"},
    )
    fig_camp_trend.update_traces(
        hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: $%{y:,.0f}<extra></extra>"
    )
    fig_camp_trend.update_layout(height=320, legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig_camp_trend, use_container_width=True)


# ===========================================================================
# TAB 4 — Funnel Activity
# ===========================================================================
with tab4:
    st.markdown("#### Full-Funnel Acquisition View")

    totals4 = {
        "spend": df_filtered["spend"].sum(),
        "impressions": df_filtered["impressions"].sum(),
        "clicks": df_filtered["clicks"].sum(),
        "installs": df_filtered["installs"].sum(),
        "trials": df_filtered["trials"].sum(),
        "paid": df_filtered["paid_eff"].sum(),
        "revenue": df_filtered["revenue_eff"].sum(),
    }

    # Top funnel KPIs — row 1
    funnel_cols = st.columns(7)
    ctr4 = totals4["clicks"] / totals4["impressions"] if totals4["impressions"] > 0 else np.nan
    i2t4 = totals4["trials"] / totals4["installs"] if totals4["installs"] > 0 else np.nan
    t2p4 = totals4["paid"] / totals4["trials"] if totals4["trials"] > 0 else np.nan
    for col, (label, val) in zip(
        funnel_cols,
        [
            ("Impressions", fmt_num(totals4["impressions"])),
            ("Clicks", fmt_num(totals4["clicks"])),
            ("Installs", fmt_num(totals4["installs"])),
            ("CTR", fmt_pct(ctr4)),
            ("Trials", fmt_num(totals4["trials"])),
            ("I2T", fmt_pct(i2t4)),
            ("Paid", fmt_num(totals4["paid"])),
        ],
    ):
        col.metric(label, val)

    # Row 2: summary KPIs
    kpi_cols2 = st.columns(4)
    roas4 = totals4["revenue"] / totals4["spend"] if totals4["spend"] > 0 else np.nan
    cac4 = totals4["spend"] / totals4["paid"] if totals4["paid"] > 0 else np.nan
    for col, (label, val) in zip(
        kpi_cols2,
        [
            ("T2P", fmt_pct(t2p4)),
            ("Revenue", fmt_currency(totals4["revenue"])),
            ("CAC", fmt_currency(cac4, 2)),
            ("ROAS", fmt_x(roas4)),
        ],
    ):
        col.metric(label, val)

    st.divider()

    # Funnel visualization  |  Attribution Source Mix
    col_funnel, col_attrib = st.columns([3, 2])

    with col_funnel:
        fig_funnel = go.Figure(
            go.Funnel(
                y=["Impressions", "Clicks", "Installs", "Trials", "Paid"],
                x=[
                    totals4["impressions"],
                    totals4["clicks"],
                    totals4["installs"],
                    totals4["trials"],
                    totals4["paid"],
                ],
                textinfo="value+percent initial",
                hovertemplate="%{label}: %{value:,.0f}<extra></extra>",
                marker=dict(color=["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]),
            )
        )
        fig_funnel.update_layout(title="Conversion Funnel — All Channels", height=380)
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col_attrib:
        attrib = (
            df_filtered.groupby("network_display", as_index=False)
            .agg(installs=("installs", "sum"))
            .query("installs > 0")
            .sort_values("installs", ascending=False)
        )
        fig_attrib = px.pie(
            attrib,
            values="installs",
            names="network_display",
            title="Attribution Source Mix (Installs)",
            hole=0.45,
            color="network_display",
            color_discrete_map=NETWORK_COLORS,
        )
        fig_attrib.update_traces(
            textposition="outside",
            textinfo="label+percent",
            hovertemplate="%{label}<br>Installs: %{value:,.0f}<br>Share: %{percent}<extra></extra>",
        )
        fig_attrib.update_layout(showlegend=False, height=380, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig_attrib, use_container_width=True)

    st.divider()

    # Funnel Rates by Channel table
    st.markdown("#### Funnel Rates by Channel")
    funnel_ch4 = aggregate_metrics(df_filtered, ["network_display"]).sort_values(
        "spend", ascending=False
    )
    funnel_ch4 = funnel_ch4.reset_index(drop=True)
    ir4 = np.where(funnel_ch4["clicks"] > 0, funnel_ch4["installs"] / funnel_ch4["clicks"], np.nan)
    st.dataframe(
        pd.DataFrame(
            {
                "Channel": funnel_ch4["network_display"],
                "Spend": funnel_ch4["spend"].map(fmt_currency),
                "Impressions": funnel_ch4["impressions"].map(fmt_num),
                "Clicks": funnel_ch4["clicks"].map(fmt_num),
                "CTR": funnel_ch4["CTR"].map(fmt_pct),
                "Installs": funnel_ch4["installs"].map(fmt_num),
                "Install Rate": pd.Series(ir4).map(fmt_pct),
                "Trials": funnel_ch4["trials"].map(fmt_num),
                "I2T": funnel_ch4["I2T"].map(fmt_pct),
                "Paid": funnel_ch4["paid"].map(fmt_num),
                "T2P": funnel_ch4["T2P"].map(fmt_pct),
                "ROAS": funnel_ch4["ROAS"].map(fmt_x),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Spend area trend by channel
    st.markdown("#### Spend Trend by Channel")
    spend_trend4 = df_period.groupby(["period", "network_display"], as_index=False).agg(
        spend=("spend", "sum"), installs=("installs", "sum")
    )
    fig_area = px.area(
        spend_trend4,
        x="period",
        y="spend",
        color="network_display",
        title="Spend Over Time by Channel",
        color_discrete_map=NETWORK_COLORS,
        labels={"spend": "Spend ($)", "period": "Date", "network_display": "Channel"},
    )
    fig_area.update_traces(
        hovertemplate="%{x|%Y-%m-%d}<br>%{fullData.name}: $%{y:,.0f}<extra></extra>"
    )
    fig_area.update_layout(height=320, legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_area, use_container_width=True)
