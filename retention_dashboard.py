"""
Customer Engagement & Product Utilization Analytics for Retention Strategy
----------------------------------------------------------------------------
Streamlit dashboard for European Bank customer churn data.
Soft pastel, non-neon institutional interface with executive metrics and cohort presets.

Run with:
    streamlit run retention_dashboard.py
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Customer Retention Analytics",
    page_icon="🏦",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FILE = BASE_DIR / "European_Bank_-_European_Bank.csv"
REQUIRED_COLUMNS = {
    "CustomerId", "Surname", "CreditScore", "Geography", "Gender", "Age",
    "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
    "EstimatedSalary", "Exited"
}

# ---------------------------------------------------------------------------
# Pastel Color Tokens (Zero Neon)
# ---------------------------------------------------------------------------
PASTEL_MINT = "#A7D7C5"        # Active / Retained / Low Risk
PASTEL_ROSE = "#F8B4B4"        # Inactive / Churned / High Risk
PASTEL_PEACH = "#FED7AA"       # Moderate / Warning
PASTEL_PERIWINKLE = "#C7D2FE"  # Multi-Product / Neutral
PASTEL_SKY = "#BAE6FD"         # Good engagement
PASTEL_SLATE = "#64748B"       # Subtle structural

PASTEL_GRADIENT = ["#A7D7C5", "#FED7AA", "#F8B4B4"]  # Gentle mint -> peach -> rose

PASTEL_MAP = {
    "Active": PASTEL_MINT,
    "Inactive": PASTEL_ROSE,
    "1 product": PASTEL_ROSE,
    "2+ products": PASTEL_MINT,
}

# ---------------------------------------------------------------------------
# Custom CSS for Soft Pastel Executive UI
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* Base typography and smooth layout */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1E293B;
}

/* Elevated KPI Cards */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.02), 0 4px 12px rgba(15, 23, 42, 0.01);
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.kpi-card:hover {
    border-color: #CBD5E1;
    transform: translateY(-1px);
}
.kpi-eyebrow {
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-bottom: 4px;
}
.kpi-value {
    font-size: 1.65rem;
    font-weight: 700;
    color: #0F172A;
    line-height: 1.2;
}
.kpi-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 9999px;
    margin-top: 8px;
}
.badge-mint {
    background-color: #E8F5E9;
    color: #22543D;
    border: 1px solid #A7D7C5;
}
.badge-rose {
    background-color: #FEE2E2;
    color: #9B1C1C;
    border: 1px solid #F8B4B4;
}
.badge-peach {
    background-color: #FEF3C7;
    color: #92400E;
    border: 1px solid #FED7AA;
}
.badge-periwinkle {
    background-color: #EEF2FF;
    color: #3730A3;
    border: 1px solid #C7D2FE;
}
.badge-slate {
    background-color: #F1F5F9;
    color: #475569;
    border: 1px solid #E2E8F0;
}

/* Sidebar Section Headers */
.sidebar-section-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94A3B8;
    margin-top: 18px;
    margin-bottom: 8px;
}

/* Dossier Card in Tab 4 */
.dossier-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.02);
}
</style>
""", unsafe_allow_html=True)


def apply_chart_theme(fig, height=360):
    """Standardizes Plotly charts to soft pastel styling with gentle gridlines."""
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color="#334155", size=12),
        margin=dict(l=20, r=20, t=45, b=20),
        height=height,
        xaxis=dict(showgrid=False, linecolor="#E2E8F0", tickfont=dict(size=11, color="#64748B")),
        yaxis=dict(gridcolor="rgba(226, 232, 240, 0.7)", linecolor="#E2E8F0", tickfont=dict(size=11, color="#64748B")),
        hoverlabel=dict(bgcolor="#0F172A", font_color="#F8FAFC", font_size=12, bordercolor="rgba(0,0,0,0)"),
    )
    fig.update_traces(marker_line_width=1, marker_line_color="rgba(0,0,0,0.08)")
    return fig


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required column(s): {', '.join(sorted(missing))}")
    df["Segment"] = df.apply(classify_segment, axis=1)
    df["RSI"] = df["IsActiveMember"] * 0.5 + (df["NumOfProducts"].clip(upper=2) / 2) * 0.5
    return df


def classify_segment(row):
    if row["IsActiveMember"] == 1 and row["NumOfProducts"] >= 2:
        return "Active Engaged"
    elif row["IsActiveMember"] == 0:
        return "Inactive High-Balance" if row["Balance"] >= 100000 else "Inactive Disengaged"
    elif row["IsActiveMember"] == 1 and row["NumOfProducts"] == 1:
        return "Active Low-Product"
    return "Other"


st.sidebar.markdown("<h2 style='font-size: 1.25rem; font-weight: 700; color: #0F172A; margin-bottom: 2px;'>Retention Analytics</h2>", unsafe_allow_html=True)
st.sidebar.caption("European Bank Portfolio Strategy")
uploaded = st.sidebar.file_uploader("Upload customer CSV", type="csv")

try:
    df_raw = load_data(uploaded if uploaded is not None else DEFAULT_FILE)
except FileNotFoundError:
    st.error(
        f"Couldn't find `{DEFAULT_FILE.name}` at `{DEFAULT_FILE.parent}`. "
        "Upload the dataset using the sidebar to continue."
    )
    st.stop()
except ValueError as e:
    st.error(str(e))
    st.stop()

# ---------------------------------------------------------------------------
# 1-Click Interactive Cohort Presets (Solves the "Static" feel)
# ---------------------------------------------------------------------------
st.sidebar.markdown("<div class='sidebar-section-title'>⚡ Quick Cohort Presets</div>", unsafe_allow_html=True)

# Preset state management
if "preset_geo" not in st.session_state:
    st.session_state["preset_geo"] = sorted(df_raw["Geography"].unique())
if "preset_gender" not in st.session_state:
    st.session_state["preset_gender"] = sorted(df_raw["Gender"].unique())
if "preset_eng" not in st.session_state:
    st.session_state["preset_eng"] = ["Active", "Inactive"]
if "preset_prod" not in st.session_state:
    st.session_state["preset_prod"] = (int(df_raw["NumOfProducts"].min()), int(df_raw["NumOfProducts"].max()))
if "preset_bal" not in st.session_state:
    st.session_state["preset_bal"] = (float(df_raw["Balance"].min()), float(df_raw["Balance"].max()))

pcol1, pcol2 = st.sidebar.columns(2)
if pcol1.button("All Customers", use_container_width=True, help="Reset all filters"):
    st.session_state["preset_geo"] = sorted(df_raw["Geography"].unique())
    st.session_state["preset_gender"] = sorted(df_raw["Gender"].unique())
    st.session_state["preset_eng"] = ["Active", "Inactive"]
    st.session_state["preset_prod"] = (int(df_raw["NumOfProducts"].min()), int(df_raw["NumOfProducts"].max()))
    st.session_state["preset_bal"] = (float(df_raw["Balance"].min()), float(df_raw["Balance"].max()))
    st.rerun()

if pcol2.button("At-Risk Wealth", use_container_width=True, help="Inactive members in top balance quartile"):
    st.session_state["preset_eng"] = ["Inactive"]
    q75 = float(df_raw["Balance"].quantile(0.75))
    st.session_state["preset_bal"] = (q75, float(df_raw["Balance"].max()))
    st.session_state["preset_prod"] = (int(df_raw["NumOfProducts"].min()), int(df_raw["NumOfProducts"].max()))
    st.rerun()

pcol3, pcol4 = st.sidebar.columns(2)
if pcol3.button("Sticky Champions", use_container_width=True, help="Active members with exactly 2 products"):
    st.session_state["preset_eng"] = ["Active"]
    st.session_state["preset_prod"] = (2, 2)
    st.session_state["preset_bal"] = (float(df_raw["Balance"].min()), float(df_raw["Balance"].max()))
    st.rerun()

if pcol4.button("Over-Sold (3+)", use_container_width=True, help="Customers with 3 or 4 products (high churn cohort)"):
    st.session_state["preset_prod"] = (3, int(df_raw["NumOfProducts"].max()))
    st.session_state["preset_eng"] = ["Active", "Inactive"]
    st.session_state["preset_bal"] = (float(df_raw["Balance"].min()), float(df_raw["Balance"].max()))
    st.rerun()

# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------
st.sidebar.markdown("<div class='sidebar-section-title'>Filters & Segments</div>", unsafe_allow_html=True)

geo_sel = st.sidebar.multiselect(
    "Geography", sorted(df_raw["Geography"].unique()), default=st.session_state["preset_geo"]
)
gender_sel = st.sidebar.multiselect(
    "Gender", sorted(df_raw["Gender"].unique()), default=st.session_state["preset_gender"]
)
engagement_sel = st.sidebar.multiselect(
    "Engagement status",
    ["Active", "Inactive"],
    default=st.session_state["preset_eng"],
)

def safe_slider_range(label, series, default_val=None, is_float=False):
    s_min = float(series.min()) if is_float else int(series.min())
    s_max = float(series.max()) if is_float else int(series.max())
    if s_min == s_max:
        s_max = s_min + (1.0 if is_float else 1)
    if default_val is not None:
        low = max(s_min, min(default_val[0], s_max))
        high = max(s_min, min(default_val[1], s_max))
        if low > high:
            low, high = s_min, s_max
        initial = (float(low), float(high)) if is_float else (int(low), int(high))
    else:
        initial = (s_min, s_max)
    return st.sidebar.slider(label, s_min, s_max, initial)


product_range = safe_slider_range("Product count range", df_raw["NumOfProducts"], default_val=st.session_state["preset_prod"])
balance_range = safe_slider_range("Balance range (€)", df_raw["Balance"], default_val=st.session_state["preset_bal"], is_float=True)
salary_range = safe_slider_range("Estimated salary range (€)", df_raw["EstimatedSalary"], is_float=True)
age_range = safe_slider_range("Age range", df_raw["Age"])

active_flags = []
if "Active" in engagement_sel:
    active_flags.append(1)
if "Inactive" in engagement_sel:
    active_flags.append(0)

df = df_raw[
    df_raw["Geography"].isin(geo_sel)
    & df_raw["Gender"].isin(gender_sel)
    & df_raw["IsActiveMember"].isin(active_flags)
    & df_raw["NumOfProducts"].between(*product_range)
    & df_raw["Balance"].between(*balance_range)
    & df_raw["EstimatedSalary"].between(*salary_range)
    & df_raw["Age"].between(*age_range)
].copy()

if df.empty:
    st.warning("No customers match the current filters. Adjust or reset filters in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Header & KPI Strip
# ---------------------------------------------------------------------------
st.markdown("<h1 style='font-size: 1.85rem; font-weight: 800; color: #0F172A; margin-bottom: 2px;'>Customer Engagement & Retention Analytics</h1>", unsafe_allow_html=True)
st.caption("A Behavioral & Relationship-Strength Framework — European Bank Customer Base")

overall_churn = df["Exited"].mean()
active_churn = df.loc[df.IsActiveMember == 1, "Exited"].mean() if (df.IsActiveMember == 1).any() else np.nan
inactive_churn = df.loc[df.IsActiveMember == 0, "Exited"].mean() if (df.IsActiveMember == 0).any() else np.nan
err_ratio = (
    inactive_churn / active_churn
    if (pd.notna(active_churn) and active_churn > 0 and pd.notna(inactive_churn))
    else np.nan
)
premium_at_risk = df[(df.Balance > df.Balance.quantile(0.75)) & (df.IsActiveMember == 0)]
premium_at_risk_rate = premium_at_risk["Exited"].mean() if len(premium_at_risk) else np.nan

# Styled Pastel KPI Strip
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-eyebrow">Cohort Size</div>
        <div class="kpi-value">{len(df):,}</div>
        <span class="kpi-badge badge-slate">Filtered Base</span>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-eyebrow">Overall Churn</div>
        <div class="kpi-value">{overall_churn:.1%}</div>
        <span class="kpi-badge badge-peach">Portfolio Benchmark</span>
    </div>
    """, unsafe_allow_html=True)

with k3:
    err_text = f"{err_ratio:.2f}×" if not np.isnan(err_ratio) else "N/A"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-eyebrow">Retention Multiplier</div>
        <div class="kpi-value">{err_text}</div>
        <span class="kpi-badge badge-periwinkle">Inactive ÷ Active Gap</span>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-eyebrow">At-Risk Wealth</div>
        <div class="kpi-value">{len(premium_at_risk):,}</div>
        <span class="kpi-badge badge-rose">Top 25% Bal + Inactive</span>
    </div>
    """, unsafe_allow_html=True)

with k5:
    risk_rate_text = f"{premium_at_risk_rate:.1%} " if not np.isnan(premium_at_risk_rate) else "N/A"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-eyebrow">At-Risk Churn Rate</div>
        <div class="kpi-value">{risk_rate_text}</div>
        <span class="kpi-badge badge-rose">High Severity Pool</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Engagement vs Churn", "Product Utilization", "High-Value Disengaged", "Relationship Strength Index"]
)

# ---------------------------------------------------------------------------
# TAB 1 — Engagement vs churn overview
# ---------------------------------------------------------------------------
with tab1:
    left, right = st.columns([1.2, 1])

    with left:
        seg = df.groupby("Segment")["Exited"].agg(["mean", "count"]).reset_index()
        seg.columns = ["Segment", "Churn Rate", "Customers"]
        seg = seg.sort_values("Churn Rate", ascending=False)
        fig1 = px.bar(
            seg, x="Segment", y="Churn Rate", color="Churn Rate",
            color_continuous_scale=PASTEL_GRADIENT,
            text=seg["Churn Rate"].map(lambda x: f"{x:.1%}"),
            title="Churn Rate by Engagement Segment",
        )
        fig1.update_layout(yaxis_tickformat=".0%", showlegend=False, coloraxis_showscale=False)
        fig1 = apply_chart_theme(fig1)
        st.plotly_chart(fig1, use_container_width=True)

    with right:
        act = df.groupby("IsActiveMember")["Exited"].mean().rename({0: "Inactive", 1: "Active"})
        status_colors = {"Inactive": PASTEL_ROSE, "Active": PASTEL_MINT}
        fig2 = go.Figure(go.Bar(
            x=act.index,
            y=act.values,
            marker_color=[status_colors.get(str(k), PASTEL_MINT) for k in act.index],
            text=[f"{v:.1%}" for v in act.values],
            textposition="outside",
        ))
        fig2.update_layout(title="Active vs Inactive Churn Rate", yaxis_tickformat=".0%")
        fig2 = apply_chart_theme(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Geography × Engagement")
    geo_eng = df.groupby(["Geography", "IsActiveMember"])["Exited"].mean().reset_index()
    geo_eng["IsActiveMember"] = geo_eng["IsActiveMember"].map({0: "Inactive", 1: "Active"})
    fig3 = px.bar(
        geo_eng, x="Geography", y="Exited", color="IsActiveMember", barmode="group",
        labels={"Exited": "Churn Rate", "IsActiveMember": "Status"},
        color_discrete_map={"Active": PASTEL_MINT, "Inactive": PASTEL_ROSE}
    )
    fig3.update_layout(yaxis_tickformat=".0%")
    fig3 = apply_chart_theme(fig3)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Age vs Churn")
    df["AgeBin"] = pd.cut(df["Age"], [0, 30, 40, 50, 60, 120], labels=["18-30", "31-40", "41-50", "51-60", "60+"], include_lowest=True)
    age_churn = df.groupby("AgeBin", observed=True)["Exited"].mean().reset_index()
    fig4 = px.line(
        age_churn, x="AgeBin", y="Exited", markers=True,
        labels={"Exited": "Churn Rate", "AgeBin": "Age group"}
    )
    fig4.update_traces(
        line_color="#6366F1", line_width=2.5,
        marker=dict(size=7, color="#4338CA"),
        fill="tozeroy", fillcolor="rgba(199, 210, 254, 0.25)"
    )
    fig4.update_layout(yaxis_tickformat=".0%")
    fig4 = apply_chart_theme(fig4)
    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 2 — Product utilization
# ---------------------------------------------------------------------------
with tab2:
    left, right = st.columns(2)
    with left:
        prod = df.groupby("NumOfProducts")["Exited"].agg(["mean", "count"]).reset_index()
        prod.columns = ["NumOfProducts", "Churn Rate", "Customers"]
        prod["NumOfProducts"] = prod["NumOfProducts"].astype(str)
        fig5 = px.bar(
            prod, x="NumOfProducts", y="Churn Rate", text=prod["Churn Rate"].map(lambda x: f"{x:.1%}"),
            title="Churn Rate by Number of Products", color="Churn Rate",
            color_continuous_scale=PASTEL_GRADIENT
        )
        fig5.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False, xaxis=dict(type="category"))
        fig5 = apply_chart_theme(fig5)
        st.plotly_chart(fig5, use_container_width=True)

    with right:
        df["MultiProduct"] = np.where(df["NumOfProducts"] >= 2, "2+ products", "1 product")
        mp = df.groupby("MultiProduct")["Exited"].mean().reset_index()
        fig6 = px.bar(
            mp, x="MultiProduct", y="Exited", text=mp["Exited"].map(lambda x: f"{x:.1%}"),
            title="Single vs Multi-Product Retention", color="MultiProduct",
            color_discrete_map={"1 product": PASTEL_ROSE, "2+ products": PASTEL_MINT}
        )
        fig6.update_layout(yaxis_tickformat=".0%", showlegend=False)
        fig6 = apply_chart_theme(fig6)
        st.plotly_chart(fig6, use_container_width=True)

    st.subheader("Product depth vs churn, by engagement status")
    pe = df.groupby(["NumOfProducts", "IsActiveMember"])["Exited"].mean().reset_index()
    pe["NumOfProducts"] = pe["NumOfProducts"].astype(str)
    pe["IsActiveMember"] = pe["IsActiveMember"].map({0: "Inactive", 1: "Active"})
    fig7 = px.bar(
        pe, x="NumOfProducts", y="Exited", color="IsActiveMember", barmode="group",
        labels={"Exited": "Churn Rate", "NumOfProducts": "Number of Products", "IsActiveMember": "Status"},
        color_discrete_map={"Active": PASTEL_MINT, "Inactive": PASTEL_ROSE}
    )
    fig7.update_layout(yaxis_tickformat=".0%", xaxis=dict(type="category"))
    fig7 = apply_chart_theme(fig7)
    st.plotly_chart(fig7, use_container_width=True)

    st.info(
        "💡 **Key Portfolio Finding:** Churn falls sharply to its lowest point (5.6%) at exactly 2 products, "
        "then surges past 85% at 3–4 products. Product depth beyond 2 is a strong warning indicator, not a cross-sell victory."
    )

# ---------------------------------------------------------------------------
# TAB 3 — High-value disengaged customer detector
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("High-Value Disengaged Customer Detector")
    st.caption("Customers who look financially affluent but show zero active engagement — the bank's greatest hidden revenue risk.")

    b_thresh_default = float(df["Balance"].quantile(0.75))
    s_thresh_default = float(df["EstimatedSalary"].quantile(0.75))

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        b_thresh = st.number_input("Balance threshold (€, top-value cutoff)", value=round(b_thresh_default, 0), step=1000.0)
    with fcol2:
        s_thresh = st.number_input("Salary threshold (€, top-value cutoff)", value=round(s_thresh_default, 0), step=1000.0)
    with fcol3:
        require_inactive = st.checkbox("Require inactive status", value=True)

    if require_inactive and "Inactive" not in engagement_sel:
        st.warning("Sidebar filter currently excludes Inactive customers. Uncheck 'Require inactive status' or re-enable Inactive in the sidebar.")

    hv = df[(df["Balance"] >= b_thresh) | (df["EstimatedSalary"] >= s_thresh)]
    if require_inactive:
        hv = hv[hv["IsActiveMember"] == 0]

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-eyebrow">Identified High-Value Risk</div>
            <div class="kpi-value">{len(hv):,}</div>
            <span class="kpi-badge badge-rose">Flagged Customers</span>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        hv_rate = f"{hv['Exited'].mean():.1%}" if len(hv) else "N/A"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-eyebrow">Cohort Churn Rate</div>
            <div class="kpi-value">{hv_rate}</div>
            <span class="kpi-badge badge-rose">Substantial Attrition</span>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        hv_share = f"{len(hv) / len(df):.1%}" if len(df) else "N/A"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-eyebrow">Share of Filtered Base</div>
            <div class="kpi-value">{hv_share}</div>
            <span class="kpi-badge badge-slate">Segment Concentration</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    st.dataframe(
        hv[["CustomerId", "Surname", "Geography", "Age", "Balance", "EstimatedSalary",
            "NumOfProducts", "IsActiveMember", "Exited"]]
        .sort_values("Balance", ascending=False)
        .rename(columns={"IsActiveMember": "Active", "Exited": "Churned"}),
        column_config={
            "Balance": st.column_config.NumberColumn(format="€%.2f"),
            "EstimatedSalary": st.column_config.NumberColumn(format="€%.2f"),
        },
        use_container_width=True,
        height=380,
    )

    csv = hv.to_csv(index=False).encode("utf-8")
    st.download_button("Download High-Value Target Queue (CSV)", csv, "high_value_disengaged_queue.csv", "text/csv")

# ---------------------------------------------------------------------------
# TAB 4 — Retention strength scoring
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Relationship Strength Index (RSI)")
    st.caption("Composite score = 0.5 × active status + 0.5 × product depth (capped at 2 products).")

    rsi_labels = {
        0.25: "0.25 (Very Low — Inactive, 1 Prod)",
        0.50: "0.50 (Moderate — Inactive, 2+ Prod)",
        0.75: "0.75 (Good — Active, 1 Prod)",
        1.00: "1.00 (High — Active, 2+ Prod)",
    }
    df["RSI Band"] = df["RSI"].round(2).map(rsi_labels).fillna(df["RSI"].round(2).astype(str))

    rsi_summary = df.groupby("RSI Band", observed=True)["Exited"].agg(["mean", "count"]).reset_index()
    rsi_summary.columns = ["RSI band", "Churn Rate", "Customers"]
    rsi_order = list(rsi_labels.values())
    rsi_summary["sort_order"] = rsi_summary["RSI band"].map(lambda x: rsi_order.index(x) if x in rsi_order else 99)
    rsi_summary = rsi_summary.sort_values("sort_order").drop(columns=["sort_order"])

    fig8 = px.bar(
        rsi_summary, x="RSI band", y="Churn Rate", text=rsi_summary["Churn Rate"].map(lambda x: f"{x:.1%}"),
        title="Churn Rate by Relationship Strength Tier", color="Churn Rate",
        color_continuous_scale=PASTEL_GRADIENT
    )
    fig8.update_layout(yaxis_tickformat=".0%", coloraxis_showscale=False)
    fig8 = apply_chart_theme(fig8)
    st.plotly_chart(fig8, use_container_width=True)

    st.subheader("Sticky vs Risky Behavioral Profiles")
    sticky = df[(df.IsActiveMember == 1) & (df.NumOfProducts == 2)]
    risky = df[df.NumOfProducts >= 3]
    p1, p2 = st.columns(2)
    with p1:
        if len(sticky) > 0:
            st.markdown(f"""
            <div style="background-color: #F0FDF4; border: 1px solid #A7D7C5; border-radius: 12px; padding: 16px;">
                <div style="font-weight: 700; color: #166534; font-size: 0.95rem;">🛡️ Sticky Profile (Active + 2 Products)</div>
                <div style="font-size: 0.85rem; color: #14532D; margin-top: 6px;">
                    <strong>{len(sticky):,}</strong> customers &nbsp;|&nbsp; Churn Rate: <strong>{sticky['Exited'].mean():.1%}</strong>
                </div>
                <div style="font-size: 0.78rem; color: #4ADE80; margin-top: 4px;">Optimal loyalty benchmark — highest retention resilience.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No customers match the Sticky Profile under current filters.")

    with p2:
        if len(risky) > 0:
            st.markdown(f"""
            <div style="background-color: #FFF1F2; border: 1px solid #F8B4B4; border-radius: 12px; padding: 16px;">
                <div style="font-weight: 700; color: #9F1239; font-size: 0.95rem;">⚠️ Risky Profile (3+ Products)</div>
                <div style="font-size: 0.85rem; color: #881337; margin-top: 6px;">
                    <strong>{len(risky):,}</strong> customers &nbsp;|&nbsp; Churn Rate: <strong>{risky['Exited'].mean():.1%}</strong>
                </div>
                <div style="font-size: 0.78rem; color: #FB7185; margin-top: 4px;">Extreme attrition risk — immediate outreach required.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No customers match the Risky Profile under current filters.")

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.subheader("Customer Health Dossier")
    cust_id = st.selectbox("Select Customer by ID", df["CustomerId"].sort_values().unique())
    row = df[df["CustomerId"] == cust_id].iloc[0]

    # Formatted Executive Profile Card
    status_pill = f'<span class="kpi-badge badge-rose">Churned (Exited)</span>' if row['Exited'] == 1 else f'<span class="kpi-badge badge-mint">Retained</span>'
    active_pill = f'<span class="kpi-badge badge-mint">Active Member</span>' if row['IsActiveMember'] == 1 else f'<span class="kpi-badge badge-rose">Inactive</span>'

    st.markdown(f"""
    <div class="dossier-card">
        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #F1F5F9; pb-3; margin-bottom: 12px;">
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; color: #0F172A;">{row['Surname']} &nbsp;<span style="font-size: 0.85rem; color: #64748B;">(ID: {int(row['CustomerId'])})</span></h3>
                <div style="font-size: 0.80rem; color: #64748B; margin-top: 2px;">{row['Geography']} &bull; {row['Gender']} &bull; Age {int(row['Age'])}</div>
            </div>
            <div style="display: flex; gap: 8px;">
                {status_pill}
                {active_pill}
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-top: 10px;">
            <div>
                <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase;">Balance</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">€{row['Balance']:,.2f}</div>
            </div>
            <div>
                <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase;">Estimated Salary</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">€{row['EstimatedSalary']:,.2f}</div>
            </div>
            <div>
                <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase;">Products Held</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">{int(row['NumOfProducts'])}</div>
            </div>
            <div>
                <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase;">RSI Health Score</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A;">{row['RSI']:.2f} / 1.00</div>
            </div>
            <div>
                <div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase;">Assigned Segment</div>
                <div style="font-size: 0.95rem; font-weight: 600; color: #334155;">{row['Segment']}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
