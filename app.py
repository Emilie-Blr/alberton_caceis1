"""
CACEIS Human Capital Intelligence Cockpit
Streamlit interface for the final evolved prototype.

Launch:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_pipeline import (
    load_pipeline,
    run_scenario,
    build_segment_summary,
    build_portfolio,
    build_recommendations,
    build_country_kpis,
    build_department_summary,
    FINANCIAL_UNIT_LABEL,
    PERFORMANCE_WEIGHT,
    POTENTIAL_WEIGHT,
    ENGAGEMENT_WEIGHT,
    FINANCIAL_WEIGHT,
    CRITICALITY_WEIGHT,
    RISK_PENALTY_WEIGHT,
)


st.set_page_config(
    page_title="CACEIS Human Capital Intelligence Cockpit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        font-size: 1.05rem;
        color: #566573;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        padding: 1rem 1.1rem;
        border-radius: 1rem;
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03);
        min-height: 115px;
    }
    .metric-label {
        color: #6B7280;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .metric-value {
        color: #111827;
        font-size: 1.75rem;
        font-weight: 800;
        margin-top: 0.25rem;
    }
    .metric-help {
        color: #6B7280;
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    .insight-box {
        padding: 1rem;
        border-radius: 0.9rem;
        border-left: 5px solid #1F4E79;
        background: #F8FAFC;
        margin-bottom: 0.8rem;
    }
    .warning-box {
        padding: 0.9rem;
        border-radius: 0.8rem;
        border-left: 5px solid #B7791F;
        background: #FFFBEB;
        margin-bottom: 0.8rem;
    }
    .capability-pill {
        display: inline-block;
        padding: 0.4rem 0.7rem;
        margin: 0.15rem;
        border-radius: 999px;
        background: #EEF2FF;
        color: #1E3A8A;
        font-weight: 600;
        font-size: 0.86rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def candidate_data_dirs() -> list[Path]:
    dirs = [Path.cwd(), Path(__file__).resolve().parent, Path("/mnt/data")]
    return list(dict.fromkeys([d for d in dirs if d.exists()]))


def html_metric(label: str, value: str, help_text: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_num(value, digits=1, suffix="") -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:,.{digits}f}{suffix}"


def pct_format(series: pd.Series) -> pd.Series:
    return series.apply(lambda x: "n/a" if pd.isna(x) else f"{x:.1%}")


def add_population_share_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "population_share" in out.columns:
        out["population_share"] = pct_format(pd.to_numeric(out["population_share"], errors="coerce"))
    return out


def line_chart(df: pd.DataFrame, x: str, y_cols: list[str], title: str, ylabel: str = ""):
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in y_cols:
        if col in df.columns:
            ax.plot(df[x], df[col], marker="o", label=col.replace("_", " ").title())
    ax.set_title(title)
    ax.set_xlabel(x.title())
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.25)
    st.pyplot(fig, use_container_width=True)


def portfolio_matrix(portfolio: pd.DataFrame, title: str = "Human Capital Portfolio Matrix — Value vs Risk"):
    if portfolio.empty:
        st.info("Portfolio data is not available.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    x = portfolio["strategic_value_score"]
    y = portfolio["organizational_risk_index"] * 100
    size_basis = portfolio["population_share"] if "population_share" in portfolio.columns else portfolio.get("population", pd.Series(1, index=portfolio.index))
    size = np.maximum(pd.to_numeric(size_basis, errors="coerce").fillna(0.02), 0.02) * 3000
    ax.scatter(x, y, s=size, alpha=0.65)
    for _, row in portfolio.iterrows():
        ax.text(row["strategic_value_score"], row["organizational_risk_index"] * 100, row["profile_segment"], fontsize=8)
    ax.axvline(x.median(), linestyle="--", alpha=0.5)
    ax.axhline(y.median(), linestyle="--", alpha=0.5)
    ax.set_xlabel("Strategic Value Score")
    ax.set_ylabel("Organizational Risk Score")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    st.pyplot(fig, use_container_width=True)


def radar_chart(row: pd.Series, title: str, benchmark: pd.Series | None = None):
    labels = ["Performance", "Potential", "Engagement", "Criticality", "Risk control"]
    value_cols = ["performance_index", "potential_index", "engagement_proxy_index", "criticality_index", "organizational_risk_index"]

    def extract_values(source: pd.Series):
        vals = []
        for col in value_cols:
            if col == "organizational_risk_index":
                v = 1 - source.get(col, np.nan)
            else:
                v = source.get(col, np.nan)
            vals.append(0 if pd.isna(v) else float(v))
        return vals

    values = extract_values(row)
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(5.5, 5.5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, label="Selected segment")
    ax.fill(angles, values, alpha=0.15)

    if benchmark is not None:
        bench = extract_values(benchmark)
        bench += bench[:1]
        ax.plot(angles, bench, linewidth=2, linestyle="--", label="Portfolio average")

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.10))
    st.pyplot(fig, use_container_width=True)


def score_distribution(df: pd.DataFrame, score_col: str = "risk_adjusted_hcv_score"):
    if df.empty or score_col not in df.columns:
        st.info("Score distribution is not available for the current filters.")
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df[score_col].dropna(), bins=20, alpha=0.75)
    ax.set_title("Distribution of Risk-adjusted HCV Score")
    ax.set_xlabel("Risk-adjusted HCV Score")
    ax.set_ylabel("Number of analytical records")
    ax.grid(axis="y", alpha=0.25)
    st.pyplot(fig, use_container_width=True)


def department_score_chart(department_summary: pd.DataFrame, min_population: int = 20):
    if department_summary.empty:
        st.info("No department reaches the minimum population threshold for the current filters.")
        return
    plot = department_summary[department_summary["population"] >= min_population].sort_values("avg_hcv_score").tail(15)
    if plot.empty:
        st.info("No department reaches the minimum population threshold for the current filters.")
        return
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(plot))))
    ax.barh(plot["department_view"], plot["avg_hcv_score"])
    ax.set_title(f"Average HCV Score by department — min. {min_population} people")
    ax.set_xlabel("Average Risk-adjusted HCV Score")
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig, use_container_width=True)


def barh_chart(df: pd.DataFrame, value_col: str, label_col: str, title: str, xlabel: str = ""):
    if df.empty or value_col not in df.columns:
        st.info("Chart data is not available.")
        return
    plot = df.sort_values(value_col)
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(plot))))
    ax.barh(plot[label_col], plot[value_col])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", alpha=0.25)
    st.pyplot(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------

st.sidebar.title("Cockpit settings")

available_dirs = candidate_data_dirs()
default_dir = available_dirs[0]
for d in available_dirs:
    if any(d.glob("*.xlsx")):
        default_dir = d
        break

data_dir_input = st.sidebar.text_input("Data folder", value=str(default_dir))
data_dir = Path(data_dir_input)

prefer_cache = st.sidebar.checkbox("Use cached CSVs when available", value=True)
fast_mode = st.sidebar.checkbox("FAST_MODE for large Excel files", value=True, help="Recommended for demos. It samples the largest absenteeism files during rebuild.")
rebuild_cache = st.sidebar.checkbox("Force rebuild from Excel", value=False)

st.sidebar.markdown("---")


@st.cache_data(show_spinner="Loading and preparing Human Capital Intelligence data...")
def cached_load(data_dir_str: str, prefer_cache: bool, fast_mode: bool, rebuild_cache: bool):
    return load_pipeline(Path(data_dir_str), prefer_cache=prefer_cache, fast_mode=fast_mode, rebuild_cache=rebuild_cache)

try:
    data = cached_load(str(data_dir), prefer_cache, fast_mode, rebuild_cache)
except Exception as exc:
    st.error("The cockpit could not load the data pipeline.")
    st.exception(exc)
    st.stop()

finance = data.get("finance", pd.DataFrame())
segment_summary = data.get("segment_summary", pd.DataFrame())
portfolio = data.get("portfolio", pd.DataFrame())
recommendations = data.get("recommendations", pd.DataFrame())
country_kpis = data.get("country_kpis", pd.DataFrame())
department_summary = data.get("department_summary", pd.DataFrame())
data_quality = data.get("data_quality", pd.DataFrame())
training_taxonomy = data.get("training_taxonomy", pd.DataFrame())
performance_quality = data.get("performance_quality", pd.DataFrame())
valuation = data.get("valuation", data.get("valuation_view", pd.DataFrame()))
source_mode = data.get("_source", pd.DataFrame({"mode": ["unknown"]}))["mode"].iloc[0]

# Filters
available_years = sorted(country_kpis["year"].dropna().astype(int).unique().tolist()) if not country_kpis.empty and "year" in country_kpis else []
if not valuation.empty and "year" in valuation.columns:
    available_years = sorted(pd.to_numeric(valuation["year"], errors="coerce").dropna().astype(int).unique().tolist())
year_filter = st.sidebar.multiselect("Years", available_years, default=available_years)
country_filter = st.sidebar.multiselect("Countries", ["France", "Luxembourg"], default=["France", "Luxembourg"])

# Department filter, only departments with at least 20 records under the country/year filters.
if not valuation.empty and "department_view" in valuation.columns:
    dept_base = valuation.copy()
    if year_filter and "year" in dept_base.columns:
        dept_base = dept_base[pd.to_numeric(dept_base["year"], errors="coerce").isin(year_filter)]
    if country_filter and "country_scope" in dept_base.columns:
        dept_base = dept_base[dept_base["country_scope"].isin(country_filter)]
    dept_counts = dept_base.groupby("department_view").size().sort_values(ascending=False)
    available_departments = dept_counts[dept_counts >= 20].index.tolist()
else:
    available_departments = []
department_filter = st.sidebar.multiselect("Departments", available_departments, default=available_departments, help="Only departments with at least 20 analytical records are available.")

# Apply filters dynamically when detailed valuation view exists.
if not valuation.empty:
    valuation_filtered = valuation.copy()
    if year_filter and "year" in valuation_filtered.columns:
        valuation_filtered = valuation_filtered[pd.to_numeric(valuation_filtered["year"], errors="coerce").isin(year_filter)]
    if country_filter and "country_scope" in valuation_filtered.columns:
        valuation_filtered = valuation_filtered[valuation_filtered["country_scope"].isin(country_filter)]
    if department_filter and "department_view" in valuation_filtered.columns:
        valuation_filtered = valuation_filtered[valuation_filtered["department_view"].isin(department_filter)]

    if not valuation_filtered.empty and "profile_segment" in valuation_filtered.columns:
        segment_summary_filtered = build_segment_summary(valuation_filtered)
        portfolio_filtered = build_portfolio(segment_summary_filtered)
        recommendations_filtered = build_recommendations(segment_summary_filtered)
        country_kpis_filtered = build_country_kpis(valuation_filtered)
        department_summary_filtered = build_department_summary(valuation_filtered, min_population=20)
    else:
        segment_summary_filtered = segment_summary.copy()
        portfolio_filtered = portfolio.copy()
        recommendations_filtered = recommendations.copy()
        country_kpis_filtered = country_kpis.copy()
        department_summary_filtered = department_summary.copy()
else:
    valuation_filtered = pd.DataFrame()
    segment_summary_filtered = segment_summary.copy()
    portfolio_filtered = portfolio.copy()
    recommendations_filtered = recommendations.copy()
    department_summary_filtered = department_summary.copy()
    if year_filter and not country_kpis.empty:
        country_kpis_filtered = country_kpis[country_kpis["year"].isin(year_filter) & country_kpis["country_scope"].isin(country_filter)].copy()
    else:
        country_kpis_filtered = country_kpis.copy()


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.markdown('<div class="main-title">CACEIS Human Capital Intelligence Cockpit</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-enhanced strategic workforce valuation, segmentation, scenario simulation and recommendation engine.</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <span class="capability-pill">ML segmentation</span>
    <span class="capability-pill">Recommendation engine</span>
    <span class="capability-pill">Scenario simulation</span>
    <span class="capability-pill">Copilot narratives</span>
    """,
    unsafe_allow_html=True,
)

st.caption(f"Data mode: `{source_mode}` | Financial unit label: `{FINANCIAL_UNIT_LABEL}`")


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Executive Overview",
    "2. Human Capital Portfolio",
    "3. Segment Intelligence",
    "4. Scenario Simulator",
    "5. Recommendations",
    "6. Governance & Explainability",
])


# -----------------------------------------------------------------------------
# Tab 1 - Executive Overview
# -----------------------------------------------------------------------------

with tab1:
    st.subheader("Executive Overview")
    st.markdown(
        """
        This cockpit translates HR, training, performance, absenteeism and finance data into a strategic view of human capital value creation.  
        It is designed for decision support, not individual employee assessment.
        """
    )

    latest_fin = pd.DataFrame()
    if not finance.empty:
        f_eu = finance[finance["scope"].eq("Europe")].sort_values("year")
        latest_fin = f_eu.tail(1)

    latest_score = country_kpis_filtered.sort_values("year").tail(1) if not country_kpis_filtered.empty else pd.DataFrame()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        html_metric("Human Capital ROI", format_num(latest_fin["human_capital_roi"].iloc[0], 2) if not latest_fin.empty else "n/a", "PNB / personnel costs")
    with c2:
        html_metric("Value per FTE", format_num(latest_fin["value_per_fte"].iloc[0], 1) if not latest_fin.empty else "n/a", FINANCIAL_UNIT_LABEL)
    with c3:
        html_metric("Avg. HCV Score", format_num(latest_score["avg_hcv_score"].mean(), 1, "/100") if not latest_score.empty else "n/a", "Risk-adjusted workforce value")
    with c4:
        html_metric("Segments", str(len(segment_summary_filtered)), "AI-generated workforce profiles")

    st.markdown("### Strategic value creation thesis")
    st.markdown(
        """
        <div class="insight-box">
        Human capital value is interpreted as a portfolio of value drivers: current contribution, future capability, engagement/resilience, strategic criticality and organizational risk. The model does not claim strict causal inference; it identifies economically plausible drivers associated with stronger workforce value creation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        if not finance.empty:
            f_eu = finance[finance["scope"].eq("Europe")].sort_values("year")
            line_chart(f_eu, "year", ["human_capital_roi"], "Human Capital ROI — Europe", "PNB / personnel costs")
    with c2:
        if not finance.empty:
            f_eu = finance[finance["scope"].eq("Europe")].sort_values("year")
            line_chart(f_eu, "year", ["value_per_fte"], "Value per FTE — Europe", FINANCIAL_UNIT_LABEL)

    st.markdown("### Score distribution and value/risk snapshot")
    c1, c2 = st.columns(2)
    with c1:
        score_distribution(valuation_filtered)
    with c2:
        portfolio_matrix(portfolio_filtered, title="Strategic Value vs Organizational Risk")

    st.markdown("### Department-level value signals")
    department_score_chart(department_summary_filtered, min_population=20)

    st.markdown("### Country-level workforce value signals")
    if not country_kpis_filtered.empty:
        display_cols = ["country_scope", "year", "population_share", "avg_hcv_score", "avg_performance", "avg_training_hours", "avg_risk_absence_days", "avg_criticality", "avg_risk_index"]
        display_df = add_population_share_display(country_kpis_filtered[[c for c in display_cols if c in country_kpis_filtered.columns]].round(3))
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Country-level KPIs are not available.")


# -----------------------------------------------------------------------------
# Tab 2 - Portfolio
# -----------------------------------------------------------------------------

with tab2:
    st.subheader("Human Capital Portfolio")
    st.markdown(
        """
        The portfolio view treats workforce segments as strategic human capital asset classes.  
        The objective is to support capital allocation decisions: where to invest, protect, develop or de-risk.
        """
    )

    portfolio_matrix(portfolio_filtered)

    st.markdown("### Portfolio asset classes")
    if not portfolio_filtered.empty:
        cols = ["profile_segment", "asset_class", "population_share", "strategic_value_score", "organizational_risk_index", "criticality_index", "avg_hcv_score"]
        display_df = add_population_share_display(portfolio_filtered[[c for c in cols if c in portfolio_filtered.columns]].round(3))
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Portfolio data is not available.")

    st.markdown("### Segment size")
    if not segment_summary_filtered.empty:
        barh_chart(segment_summary_filtered, "population_share", "profile_segment", "Population share by strategic segment", "Population share")


# -----------------------------------------------------------------------------
# Tab 3 - Segment Intelligence
# -----------------------------------------------------------------------------

with tab3:
    st.subheader("Segment Intelligence")
    st.markdown("Select a segment to view its explainability radar, top drivers, risk signals and recommendation narrative.")

    if segment_summary_filtered.empty:
        st.info("No segment data available.")
    else:
        selected_segment = st.selectbox("Select profile segment", segment_summary_filtered["profile_segment"].tolist())
        row = segment_summary_filtered[segment_summary_filtered["profile_segment"].eq(selected_segment)].iloc[0]
        rec_row = recommendations_filtered[recommendations_filtered["profile_segment"].eq(selected_segment)].iloc[0] if not recommendations_filtered.empty and selected_segment in recommendations_filtered["profile_segment"].values else None

        c1, c2, c3 = st.columns([1.1, 1, 1])
        with c1:
            radar_chart(row, selected_segment, benchmark=segment_summary_filtered.mean(numeric_only=True))
        with c2:
            html_metric("Population share", format_num(row.get("population_share", np.nan) * 100, 1, "%"), "Share of filtered population")
            html_metric("HCV Score", format_num(row["avg_hcv_score"], 1, "/100"), "Risk-adjusted score")
        with c3:
            html_metric("Criticality", format_num(row["criticality_index"] * 100, 1, "%"), "Internal strategic criticality")
            html_metric("Risk", format_num(row["organizational_risk_index"] * 100, 1, "%"), "Organizational risk index")

        st.markdown("### Copilot diagnosis")
        narrative = rec_row["copilot_narrative"] if rec_row is not None else "No narrative available."
        st.markdown(f'<div class="insight-box">{narrative}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Top drivers")
            st.write(rec_row["top_drivers"] if rec_row is not None else "n/a")
        with c2:
            st.markdown("#### Top risks")
            st.write(rec_row["top_risks"] if rec_row is not None else "n/a")

        st.markdown("#### Recommended actions")
        st.success(rec_row["recommended_actions"] if rec_row is not None else "No recommendation available.")

        st.markdown("### Segment recommendation table")
        if not recommendations_filtered.empty:
            display_cols = ["profile_segment", "population_share", "avg_hcv_score", "top_drivers", "top_risks", "recommended_actions"]
            display_df = add_population_share_display(recommendations_filtered[[c for c in display_cols if c in recommendations_filtered.columns]].round(2))
            st.dataframe(display_df, use_container_width=True, hide_index=True)


# -----------------------------------------------------------------------------
# Tab 4 - Scenario Simulator
# -----------------------------------------------------------------------------

with tab4:
    st.subheader("Scenario Simulator")
    st.markdown(
        """
        This module performs transparent sensitivity analysis. It does not forecast the future; it estimates directional impact under explicit business assumptions.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        training_uplift = st.slider("Training uplift", 0, 30, 10, step=5, format="%d%%")
    with c2:
        absence_reduction = st.slider("Risk absenteeism reduction", 0, 30, 10, step=5, format="%d%%")
    with c3:
        performance_uplift = st.slider("Performance uplift", 0.0, 0.5, 0.1, step=0.05)
    with c4:
        strategic_investment = st.checkbox("Target critical segments", value=True)

    scenario = run_scenario(segment_summary_filtered, training_uplift, absence_reduction, performance_uplift, strategic_investment)

    if not scenario.empty:
        st.markdown("### Scenario impact by segment")
        display_cols = ["profile_segment", "population_share", "avg_hcv_score", "scenario_hcv_score", "scenario_score_delta", "scenario_change_pct"]
        scenario_display = add_population_share_display(scenario[[c for c in display_cols if c in scenario.columns]].round(2))
        st.dataframe(scenario_display, use_container_width=True, hide_index=True)

        fig, ax = plt.subplots(figsize=(9, 4.5))
        plot = scenario.sort_values("scenario_score_delta")
        ax.barh(plot["profile_segment"], plot["scenario_score_delta"])
        ax.set_title("Simulated HCV score uplift by segment")
        ax.set_xlabel("Score delta")
        ax.grid(axis="x", alpha=0.25)
        st.pyplot(fig, use_container_width=True)

        best = scenario.sort_values("scenario_score_delta", ascending=False).iloc[0]
        st.markdown(
            f"""
            <div class="insight-box">
            Under the selected scenario, the strongest estimated uplift appears in <b>{best['profile_segment']}</b>, with a simulated score improvement of <b>{best['scenario_score_delta']:.1f}</b> points. This suggests that targeted investment should prioritize segments where capability renewal, risk reduction and strategic criticality intersect.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Scenario data is not available.")


# -----------------------------------------------------------------------------
# Tab 5 - Recommendations
# -----------------------------------------------------------------------------

with tab5:
    st.subheader("Recommendations")
    st.markdown(
        """
        This page consolidates the recommendation engine outputs into a COMEX-ready action view.  
        Recommendations are generated at segment level and adapt to the active filters.
        """
    )

    if recommendations_filtered.empty:
        st.info("No recommendations available for the current filters.")
    else:
        top_priority = recommendations_filtered.sort_values("avg_hcv_score").head(1)
        if not top_priority.empty:
            r = top_priority.iloc[0]
            st.markdown(
                f"""
                <div class="insight-box">
                <b>Priority segment:</b> {r['profile_segment']}<br>
                <b>Population share:</b> {r.get('population_share', 0):.1%}<br>
                <b>Recommended action:</b> {r['recommended_actions']}
                </div>
                """,
                unsafe_allow_html=True,
            )

        display_cols = ["profile_segment", "population_share", "avg_hcv_score", "top_drivers", "top_risks", "recommended_actions", "copilot_narrative"]
        display_df = add_population_share_display(recommendations_filtered[[c for c in display_cols if c in recommendations_filtered.columns]].round(2))
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.markdown("### Recommendation priority map")
        if not segment_summary_filtered.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            x = segment_summary_filtered["avg_hcv_score"]
            y = segment_summary_filtered["organizational_risk_index"] * 100
            size = np.maximum(segment_summary_filtered["population_share"], 0.02) * 3000
            ax.scatter(x, y, s=size, alpha=0.65)
            for _, rr in segment_summary_filtered.iterrows():
                ax.text(rr["avg_hcv_score"], rr["organizational_risk_index"] * 100, rr["profile_segment"], fontsize=8)
            ax.set_xlabel("Average Risk-adjusted HCV Score")
            ax.set_ylabel("Organizational Risk Score")
            ax.set_title("Recommendation priority map — value vs urgency")
            ax.grid(alpha=0.25)
            st.pyplot(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# Tab 6 - Governance & Explainability
# -----------------------------------------------------------------------------

with tab6:
    st.subheader("Governance & Explainability")

    st.markdown("### Functional AI capabilities demonstrated")
    st.markdown(
        """
        - **Segmentation:** unsupervised learning groups workforce profiles into strategic segments.  
        - **Recommendation:** rules-based decision engine recommends actions based on segment drivers and risks.  
        - **Scenario simulation:** sensitivity engine estimates directional impact of management levers.  
        - **Copilot narratives:** template-based narrative layer translates analytics into executive explanations.  
        """
    )

    st.markdown("### Score decomposition")
    weights = pd.DataFrame({
        "Dimension": ["Performance", "Potential", "Engagement", "Financial contribution", "Strategic criticality", "Risk penalty"],
        "Weight / role": [PERFORMANCE_WEIGHT, POTENTIAL_WEIGHT, ENGAGEMENT_WEIGHT, FINANCIAL_WEIGHT, CRITICALITY_WEIGHT, RISK_PENALTY_WEIGHT],
        "Interpretation": [
            "Current contribution proxy",
            "Future capability and employability proxy",
            "Availability and resilience proxy",
            "Annual financial context",
            "Internal criticality of capabilities",
            "Penalty applied for organizational risk",
        ]
    })
    st.dataframe(weights, use_container_width=True, hide_index=True)

    st.markdown("### Responsible use principles")
    st.markdown(
        """
        <div class="warning-box">
        <b>Governance guardrails:</b><br>
        1. Input data is already anonymized.<br>
        2. Anonymous IDs are used only as technical keys and are never displayed.<br>
        3. The cockpit does not rank employees individually.<br>
        4. Maternity, paternity and family-related absences are treated as protected/sensitive and excluded from punitive risk scoring.<br>
        5. The model supports strategic workforce planning, not disciplinary decisions.<br>
        6. Scenario results are sensitivity analyses, not causal forecasts.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Data quality overview")
    if not data_quality.empty:
        st.dataframe(data_quality, use_container_width=True, hide_index=True)

    if not performance_quality.empty:
        st.markdown("### Performance data quality")
        st.dataframe(performance_quality.round(3), use_container_width=True, hide_index=True)

    if not training_taxonomy.empty:
        st.markdown("### Training taxonomy")
        st.dataframe(training_taxonomy.sort_values(["year", "records"], ascending=[True, False]).round(2), use_container_width=True, hide_index=True)

    st.markdown("### Optional Local / Enterprise AI Copilot Enhancement")
    st.caption("The cockpit is fully functional without an external AI API. Because HR data is confidential, any generative AI extension should rely on a locally hosted model or a proprietary enterprise AI environment. Only aggregated segment-level data should be processed.")
    with st.expander("Optional copilot prompt template"):
        st.code(
            textwrap.dedent(
                """
                You are an executive HR analytics copilot running in a private enterprise environment. Rewrite the following aggregated segment diagnosis
                in a concise COMEX style. Do not mention individual employees. Do not infer causality.
                Use only the provided aggregated segment metrics and recommendations.
                """
            ),
            language="text",
        )
