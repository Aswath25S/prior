from __future__ import annotations

import os
import hashlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.benchmark import BenchmarkError, prepare_benchmark, recommended_features, run_tabpfn_benchmark
from src.data_loader import DataValidationError, load_csv
from src.data_profiler import profile_dataset
from src.poc_generator import generate_poc_markdown
from src.prioritization import prioritize, recommendation
from src.readiness import assess_readiness, suspected_leakage_columns
from src.schemas import BusinessContext, ReadinessStatus, SupportLevel
from src.use_case_discovery import discover_use_cases


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
SAMPLES = {
    "Manufacturing — equipment failure": ROOT / "data/samples/synthetic_manufacturing.csv",
    "Retail — product demand": ROOT / "data/samples/synthetic_retail.csv",
    "Manufacturing — outcome absent": ROOT / "data/samples/synthetic_manufacturing_unlabeled.csv",
}
SAMPLE_DESCRIPTIONS = {
    "Manufacturing — equipment failure": (
        "Each row represents a synthetic machine observation, with equipment age, operating hours, "
        "temperature, vibration, maintenance history, and load. The outcome records whether the machine "
        "fails within the next 30 days. Missing readings, uncommon failures, and a post-failure resolution "
        "field demonstrate data-quality and leakage checks."
    ),
    "Retail — product demand": (
        "Each row represents a synthetic product sold at one store on one day. Sales vary with price, "
        "promotions, holidays, seasonal patterns, and store/product differences. The proposed outcome is "
        "units sold; the dated history demonstrates why demand forecasting needs chronological evaluation."
    ),
    "Manufacturing — outcome absent": (
        "Each row represents a synthetic machine observation with operating conditions and maintenance "
        "history. Failure outcomes have deliberately been removed. This example demonstrates how the "
        "engine identifies a possible opportunity but requests historical labels before benchmarking."
    ),
}
STATUS_ICON = {
    ReadinessStatus.READY: "✅", ReadinessStatus.INVESTIGATE: "⚠️", ReadinessStatus.BLOCKED: "⛔",
}


st.set_page_config(page_title="TabPFN Use-Case Discovery", page_icon="◆", layout="wide")
st.markdown("""
<style>
  .block-container {max-width: 1200px; padding-top: 3.5rem; padding-bottom: 4rem;}
  [data-testid="stMetric"] {background:#f7f9fc;border:1px solid #e6eaf0;border-radius:12px;padding:14px 16px;}
  [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color:#24324a !important;}
  [data-testid="stMetricLabel"] p {color:#46566f !important;}
  .eyebrow {margin-top:1rem;font-size:.85rem;line-height:1.6;letter-spacing:.1em;text-transform:uppercase;color:var(--text-color);opacity:.85;font-weight:700;}
  .hero {font-size:2.25rem;line-height:1.12;font-weight:750;letter-spacing:-.04em;margin:.2rem 0 .5rem;}
  .subtle {color:var(--text-color);opacity:.8;max-width:780px;font-size:1.02rem;}
  .steps {margin-top:1.5rem;margin-bottom:.5rem;}
  .notice {background:#f2f6ff;border-left:3px solid #5271e6;padding:.8rem 1rem;border-radius:6px;color:#34425a;}
  .step {display:inline-block;padding:.35rem .7rem;margin:.1rem;border-radius:999px;background:#eef2f8;color:#536076;font-size:.8rem;}
  .step.active {background:#dfe7ff;color:#2949bd;font-weight:650;}
  .card {border:1px solid #e2e7ef;border-radius:14px;padding:1rem 1.1rem;margin:.45rem 0;background:#fff;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="eyebrow">Unofficial local prototype</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">TabPFN Use-Case Discovery Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="subtle">Turn a tabular dataset and a business question into transparent opportunity candidates, a data-readiness assessment, and an actionable POC scope.</div>', unsafe_allow_html=True)
st.markdown("<div class='steps'><span class='step active'>1 Dataset</span><span class='step'>2 Discover</span><span class='step'>3 Assess</span><span class='step'>4 Prioritize</span><span class='step'>5 Benchmark</span><span class='step'>6 POC Plan</span></div>", unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.header("About this V1")
    st.caption("Uploaded data is processed in this local Streamlit process and is not sent to an external API or stored by the app.")
    st.caption("Discovery uses explicit heuristics. TabPFN runs only when you explicitly start a benchmark. Results do not establish ROI or production readiness.")

st.header("1 · Choose a dataset")
source_tab, upload_tab = st.tabs(
    ["Try a synthetic example", "Upload your own CSV"], key="dataset_path", on_change="rerun",
)
df = None
dataset_name = ""
with source_tab:
    example = st.selectbox("Example dataset", list(SAMPLES))
    st.write(SAMPLE_DESCRIPTIONS[example])
    st.caption("All built-in records are deterministic, locally generated, and entirely synthetic.")
    if st.button("Load synthetic example", type="primary", width="stretch"):
        st.session_state["sample_source"] = ("sample", example)
with upload_tab:
    uploaded = st.file_uploader("CSV file", type=["csv"], help="UTF-8 CSV, up to 20 MB, 100,000 rows, and 500 columns.")
    if uploaded and st.button("Analyze upload", type="primary", width="stretch"):
        st.session_state["upload_source"] = ("upload", uploaded.name, uploaded.getvalue())

source = st.session_state.get("sample_source" if source_tab.open else "upload_source")
source_identity = (
    (source[0], source[1], hashlib.sha256(source[2]).hexdigest() if source[0] == "upload" else None)
    if source else None
)
if st.session_state.get("active_source_identity") != source_identity:
    for state_key in (
        "proposal", "proposal_key", "benchmark_result", "benchmark_context",
        "assessment_choice", "benchmark_choice", "poc_choice",
    ):
        st.session_state.pop(state_key, None)
    st.session_state["active_source_identity"] = source_identity
if source:
    try:
        if source[0] == "sample":
            dataset_name = source[1]
            df = load_csv(SAMPLES[dataset_name])
        else:
            dataset_name = source[1]
            df = load_csv(source[2])
    except DataValidationError as exc:
        st.error(str(exc))
        df = None

if df is None:
    st.info("Choose an example or upload a CSV to begin.")
    st.stop()

profile = profile_dataset(df)
st.success(f"Loaded {dataset_name}" + (" · Synthetic data" if source[0] == "sample" else ""))
metrics = st.columns(4)
metrics[0].metric("Rows", f"{profile.rows:,}")
metrics[1].metric("Columns", profile.columns)
metrics[2].metric("Missing cells", f"{profile.missing_total:,}")
metrics[3].metric("Duplicate rows", f"{profile.duplicate_rows:,}")

with st.expander("Preview and detected schema", expanded=True):
    st.dataframe(df.head(12), width="stretch", hide_index=True)
    schema = pd.DataFrame({"Column": list(profile.type_by_column), "Detected type": list(profile.type_by_column.values()),
                           "Missing": [profile.missing_by_column[c] for c in profile.type_by_column],
                           "Identifier-like": [c in profile.identifier_columns for c in profile.type_by_column]})
    st.dataframe(schema, width="stretch", hide_index=True)

chart_cols = st.columns(2)
with chart_cols[0]:
    type_counts = pd.Series(profile.type_by_column).value_counts().rename_axis("type").reset_index(name="columns")
    st.plotly_chart(px.bar(type_counts, x="type", y="columns", title="Feature-type breakdown", color_discrete_sequence=["#5271e6"]), width="stretch")
with chart_cols[1]:
    miss = pd.Series(profile.missing_by_column).sort_values(ascending=False)
    miss = miss[miss > 0].head(15).rename_axis("column").reset_index(name="missing")
    if miss.empty:
        st.info("No missing values detected.")
    else:
        st.plotly_chart(px.bar(miss, x="missing", y="column", orientation="h", title="Missing values", color_discrete_sequence=["#7990e8"]), width="stretch")

st.divider()
st.header("2 · Add business context and discover opportunities")
left, right = st.columns(2)
with left:
    industry = st.selectbox("Industry", ["Manufacturing", "Financial Services", "Retail", "Healthcare", "Other"],
                            index=0 if "Manufacturing" in dataset_name else 2 if "Retail" in dataset_name else 4)
    default_objective = "Reduce unplanned machine downtime by predicting equipment failure." if "Manufacturing" in dataset_name else "Improve product replenishment by anticipating demand." if "Retail" in dataset_name else ""
    objective = st.text_area("Business objective", value=default_objective, placeholder="What decision or outcome should improve?")
    priority = st.select_slider("Business priority (user-provided)", ["Low", "Medium", "High"], value="High")
with right:
    default_row = "One machine observation" if "Manufacturing" in dataset_name else "One product-store-day" if "Retail" in dataset_name else ""
    row_definition = st.text_input("What does each row represent?", value=default_row)
    timing = st.text_input("When would the prediction be made?", placeholder="e.g. At the start of each week, before scheduling maintenance")
    target_options = ["Not specified"] + list(df.columns)
    suggested = profile.target_candidates[0] if profile.target_candidates else "Not specified"
    selected_target = st.selectbox("Optional target variable", target_options, index=target_options.index(suggested) if suggested in target_options else 0)
    operational = st.select_slider("Operational importance (user-provided)", ["Low", "Medium", "High"], value="High")
cost_context = st.text_input("Approximate cost of the problem (optional, user-provided)", placeholder="e.g. Downtime cost is not yet quantified")

context = BusinessContext(industry, objective, row_definition, timing, None if selected_target == "Not specified" else selected_target)
opportunities = discover_use_cases(df, profile, context)
if not opportunities:
    st.warning("No defensible use-case candidate was found from the available context. Add a target or a more specific objective; the engine will not invent one.")
    st.stop()

assessments = {uc.key: assess_readiness(df, profile, uc) for uc in opportunities}
for uc in opportunities:
    assessment = assessments[uc.key]
    with st.container(border=True):
        heading_cols = st.columns([4, 1])
        heading_cols[0].subheader(uc.name)
        heading_cols[1].caption(uc.support.value)
        st.write(uc.problem_statement)
        st.caption(f"Target: {uc.target or 'Outcome data required'} · Task: {uc.task_type}")
        st.write("**Why it appeared**")
        for evidence in uc.evidence:
            st.markdown(f"- {evidence}")
        for uncertainty in uc.uncertainties:
            st.caption(f"⚑ {uncertainty}")

st.divider()
st.header("3 · Assess data readiness")
assessment_choice = st.selectbox("Use case to assess", opportunities, format_func=lambda x: x.name, key="assessment_choice")
assessment = assessments[assessment_choice.key]
st.subheader(f"{STATUS_ICON[assessment.status]} {assessment.status.value}")
facts_col, risk_col, action_col = st.columns(3)
with facts_col:
    st.markdown("**Observed facts**")
    for value in assessment.observed_facts:
        st.markdown(f"- {value}")
with risk_col:
    st.markdown("**Heuristic risks**")
    for value in assessment.risks or ["No material risk detected automatically."]:
        st.markdown(f"- {value}")
    for value in assessment.leakage_warnings:
        st.markdown(f"- Leakage check: {value}")
with action_col:
    st.markdown("**Actions**")
    for value in assessment.actions:
        st.markdown(f"- {value}")

if assessment.target_summary.get("class_distribution"):
    distribution = pd.Series(assessment.target_summary["class_distribution"]).rename_axis("class").reset_index(name="rows")
    st.plotly_chart(px.bar(distribution, x="class", y="rows", title="Target distribution", color_discrete_sequence=["#5271e6"]), width="stretch")
elif assessment_choice.target and assessment_choice.target in df and assessment_choice.task_type == "regression":
    st.plotly_chart(px.histogram(df, x=assessment_choice.target, nbins=30, title="Target distribution", color_discrete_sequence=["#5271e6"]), width="stretch")

with st.expander("Questions the data profile cannot answer"):
    for question in assessment.deployment_questions:
        st.markdown(f"- {question}")

st.divider()
st.header("4 · Prioritize opportunities")
ranking = prioritize(opportunities, assessments, priority, operational)
st.info(recommendation(ranking))
for rank, result in enumerate(ranking, 1):
    with st.container(border=True):
        cols = st.columns([.5, 3, 1.4, 1.2, 1.4])
        cols[0].markdown(f"### {rank}")
        cols[1].markdown(f"**{result.use_case.name}**  \n{result.use_case.business_rationale}")
        cols[2].caption("DATA READINESS")
        cols[2].write(result.data_feasibility)
        cols[3].caption("COMPLEXITY")
        cols[3].write(result.implementation_complexity)
        cols[4].caption("NEXT ACTION")
        cols[4].write("Scope evaluation" if "Sufficient" in result.decision_readiness else "Resolve data gap")

st.divider()
st.header("5 · Benchmark with TabPFN")
benchmark_case = st.selectbox("Confirmed use case", opportunities, format_func=lambda x: x.name, key="benchmark_choice")
benchmark_assessment = assessments[benchmark_case.key]
benchmark_result = None
if benchmark_assessment.status == ReadinessStatus.BLOCKED or not benchmark_case.target:
    st.error("Benchmarking is blocked until a usable target and sufficient historical labels are available.")
else:
    leakage_columns = suspected_leakage_columns(df, benchmark_case.target)
    defaults, excluded = recommended_features(df, profile, benchmark_case, leakage_columns)
    candidate_features = [str(c) for c in df.columns if str(c) != benchmark_case.target]
    selected_features = st.multiselect(
        "Features to use", candidate_features, default=defaults,
        help="Review these with a domain owner. Identifier-like and suspected leakage fields are excluded by default.",
    )
    if excluded:
        st.caption("Excluded by default: " + "; ".join(f"`{name}` — {reason}" for name, reason in excluded.items() if name != benchmark_case.target))

    try:
        preview = prepare_benchmark(df, profile, benchmark_case, selected_features)
        preview.plan.excluded_columns = excluded
        plan_cols = st.columns(3)
        plan_cols[0].metric("Training rows", f"{preview.plan.train_rows:,}")
        plan_cols[1].metric("Holdout rows", f"{preview.plan.test_rows:,}")
        plan_cols[2].metric("Selected features", len(selected_features))
        st.caption(f"Evaluation design: {preview.plan.split_strategy}.")
        benchmark_eligible = True
    except BenchmarkError as exc:
        st.warning(str(exc))
        benchmark_eligible = False

    env_token_available = bool(os.getenv("TABPFN_TOKEN"))
    token_input = st.text_input(
        "TabPFN API token" + (" (using TABPFN_TOKEN from environment)" if env_token_available else ""),
        type="password", placeholder="Leave blank to use TABPFN_TOKEN", help="The value is held only in this Streamlit session and is not written by this app.",
    )
    consent = st.checkbox(
        "I confirm I am permitted to send the selected training and holdout data to the TabPFN cloud service.",
    )
    if not env_token_available and not token_input:
        st.info("Enter a token or launch Streamlit with `TABPFN_TOKEN` set. Create a key at `platform.priorlabs.ai/account/api-keys`.")
    if st.button(
        "Run TabPFN benchmark", type="primary",
        disabled=not benchmark_eligible or not consent or not (env_token_available or bool(token_input)),
    ):
        with st.spinner("Running baselines and TabPFN on the frozen holdout…"):
            try:
                result = run_tabpfn_benchmark(df, profile, benchmark_case, selected_features, token_input or None)
                result.plan.excluded_columns = excluded
                st.session_state["benchmark_result"] = result
                st.session_state["benchmark_context"] = (dataset_name, benchmark_case.key, tuple(selected_features))
            except BenchmarkError as exc:
                st.error(str(exc))

    expected_context = (dataset_name, benchmark_case.key, tuple(selected_features))
    if st.session_state.get("benchmark_context") == expected_context:
        benchmark_result = st.session_state.get("benchmark_result")
    if benchmark_result:
        st.success("TabPFN inference completed. The values below are measured on the displayed holdout.")
        metrics_frame = pd.DataFrame(benchmark_result.metrics).T
        st.dataframe(metrics_frame.style.format("{:.3f}"), width="stretch")
        for note in benchmark_result.notes:
            st.caption(f"• {note}")

st.divider()
st.header("6 · Generate a POC plan")
chosen = st.selectbox("Select an opportunity", opportunities, format_func=lambda x: x.name, key="poc_choice")
if st.button("Generate POC proposal", type="primary"):
    matching_benchmark = st.session_state.get("benchmark_result") if st.session_state.get("benchmark_context", (None, None))[0:2] == (dataset_name, chosen.key) else None
    st.session_state["proposal"] = generate_poc_markdown(dataset_name, profile, context, chosen, assessments[chosen.key], cost_context or "Not provided", matching_benchmark)
    st.session_state["proposal_key"] = chosen.key
proposal = st.session_state.get("proposal")
if proposal:
    st.markdown(proposal)
    st.download_button("Download Markdown proposal", proposal, file_name=f"{st.session_state.get('proposal_key', 'use-case')}-poc-proposal.md", mime="text/markdown")

st.caption("Unofficial prototype for demonstration purposes. Not affiliated with or endorsed by Prior Labs or SAP.")
