from __future__ import annotations

import os
import hashlib
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.benchmark import BenchmarkError, prepare_benchmark, recommended_features, run_tabpfn_benchmark
from src.analysis import analysis_columns, run_tabpfn_analysis
from src.result_views import render_analysis, render_benchmark
from src.capabilities import CAPABILITIES
from src.objective_refinement import RefinedObjective, RefinementError, ROUTE_LABELS, refine_objective, route_brief
from src.data_loader import DataValidationError, load_csv
from src.data_profiler import profile_dataset
from src.prioritization import prioritize, recommendation
from src.readiness import assess_readiness, suspected_leakage_columns
from src.schemas import BusinessContext, ReadinessStatus, SupportLevel


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env", override=False)
SAMPLES = {
    "Manufacturing — equipment failure": ROOT / "data/samples/synthetic_manufacturing.csv",
    "Retail — product demand": ROOT / "data/samples/synthetic_retail.csv",
}
SAMPLE_DESCRIPTIONS = {
    "Manufacturing — equipment failure": (
        "Each row represents a synthetic machine observation, with equipment age, operating hours, "
        "temperature, vibration, maintenance history, and load. The outcome records whether the machine "
        "fails within the next 30 days. Missing readings, uncommon failures, and a post-failure resolution "
        "field demonstrate data-quality and leakage checks."
    ),
    "Retail — product demand": (
        "Each row represents one product at one store over a full trading day, from opening to closing. "
        "units_sold is the actual number of units sold during that day. Inputs describe conditions known "
        "before opening: planned selling price, discount percentage, expected store visitors, opening stock, "
        "the prior seven-day sales average, and the prior four-week same-weekday sales average. Store and "
        "product IDs identify the item and location; the date identifies the trading day. Sales history comes "
        "from completed prior days. This is a daily sales regression example. Sales can be limited by "
        "stock availability, so units_sold does not measure all unmet customer demand."
    ),
}
SAMPLE_GENERATION_OBJECTIVES = {
    "Manufacturing — equipment failure": (
        "Create realistic machine history for exploring whether equipment likely to fail within the next "
        "30 days can be identified early enough to prioritize preventive inspection and maintenance."
    ),
    "Retail — product demand": (
        "Estimate how many units of a product a store will sell during one full trading day using "
        "information available before opening, to support stock allocation and daily operational planning."
    ),
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
st.markdown('<div class="subtle">Turn a tabular dataset and a business question into a clear TabPFN opportunity and a measured initial evaluation.</div>', unsafe_allow_html=True)
progress = st.empty()


def show_step(step: int) -> None:
    labels = ['Dataset', 'Objective', 'Recommendation', 'Evaluate', 'Next steps']
    badges = ''.join(f"<span class='step{' active' if i == step else ''}'>{i} {label}</span>" for i, label in enumerate(labels, 1))
    progress.markdown(f"<div class='steps'>{badges}</div>", unsafe_allow_html=True)


def continue_to(step: int, label: str) -> None:
    if st.session_state.get('workflow_step', 3) < step:
        if st.button(label, type='primary', key=f'continue_step_{step}'):
            st.session_state['workflow_step'] = step
        else:
            st.stop()
    show_step(step)


show_step(1)
st.write("")

with st.sidebar:
    st.header("About this V1")
    st.caption("Profiling runs locally. Refine objective sends your context and computed schema summary to OpenAI. Evaluations send selected data to TabPFN only when you start them.")
    st.caption("Discovery can use local rules or a confirmed AI brief. Results do not establish ROI or production readiness.")

st.header("1 · Choose a dataset")
source_tab, upload_tab = st.tabs(
    ["Try a synthetic example", "Upload your own CSV"], key="dataset_path", on_change="rerun",
)
df = None
dataset_name = ""
with source_tab:
    example = st.selectbox("Example dataset", list(SAMPLES))
    st.write(SAMPLE_DESCRIPTIONS[example])
    st.markdown(f"**Why this dataset was generated:** {SAMPLE_GENERATION_OBJECTIVES[example]}")
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
    st.session_state.pop('workflow_step', None)
    for state_key in (
        "benchmark_result", "benchmark_context",
        "benchmark_choice", "evaluation_fields", "analysis_fields",
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
show_step(2)
st.header("2 · Add business context and discover opportunities")
left, right = st.columns(2)
with left:
    industry = st.selectbox("Industry", ["Manufacturing", "Financial Services", "Retail", "Healthcare", "Other"],
                            index=0 if "Manufacturing" in dataset_name else 2 if "Retail" in dataset_name else 4)
    objective = st.text_area("Business objective", value="", placeholder="Describe what you want to achieve: improve decisions, identify unusual records, recover missing data, explain outcomes, or generate test data.")
with right:
    data_description = st.text_area("Information about the data", value=SAMPLE_DESCRIPTIONS[dataset_name] if source[0] == "sample" else "", placeholder="Describe the records, column meanings, collection process, time coverage, and any known limitations.")
    target_options = ["Not specified"] + list(df.columns)
    selected_target = st.selectbox("Optional target variable", target_options, help="Leave unspecified for exploration, outlier detection, missing-data recovery, or synthetic generation.")

context = BusinessContext(industry=industry, objective=objective, data_description=data_description, selected_target=None if selected_target == "Not specified" else selected_target)
context_identity = (source_identity, industry, objective, data_description, selected_target)
if st.session_state.get("analysis_context") != context_identity:
    st.session_state.pop('workflow_step', None)
    for state_key in ("benchmark_result", "benchmark_context", "benchmark_choice", "refinement_draft", "confirmed_brief", "evaluation_fields", "analysis_fields"):
        st.session_state.pop(state_key, None)
    for state_key in list(st.session_state):
        if state_key.startswith('brief_edit_'):
            st.session_state.pop(state_key, None)
    st.session_state["analysis_context"] = context_identity
st.caption("Refine objective sends these four inputs and computed column statistics to OpenAI, without raw rows. Review the proposed TabPFN approach before applying it.")
openai_key = os.getenv('OPENAI_API_KEY', '')
if not openai_key:
    st.info('To enable refinement, add OPENAI_API_KEY to the project .env file and restart Streamlit.')
if st.button('Refine objective', disabled=not openai_key or not objective.strip()):
    try:
        with st.spinner('Refining the objective and selecting a TabPFN capability…'):
            draft = refine_objective(context, profile, api_key=openai_key, model=os.getenv('OPENAI_MODEL') or 'gpt-4.1-mini')
        st.session_state['refinement_draft'] = draft
        st.session_state.pop('workflow_step', None)
        for state_key in list(st.session_state):
            if state_key.startswith('brief_edit_') or state_key in ('confirmed_brief', 'benchmark_result', 'benchmark_context', 'benchmark_choice'):
                st.session_state.pop(state_key, None)
    except RefinementError as exc:
        st.error(str(exc))

draft = st.session_state.get('refinement_draft')
if draft:
    with st.form('review_objective_brief'):
        st.subheader('Review the proposed analysis brief')
        brief_objective = st.text_area('Refined objective', value=draft.objective, key='brief_edit_objective')
        brief_route = st.selectbox('Recommended TabPFN capability', list(ROUTE_LABELS), index=list(ROUTE_LABELS).index(draft.route), format_func=lambda x: ROUTE_LABELS[x], key='brief_edit_route')
        brief_target = st.selectbox('Outcome for this analysis (if required)', target_options, index=target_options.index(draft.target) if draft.target else 0, key='brief_edit_target')
        brief_columns = st.multiselect('Relevant columns to review', list(df.columns), default=draft.relevant_columns, key='brief_edit_columns')
        brief_reason = st.text_area('Why this capability fits', value=draft.rationale, key='brief_edit_reason')
        brief_deliverable = st.text_area('Expected deliverable', value=draft.deliverable, key='brief_edit_deliverable')
        apply_brief = st.form_submit_button('Use this objective and capability', type='primary')
    if apply_brief:
        try:
            confirmed = RefinedObjective(objective=brief_objective, business_decision=draft.business_decision, route=brief_route,
                target=None if brief_target == 'Not specified' else brief_target, relevant_columns=brief_columns,
                rationale=brief_reason, deliverable=brief_deliverable,
                success_criteria=draft.success_criteria, questions=draft.questions)
            route_brief(confirmed, df, profile)
            st.session_state['confirmed_brief'] = confirmed
            st.session_state['workflow_step'] = 3
            for state_key in ('benchmark_result', 'benchmark_context', 'benchmark_choice', 'evaluation_fields', 'analysis_fields'):
                st.session_state.pop(state_key, None)
        except RefinementError as exc:
            st.error(str(exc))
            st.stop()
    if st.button('Discard refined objective'):
        st.session_state.pop('workflow_step', None)
        for state_key in ('refinement_draft', 'confirmed_brief', 'benchmark_result', 'benchmark_context', 'benchmark_choice'):
            st.session_state.pop(state_key, None)
        st.rerun()
    if 'confirmed_brief' not in st.session_state:
        st.info('Review and apply the brief to continue to the recommendation and evaluation.')
        st.stop()

confirmed = st.session_state.get('confirmed_brief')
if confirmed:
    routed = route_brief(confirmed, df, profile)
    if routed is None:
        st.warning('Clarification is needed before choosing a TabPFN capability.')
        for question in confirmed.questions:
            st.write(f'- {question}')
        st.stop()
    opportunities = [routed]
    st.success(f'Confirmed approach: {ROUTE_LABELS[confirmed.route]}')
else:
    st.info('Enter your business objective above, then click Refine objective to generate an analysis brief.')
    st.stop()
if not opportunities:
    st.warning("No defensible use-case candidate was found from the available context. Add a target or a more specific objective; the engine will not invent one.")
    st.stop()

assessments = {uc.key: assess_readiness(df, profile, uc) for uc in opportunities}
for uc in opportunities:
    with st.container(border=True):
        st.write(uc.problem_statement)
        st.caption(f"Approach: {ROUTE_LABELS[confirmed.route]}")
        if uc.target:
            st.markdown(f"**Proposed outcome:** `{uc.target}`")
            summary = assessments[uc.key].target_summary
            if summary:
                st.write(f"The outcome is missing in {summary.get('missing_rate', 0):.1%} of records and has {summary.get('unique_values', 0):,} observed value(s).")
                if summary.get("minority_rate") is not None:
                    st.write(f"The less common outcome appears in {summary['minority_rate']:.1%} of labeled records. The evaluation will focus on finding that outcome, rather than relying on overall accuracy.")
        elif uc.capability == 'prediction' or CAPABILITIES[uc.capability]['requires_target']:
            st.caption("Outcome needs confirmation")
        else:
            st.markdown("**No single outcome is required for this analysis.**")
        fields_to_check = suspected_leakage_columns(df, uc.target)
        if fields_to_check:
            st.write("**Fields to check:** " + ", ".join(f"`{column}`" for column in fields_to_check))
            st.caption("These fields may only be known after the outcome. They are excluded from the recommended inputs by default.")
        if profile.identifier_columns:
            st.write("**Identifier-like fields:** " + ", ".join(f"`{column}`" for column in profile.identifier_columns))
            st.caption("Unique identifiers are excluded by default because they usually name a record rather than describe its condition.")
        st.caption('An initial evaluation can be run in Step 4.')

st.divider()
show_step(3)
st.header("3 · Review the recommendation")
ranking = prioritize(opportunities, assessments)
st.info(recommendation(ranking))
for result in ranking:
    with st.container(border=True):
        st.subheader(result.use_case.name)
        st.write(result.use_case.business_rationale)
        cols = st.columns(3)
        cols[0].caption("DATA CONFIDENCE")
        cols[0].write(result.data_feasibility)
        cols[1].caption("IMPLEMENTATION EFFORT")
        cols[1].write(result.implementation_complexity)
        cols[2].caption("NEXT STEP")
        blocked = assessments[result.use_case.key].status == ReadinessStatus.BLOCKED
        cols[2].write("Update the dataset or objective" if blocked else "Run an initial evaluation")

assessment = assessments[opportunities[0].key]
if assessment.status == ReadinessStatus.BLOCKED:
    for reason in assessment.risks:
        st.warning(reason)
    for action in assessment.actions:
        st.write(action)
    st.stop()

st.divider()
continue_to(4, 'Continue to evaluation')
st.header("4 · Evaluate the proposed approach")
benchmark_case = st.selectbox("Confirmed use case", opportunities, format_func=lambda x: x.name, key="benchmark_choice")
benchmark_assessment = assessments[benchmark_case.key]
benchmark_result = None
supervised = benchmark_case.capability in {"prediction", "interpretability"}
runner_case = benchmark_case
generated_rows = 100
explain = False
benchmark_eligible = False
selected_features = []
excluded = {}
if supervised:
    if benchmark_case.capability == "interpretability":
        numeric = bool(benchmark_case.target and pd.api.types.is_numeric_dtype(df[benchmark_case.target]))
        outcome_type = st.selectbox("What kind of outcome are we explaining?", ["A category or event", "A numeric value"], index=1 if numeric and df[benchmark_case.target].nunique() > 20 else 0)
        task = "regression" if outcome_type == "A numeric value" else "binary classification" if df[benchmark_case.target].nunique() == 2 else "multiclass classification"
        runner_case = replace(benchmark_case, task_type=task, capability="prediction")
        st.write("We first check the model on unseen records, then explain which fields it relied on and how individual estimates change.")
    elif benchmark_case.forecasting:
        st.write("Test estimates on the latest historical dates and compare them with the values that actually occurred.")
    else:
        st.write("Compare TabPFN with simple approaches on unseen records, then inspect correct results, mistakes, and influential fields.")
    leakage_columns = suspected_leakage_columns(df, benchmark_case.target)
    defaults, excluded = recommended_features(df, profile, runner_case, leakage_columns)
    if benchmark_case.suggested_columns:
        defaults = [c for c in defaults if c in benchmark_case.suggested_columns]
    candidate_features = [str(c) for c in df.columns if str(c) != benchmark_case.target]
    selected_features = st.multiselect(
        "Fields to use", candidate_features, default=defaults, key="evaluation_fields",
        help="Review these with a domain owner. Identifier-like and suspected leakage fields are excluded by default.",
    )
    if excluded:
        st.caption("Excluded by default: " + "; ".join(f"`{name}` — {reason}" for name, reason in excluded.items() if name != benchmark_case.target))

    try:
        preview = prepare_benchmark(df, profile, runner_case, selected_features)
        preview.plan.excluded_columns = excluded
        plan_cols = st.columns(3)
        plan_cols[0].metric("Learning records", f"{preview.plan.train_rows:,}")
        plan_cols[1].metric("Unseen test records", f"{preview.plan.test_rows:,}")
        plan_cols[2].metric("Selected fields", len(selected_features))
        st.caption(f"Evaluation design: {preview.plan.split_strategy}.")
        benchmark_eligible = True
    except BenchmarkError as exc:
        st.warning(str(exc))
        benchmark_eligible = False

    explain = st.checkbox("Explain the predictions", value=len(selected_features) <= 12,
                          disabled=len(selected_features) > 12,
                          help="Calculates SHAP contributions for individual predictions using extra API requests. Select at most 12 fields.")
    st.caption("SHAP explains up to five individual results using eight reference records. Exact contributions are calculated for up to six fields; larger analyses use 16 sampled permutations. This requires additional API usage.")
    if benchmark_case.capability == "interpretability" and not explain:
        st.info("Choose at most 12 fields to evaluate and explain this approach.")
        benchmark_eligible = False
else:
    guide = CAPABILITIES[benchmark_case.capability]
    st.write(guide['purpose'])
    available = analysis_columns(df, profile)
    defaults = [c for c in available if not benchmark_case.suggested_columns or c in benchmark_case.suggested_columns][:8]
    selected_features = st.multiselect("Fields to analyze", available, default=defaults, key="analysis_fields")
    st.caption("Initial evaluations use up to 1,000 distinct records and 2–8 fields. Identifiers, constants, and date fields are omitted. Reference records and unseen evaluation records stay separate.")
    benchmark_eligible = 2 <= len(selected_features) <= 8
    if not benchmark_eligible:
        st.info("Choose 2–8 fields to continue.")
    if benchmark_case.capability == "synthetic_generation":
        generated_rows = st.slider("Records to generate", 20, 500, 100, step=20)
        st.caption("Generate records with conditional TabPFN sampling, then compare their distributions and relationships with untouched original records and a resampling baseline.")
    elif benchmark_case.capability == "imputation":
        st.caption("Temporarily hide known values to test recovery, compare with a simple median/category baseline, and fill real gaps in unseen evaluation records. Each field is assessed separately.")
    else:
        st.caption("Rank unseen records by how surprising their values are to conditional TabPFN models. Compare with Isolation Forest and inspect the fields behind each alert.")
    st.caption("These evaluations make several TabPFN API requests per selected field. Additional API usage applies.")
    with st.expander("How this initial evaluation works"):
        st.write("The app uses conditional column models through the TabPFN API. This is a bounded initial check of the proposed approach; the full experimental joint-density extension is not used. The results include method-specific limitations for you to review.")

configured_token = os.getenv("TABPFN_TOKEN", "").strip()
consent = st.checkbox("I confirm I am permitted to send the selected data to the TabPFN cloud service for this evaluation and its explanation checks.")
if not configured_token:
    st.info("Evaluation is unavailable until TABPFN_TOKEN is configured in the project's .env file. Restart Streamlit after adding it.")
expected_context = (dataset_name, benchmark_case.key, tuple(selected_features), runner_case.task_type, explain, generated_rows, "shap-v1")
if st.session_state.get("evaluation_config") != expected_context:
    st.session_state["evaluation_config"] = expected_context
if st.button("Run TabPFN evaluation", type="primary", disabled=not benchmark_eligible or not consent or not configured_token):
    st.session_state.pop("benchmark_result", None)
    st.session_state.pop("benchmark_context", None)
    with st.spinner("Evaluating with TabPFN and preparing the result charts…"):
        try:
            if supervised:
                result = run_tabpfn_benchmark(df, profile, runner_case, selected_features, configured_token, explain=explain, include_global_importance=False)
                result.plan.excluded_columns = excluded
            else:
                result = run_tabpfn_analysis(df, profile, benchmark_case.capability, selected_features, configured_token, generated_rows=generated_rows)
            st.session_state["benchmark_result"] = result
            st.session_state["benchmark_context"] = expected_context
        except BenchmarkError as exc:
            st.error(str(exc))
if st.session_state.get("benchmark_context") == expected_context:
    benchmark_result = st.session_state.get("benchmark_result")
if benchmark_result:
    st.success("Evaluation completed. The charts below show measured results from this run.")
    if supervised:
        render_benchmark(benchmark_result, forecasting=benchmark_case.forecasting)
    else:
        render_analysis(benchmark_result)

st.divider()
continue_to(5, 'If interested, see official next steps')
st.header("5. Interested ? Possible next steps")
st.write("Use these official resources to learn more, try TabPFN directly, inspect implementation examples, or discuss a real use case with Prior Labs.")

capability_docs = {
    "classification": "https://docs.priorlabs.ai/capabilities/classification",
    "regression": "https://docs.priorlabs.ai/capabilities/regression",
    "forecasting": "https://docs.priorlabs.ai/capabilities/forecasting",
    "interpretability": "https://docs.priorlabs.ai/capabilities/interpretability",
    "anomaly_detection": "https://docs.priorlabs.ai/capabilities/anomaly-detection",
    "synthetic_generation": "https://docs.priorlabs.ai/capabilities/data-generation",
    "imputation": "https://github.com/PriorLabs/tabpfn-extensions",
}
route = confirmed.route
with st.container(border=True):
    st.subheader("Start with your recommended capability")
    st.write(f"Your confirmed approach is **{ROUTE_LABELS[route]}**. Read the official guide to understand the supported workflow and current implementation options.")
    st.link_button("Open the recommended capability guide ↗", capability_docs[route], type="primary")

learn, try_it, examples = st.columns(3)
with learn:
    st.subheader("Learn")
    st.write("Understand TabPFN and follow the official getting-started guide.")
    st.link_button("Documentation overview ↗", "https://docs.priorlabs.ai/overview")
    st.link_button("Quickstart ↗", "https://docs.priorlabs.ai/quickstart")
with try_it:
    st.subheader("Try")
    st.write("Manage API access or follow the official benchmarking walkthrough.")
    st.link_button("Manage API keys ↗", "https://platform.priorlabs.ai/account/api-keys")
    st.link_button("Benchmarking guide ↗", "https://docs.priorlabs.ai/benchmarking")
with examples:
    st.subheader("Explore examples")
    st.write("Review the maintained client and extension projects.")
    st.link_button("TabPFN client on GitHub ↗", "https://github.com/PriorLabs/tabpfn-client")
    st.link_button("TabPFN extensions on GitHub ↗", "https://github.com/PriorLabs/tabpfn-extensions")

st.subheader("Talk to Prior Labs or the community")
contact, community = st.columns(2)
contact.link_button("Book a demo with Prior Labs ↗", "https://priorlabs.ai/contact", type="primary")
community.link_button("Join the TabPFN Discord ↗", "https://discord.com/invite/VJRuU3bSxt")
st.caption("These links open official Prior Labs documentation, platforms, repositories, and community channels in a new tab.")

st.caption("Unofficial prototype for demonstration purposes. Not affiliated with or endorsed by Prior Labs or SAP.")
