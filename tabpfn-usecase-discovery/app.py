from __future__ import annotations

import os
import hashlib
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.benchmark import BenchmarkError, prepare_benchmark, recommended_features, run_tabpfn_benchmark
from src.capabilities import CAPABILITIES, execution_note
from src.objective_refinement import RefinedObjective, RefinementError, ROUTE_LABELS, refine_objective, route_brief
from src.data_loader import DataValidationError, load_csv
from src.data_profiler import profile_dataset
from src.poc_generator import generate_poc_markdown
from src.prioritization import prioritize, recommendation
from src.readiness import assess_readiness, suspected_leakage_columns
from src.schemas import BusinessContext, ReadinessStatus, SupportLevel


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
progress = st.empty()


def show_step(step: int) -> None:
    labels = ['Dataset', 'Discover', 'Assess', 'Prioritize', 'Evaluate', 'POC Plan']
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
    st.caption("Profiling runs locally. Refine objective sends your context and computed schema summary to OpenAI. The TabPFN benchmark sends selected data only when you start it.")
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
    for state_key in ("proposal", "proposal_key", "benchmark_result", "benchmark_context", "assessment_choice", "benchmark_choice", "poc_choice", "refinement_draft", "confirmed_brief"):
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
            if state_key.startswith('brief_edit_') or state_key in ('confirmed_brief', 'proposal', 'proposal_key', 'benchmark_result', 'benchmark_context', 'assessment_choice', 'benchmark_choice', 'poc_choice'):
                st.session_state.pop(state_key, None)
    except RefinementError as exc:
        st.error(str(exc))

draft = st.session_state.get('refinement_draft')
if draft:
    with st.form('review_objective_brief'):
        st.subheader('Review the proposed analysis brief')
        brief_objective = st.text_area('Refined objective', value=draft.objective, key='brief_edit_objective')
        brief_decision = st.text_area('Business decision', value=draft.business_decision, key='brief_edit_decision')
        brief_route = st.selectbox('Recommended TabPFN capability', list(ROUTE_LABELS), index=list(ROUTE_LABELS).index(draft.route), format_func=lambda x: ROUTE_LABELS[x], key='brief_edit_route')
        brief_target = st.selectbox('Outcome for this analysis (if required)', target_options, index=target_options.index(draft.target) if draft.target else 0, key='brief_edit_target')
        brief_columns = st.multiselect('Relevant columns to review', list(df.columns), default=draft.relevant_columns, key='brief_edit_columns')
        brief_reason = st.text_area('Why this capability fits', value=draft.rationale, key='brief_edit_reason')
        brief_deliverable = st.text_area('Expected deliverable', value=draft.deliverable, key='brief_edit_deliverable')
        brief_criteria = st.text_area('Success criteria (one per line)', value='\n'.join(draft.success_criteria), key='brief_edit_criteria')
        brief_questions = st.text_area('Open questions (one per line)', value='\n'.join(draft.questions), key='brief_edit_questions')
        apply_brief = st.form_submit_button('Use this objective and capability', type='primary')
    if apply_brief:
        try:
            confirmed = RefinedObjective(objective=brief_objective, business_decision=brief_decision, route=brief_route,
                target=None if brief_target == 'Not specified' else brief_target, relevant_columns=brief_columns,
                rationale=brief_reason, deliverable=brief_deliverable,
                success_criteria=[s.strip() for s in brief_criteria.splitlines() if s.strip()],
                questions=[s.strip() for s in brief_questions.splitlines() if s.strip()])
            route_brief(confirmed, df, profile)
            st.session_state['confirmed_brief'] = confirmed
            st.session_state['workflow_step'] = 3
            for state_key in ('proposal', 'proposal_key', 'benchmark_result', 'benchmark_context', 'assessment_choice', 'benchmark_choice', 'poc_choice'):
                st.session_state.pop(state_key, None)
        except RefinementError as exc:
            st.error(str(exc))
            st.stop()
    if st.button('Discard refined objective'):
        st.session_state.pop('workflow_step', None)
        for state_key in ('refinement_draft', 'confirmed_brief', 'proposal', 'benchmark_result', 'benchmark_context', 'assessment_choice', 'benchmark_choice', 'poc_choice'):
            st.session_state.pop(state_key, None)
        st.rerun()
    if 'confirmed_brief' not in st.session_state:
        st.info('Review and apply the brief to continue to assessment and evaluation.')
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
    assessment = assessments[uc.key]
    with st.container(border=True):
        heading_cols = st.columns([4, 1])
        heading_cols[0].subheader(uc.name)
        heading_cols[1].caption(uc.support.value)
        st.write(uc.problem_statement)
        target_note = uc.target or ("Not required" if uc.capability != "prediction" and not CAPABILITIES[uc.capability]["requires_target"] else "Needs confirmation")
        st.caption(f"Target: {target_note} · Analysis: {uc.task_type}")
        st.caption(execution_note(uc.capability))
        st.write("**Why it appeared**")
        for evidence in uc.evidence:
            st.markdown(f"- {evidence}")
        for uncertainty in uc.uncertainties:
            st.caption(f"⚑ {uncertainty}")

st.divider()
show_step(3)
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
continue_to(4, 'Continue to prioritization')
st.header("4 · Prioritize opportunities")
ranking = prioritize(opportunities, assessments)
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
        st.caption(result.recommendation_reason)

st.divider()
continue_to(5, 'Continue to evaluation')
st.header("5 · Evaluate the proposed approach")
benchmark_case = st.selectbox("Confirmed use case", opportunities, format_func=lambda x: x.name, key="benchmark_choice")
benchmark_assessment = assessments[benchmark_case.key]
benchmark_result = None
if benchmark_case.capability != "prediction":
    guide = CAPABILITIES[benchmark_case.capability]
    st.info(execution_note(benchmark_case.capability))
    st.write(f"**Evaluation approach:** {guide['design']}")
    st.write(f"**Baseline:** {guide['baseline']}")
    st.write(f"**Success measures:** {guide['metrics']}")
    st.write(f"**Deliverable:** {guide['deliverable']}")
elif benchmark_assessment.status == ReadinessStatus.BLOCKED or not benchmark_case.target:
    st.error("Benchmarking is blocked until a usable target and sufficient historical labels are available.")
else:
    leakage_columns = suspected_leakage_columns(df, benchmark_case.target)
    defaults, excluded = recommended_features(df, profile, benchmark_case, leakage_columns)
    if benchmark_case.suggested_columns:
        defaults = [c for c in defaults if c in benchmark_case.suggested_columns]
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
continue_to(6, 'Continue to POC plan' if benchmark_result else 'Continue to POC plan without benchmark results')
st.header("6 · Generate a POC plan")
chosen = st.selectbox("Select an opportunity", opportunities, format_func=lambda x: x.name, key="poc_choice")
if st.button("Generate POC proposal", type="primary"):
    matching_benchmark = st.session_state.get("benchmark_result") if st.session_state.get("benchmark_context", (None, None))[0:2] == (dataset_name, chosen.key) else None
    st.session_state["proposal"] = generate_poc_markdown(dataset_name, profile, context, chosen, assessments[chosen.key], benchmark=matching_benchmark)
    st.session_state["proposal_key"] = chosen.key
proposal = st.session_state.get("proposal")
if proposal:
    st.markdown(proposal)
    st.download_button("Download Markdown proposal", proposal, file_name=f"{st.session_state.get('proposal_key', 'use-case')}-poc-proposal.md", mime="text/markdown")

st.caption("Unofficial prototype for demonstration purposes. Not affiliated with or endorsed by Prior Labs or SAP.")
