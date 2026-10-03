from __future__ import annotations

from .schemas import AnalysisResult, BenchmarkResult, BusinessContext, DatasetProfile, ReadinessAssessment, UseCase
from .capabilities import CAPABILITIES


def generate_poc_markdown(
    dataset_name: str, profile: DatasetProfile, context: BusinessContext,
    use_case: UseCase, assessment: ReadinessAssessment,
    cost_context: str = "Not provided", benchmark: BenchmarkResult | AnalysisResult | None = None,
) -> str:
    proposal = _generate_poc_markdown(dataset_name, profile, context, use_case, assessment, cost_context, benchmark)
    if benchmark and use_case.capability != "prediction":
        import pandas as pd
        frame = benchmark.metrics if isinstance(benchmark, AnalysisResult) else pd.DataFrame(benchmark.metrics).T.reset_index(names="Approach")
        header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
        separator = "|" + "---|" * len(frame.columns)
        rows = ["| " + " | ".join(str(value).replace("|", "/") for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
        proposal += "\n## Measured initial evaluation\n\n" + "\n".join([header, separator, *rows]) + "\n\n" + "\n".join(f"- {note}" for note in benchmark.notes)
        if isinstance(benchmark, BenchmarkResult) and benchmark.feature_importance is not None:
            proposal += "\n\nPredictive field reliance (permutation importance):\n" + "\n".join(f"- {row['Field']}: {row['Importance']:.4f}" for _, row in benchmark.feature_importance.iterrows())
    if use_case.deliverable:
        criteria = '\n'.join(f'- {item}' for item in use_case.success_criteria)
        questions = '\n'.join(f'- {item}' for item in use_case.uncertainties)
        proposal += f"""
## Confirmed objective brief

**Business decision:** {use_case.business_rationale}

**Refined objective:** {use_case.problem_statement}

**Deliverable:** {use_case.deliverable}

**Relevant columns proposed for review:** {', '.join(use_case.suggested_columns) or 'Confirm appropriate columns.'}

**Success criteria to agree:**

{criteria}

**Open questions:**

{questions}

This brief was proposed by OpenAI and confirmed or edited by the user. It is not evidence of measured performance.
"""
    return proposal


def _generate_poc_markdown(
    dataset_name: str, profile: DatasetProfile, context: BusinessContext,
    use_case: UseCase, assessment: ReadinessAssessment,
    cost_context: str = "Not provided",
    benchmark: BenchmarkResult | None = None,
) -> str:
    if use_case.capability != "prediction":
        return _analysis_proposal(dataset_name, profile, context, use_case, assessment)
    target = f"`{use_case.target}`" if use_case.target else "Not yet available"
    classification = "classification" in use_case.task_type
    if use_case.forecasting:
        split = "Chronological holdout using the latest period, supplemented by rolling-origin backtests."
    elif profile.identifier_columns:
        split = "A grouped train/test split if entities repeat; otherwise a stratified split for classification or random holdout for regression."
    else:
        split = "A stratified holdout for classification or a random holdout for regression, after confirming rows are independent and non-temporal."
    metrics = (
        "PR-AUC and ROC-AUC, plus precision, recall, and false-positive/false-negative tradeoffs at an operational threshold."
        if classification else "MAE and RMSE, plus an error measure aligned to the business decision and relevant segments."
    )
    baseline = "Logistic regression and a tree-based classifier" if classification else "A naive historical/mean baseline and a tree-based regressor"
    facts = "\n".join(f"- {x}" for x in assessment.observed_facts)
    risks = assessment.risks + assessment.leakage_warnings
    risk_text = "\n".join(f"- {x}" for x in risks) if risks else "- No critical issue was detected automatically; expert validation is still required."
    questions = "\n".join(f"- {x}" for x in assessment.deployment_questions)
    objective = context.objective.strip() or "Confirm the business decision this prediction should improve."
    timing = context.prediction_timing.strip() or "Not provided—must be agreed before feature selection."
    if benchmark:
        metric_names = list(next(iter(benchmark.metrics.values())).keys())
        header = "| Model | " + " | ".join(metric_names) + " |"
        divider = "|---|" + "---:|" * len(metric_names)
        rows = []
        for model_name, values in benchmark.metrics.items():
            rows.append("| " + model_name + " | " + " | ".join(f"{values.get(metric, float('nan')):.3f}" for metric in metric_names) + " |")
        benchmark_section = f"""- **Measured benchmark:** TabPFN and baselines were evaluated on the same {benchmark.plan.split_strategy.lower()} ({benchmark.plan.train_rows:,} train / {benchmark.plan.test_rows:,} test rows).

{header}
{divider}
{chr(10).join(rows)}

These are single-holdout experimental results, not evidence of production performance or business impact."""
    else:
        benchmark_section = "- **TabPFN evaluation:** If TabPFN is made available later, evaluate it on the identical frozen splits and metrics against the baselines. No TabPFN inference has been run by this prototype."
    return f"""# POC Proposal: {use_case.name}

> Generated by the unofficial TabPFN Use-Case Discovery Engine. This is a scoping document, not evidence of model performance, ROI, or production readiness.

## 1. Business problem

**User-provided objective:** {objective}

**Proposed business contribution:** {use_case.business_rationale}

**Deliverable:** A validated decision-support workflow for the stated objective, including model outputs, operational review, and a go/no-go decision. Confirm the owner and action before starting.

## 2. Prediction objective

{use_case.problem_statement}

- Proposed target: {target}
- Task framing: {use_case.task_type}
- Prediction timing: {timing}
- Confirmation required: verify the target occurs after the prediction point and represents the intended business outcome.

## 3. Dataset overview

- Dataset: {dataset_name}
- Industry: {context.industry}
- User-provided data information: {context.data_description or context.row_definition or "Not provided; confirm population, provenance, and timing."}
- Shape: {profile.rows:,} rows × {profile.columns} columns
- Feature coverage: {len(profile.numeric_columns)} numeric, {len(profile.categorical_columns)} categorical, {len(profile.datetime_columns)} date/time
- Missing cells: {profile.missing_total:,}
- Duplicate rows: {profile.duplicate_rows:,}
- Identifier-like columns: {", ".join(profile.identifier_columns) if profile.identifier_columns else "None detected"}

Observed target facts:

{facts}

## 4. Evaluation design

- **Design:** {split}
- **Baselines:** {baseline}. Baselines establish whether a more capable model adds value.
- **Metrics:** {metrics}
- **Leakage precautions:** Remove direct target proxies and post-event fields; fit preprocessing only on training data; keep related entities in one split; verify feature availability at prediction time.
{benchmark_section}

## 5. Success criteria to agree

- Agree on the minimum acceptable performance relative to the baseline; do not choose a value before reviewing business tradeoffs.
- Agree on acceptable false-negative and false-positive costs with the operational owner.
- Confirm performance is sufficiently stable across relevant sites, products, machines, or customer groups.
- Confirm the prediction arrives early enough and reliably enough to support the intended action.
- Define a go/no-go review; no savings or ROI are assumed in this proposal.
- Agree how the output changes an operational decision supporting the stated business objective, and evaluate that workflow with its owner.

## 6. Key risks

{risk_text}

## 7. Suggested POC workflow

1. Confirm the business objective, population, outcome definition, horizon, and prediction timing.
2. Validate dataset lineage, target construction, feature availability, missingness, and duplicates.
3. Freeze a leakage-safe evaluation design and establish defensible baselines.
4. Evaluate TabPFN, if available, on the same frozen data and compare uncertainty and operational tradeoffs.
5. Review results with business, data, risk, and operational stakeholders.
6. Decide whether to stop, improve data, run a limited operational pilot, or proceed toward production design.

## 8. Stakeholder questions

{questions}
- Who owns the decision and who acts on each prediction?
- What errors are most costly, and what intervention capacity exists?
- How were labels created, and can that definition change over time?
"""


def _analysis_proposal(dataset_name, profile, context, use_case, assessment) -> str:
    guide = CAPABILITIES[use_case.capability]
    facts = "\n".join(f"- {fact}" for fact in assessment.observed_facts)
    risks = "\n".join(f"- {risk}" for risk in assessment.risks)
    questions = "\n".join(f"- {q}" for q in assessment.deployment_questions)
    return f"""# POC Proposal: {use_case.name}

## 1. Business objective and decision

**User-provided objective:** {context.objective or 'Agree the business objective with the stakeholder.'}

**Industry:** {context.industry}

{guide['purpose']} Confirm with the owner how this output supports the stated objective.

## 2. Proposed deliverable

{guide['deliverable']}

Analysis type: {use_case.task_type}. {f'Explanatory outcome: `{use_case.target}`.' if guide['requires_target'] else 'No single prediction target is required.'}

## 3. Dataset and limitations

Dataset: {dataset_name} — {profile.rows:,} rows × {profile.columns} columns.

**User-provided data information:** {context.data_description or 'Not provided; confirm record meaning, coverage, and provenance.'}

{facts}

## 4. Evaluation design

{guide['design']}

**Baseline:** {guide['baseline']}.

**Metrics:** {guide['metrics']}

Fit all transformations on training data and keep evaluation records separate. Confirm temporal and entity dependencies before splitting.

## 5. Success criteria to agree

{guide['success']} Agree with the business owner how the deliverable improves the stated decision; no ROI or performance thresholds are assumed.

## 6. Risks and readiness

Status: {assessment.status.value}.

{risks}

## 7. POC workflow and execution status

1. Confirm the objective, intended output, data meaning, and operational owner.
2. Validate coverage, missingness, duplicates, constraints, and relevant features.
3. Freeze the evaluation protocol and baseline.
4. Run the initial client-based TabPFN evaluation, then validate the full intended workflow.
5. Evaluate outputs against the agreed metrics and conduct stakeholder review.
6. Decide whether to pilot, improve the data, or stop.

**Execution status:** Step 5 can run a bounded initial evaluation through the TabPFN API. Only a measured evaluation section in this proposal confirms that a run occurred. Conditional column models support missing-data recovery, unusual-record ranking, and synthetic sampling; they do not use the extensions' full joint-density runner. Interpretation uses permutation importance and individual sensitivity checks rather than SHAP. These initial checks do not validate this entire POC.

Capability reference: [Official TabPFN extensions](https://github.com/PriorLabs/tabpfn-extensions).

## 8. Stakeholder questions

{questions}
"""
