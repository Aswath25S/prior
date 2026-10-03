from scripts.generate_sample_data import manufacturing
from src.data_profiler import profile_dataset
from src.poc_generator import generate_poc_markdown
from src.prioritization import prioritize, recommendation
from src.readiness import assess_readiness
from src.schemas import BenchmarkPlan, BenchmarkResult, BusinessContext
from src.use_case_discovery import discover_use_cases


def pipeline():
    df = manufacturing(rows=600)
    profile = profile_dataset(df)
    context = BusinessContext("Manufacturing", "Reduce downtime", "one machine observation", "weekly before maintenance", "failure_next_30_days")
    cases = discover_use_cases(df, profile, context)
    assessments = {case.key: assess_readiness(df, profile, case) for case in cases}
    return df, profile, context, cases, assessments


def test_prioritization_explains_user_inputs():
    _, _, _, cases, assessments = pipeline()
    ranked = prioritize(cases, assessments, "High", "Medium")
    assert ranked
    assert "high" in ranked[0].recommendation_reason
    assert "Investigate" in recommendation(ranked)


def test_poc_is_dataset_specific_and_has_no_performance_claim():
    _, profile, context, cases, assessments = pipeline()
    case = next(c for c in cases if c.key == "equipment_failure")
    proposal = generate_poc_markdown("Synthetic manufacturing", profile, context, case, assessments[case.key])
    assert "600 rows" in proposal
    assert "`failure_next_30_days`" in proposal
    assert "No TabPFN inference has been run" in proposal
    assert "Success criteria to agree" in proposal
    assert "weekly before maintenance" in proposal


def test_poc_includes_measured_benchmark_when_provided():
    _, profile, context, cases, assessments = pipeline()
    case = next(c for c in cases if c.key == "equipment_failure")
    benchmark = BenchmarkResult(
        BenchmarkPlan(case.target, case.task_type, ["temperature_c"], {}, "Stratified holdout", 480, 120),
        {"Naive baseline": {"ROC-AUC": 0.5}, "TabPFN": {"ROC-AUC": 0.71}},
        ["single holdout"],
    )
    proposal = generate_poc_markdown("Synthetic manufacturing", profile, context, case, assessments[case.key], benchmark=benchmark)
    assert "Measured benchmark" in proposal
    assert "| TabPFN | 0.710 |" in proposal
    assert "single-holdout experimental results" in proposal
