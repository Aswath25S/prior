import pandas as pd

from scripts.generate_sample_data import manufacturing, retail
from src.data_profiler import profile_dataset
from src.readiness import assess_readiness, detect_leakage_warnings
from src.schemas import BusinessContext, ReadinessStatus, SupportLevel, UseCase
from src.use_case_discovery import discover_use_cases


def context(industry, objective, target=None):
    return BusinessContext(industry, objective, "one observation", "before the outcome", target)


def test_manufacturing_discovery_and_imbalance_and_leakage():
    df = manufacturing()
    profile = profile_dataset(df)
    cases = discover_use_cases(df, profile, context("Manufacturing", "reduce downtime and failure", "failure_next_30_days"))
    failure = next(c for c in cases if c.key == "equipment_failure")
    assert failure.target == "failure_next_30_days"
    assert failure.task_type == "binary classification"
    assessment = assess_readiness(df, profile, failure)
    assert assessment.status == ReadinessStatus.INVESTIGATE
    assert assessment.target_summary["minority_rate"] < 0.1
    assert any("imbalanced" in item for item in assessment.risks)
    assert any("failure_resolution_days" in item for item in assessment.leakage_warnings)


def test_retail_discovery_is_regression_and_temporal_warning():
    df = retail(days=40)
    cases = discover_use_cases(df, profile_dataset(df), context("Retail", "forecast demand", "units_sold"))
    demand = next(c for c in cases if c.key == "demand")
    assert demand.task_type == "regression"
    assessment = assess_readiness(df, profile_dataset(df), demand)
    assert any("chronological" in warning.lower() for warning in assessment.leakage_warnings)


def test_retail_daily_sales_objective_is_regression_without_forecasting():
    df = retail(days=40)
    cases = discover_use_cases(df, profile_dataset(df), context("Retail", "Estimate daily units sold using opening-time conditions", "units_sold"))
    demand = next(c for c in cases if c.key == "demand")
    assert demand.task_type == "regression"
    assert not demand.forecasting


def test_no_target_is_honest_data_gap():
    df = manufacturing(rows=300).drop(columns=["failure_next_30_days", "failure_resolution_days"])
    cases = discover_use_cases(df, profile_dataset(df), context("Manufacturing", "reduce machine downtime"))
    failure = next(c for c in cases if c.key == "equipment_failure")
    assert failure.support == SupportLevel.DATA_GAP
    assert assess_readiness(df, profile_dataset(df), failure).status == ReadinessStatus.BLOCKED


def test_constant_invalid_target_is_blocked():
    df = pd.DataFrame({"feature": range(100), "outcome": [1] * 100})
    case = UseCase("x", "X", "Predict X", "outcome", "unknown", SupportLevel.CLARIFY, "test")
    assert assess_readiness(df, profile_dataset(df), case).status == ReadinessStatus.BLOCKED


def test_duplicate_target_and_near_duplicate_leakage():
    df = pd.DataFrame({"target": [0, 1] * 20, "copy": [0, 1] * 20, "almost": [float(x) for x in range(40)]})
    warnings = detect_leakage_warnings(df, "target")
    assert any("duplicates" in warning for warning in warnings)
