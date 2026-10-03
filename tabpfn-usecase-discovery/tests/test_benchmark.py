import sys
import types

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from scripts.generate_sample_data import manufacturing, retail
from src.benchmark import BenchmarkError, prepare_benchmark, recommended_features, run_tabpfn_benchmark
from src.data_profiler import profile_dataset
from src.readiness import suspected_leakage_columns
from src.schemas import BusinessContext, SupportLevel, UseCase
from src.use_case_discovery import discover_use_cases


def _case(df, industry, objective, target):
    profile = profile_dataset(df)
    context = BusinessContext(industry, objective, "one observation", "before outcome", target)
    return profile, discover_use_cases(df, profile, context)[0]


def test_recommended_features_exclude_target_identifier_and_leakage():
    df = manufacturing(rows=500)
    profile, case = _case(df, "Manufacturing", "reduce failure downtime", "failure_next_30_days")
    leakage = suspected_leakage_columns(df, case.target)
    features, excluded = recommended_features(df, profile, case, leakage)
    assert "failure_next_30_days" not in features
    assert excluded["machine_id"] == "identifier-like field"
    assert excluded["failure_resolution_days"] == "potential post-outcome or leakage field"


def test_classification_plan_is_stratified_and_preprocessed():
    df = manufacturing(rows=500)
    profile, case = _case(df, "Manufacturing", "reduce failure downtime", "failure_next_30_days")
    features, _ = recommended_features(df, profile, case, suspected_leakage_columns(df, case.target))
    prepared = prepare_benchmark(df, profile, case, features)
    assert prepared.plan.split_strategy.startswith("Stratified")
    assert prepared.X_train.shape[0] == 400
    assert prepared.X_test.shape[0] == 100


def test_forecasting_plan_is_chronological():
    df = retail(days=40)
    profile, case = _case(df, "Retail", "forecast demand", "units_sold")
    features, _ = recommended_features(df, profile, case, [])
    assert "store_id" in features and "product_id" in features
    prepared = prepare_benchmark(df, profile, case, features)
    assert prepared.plan.split_strategy.startswith("Chronological")
    assert prepared.plan.train_rows == 640
    assert prepared.plan.test_rows == 160


def test_target_cannot_be_feature():
    df = manufacturing(rows=100)
    profile, case = _case(df, "Manufacturing", "reduce failure downtime", "failure_next_30_days")
    try:
        prepare_benchmark(df, profile, case, ["temperature_c", "failure_next_30_days"])
        assert False
    except BenchmarkError as exc:
        assert "target" in str(exc).lower()


def test_repeated_entities_use_grouped_holdout():
    df = pd.DataFrame({
        "customer_id": [f"C-{i:03d}" for i in range(40) for _ in range(5)],
        "usage": list(range(200)),
        "churn": [i % 2 for i in range(40) for _ in range(5)],
    })
    profile = profile_dataset(df)
    case = UseCase("churn", "Churn", "Predict churn", "churn", "binary classification", SupportLevel.SUPPORTED, "Retention")
    prepared = prepare_benchmark(df, profile, case, ["usage"])
    assert prepared.plan.split_strategy.startswith("Grouped")


def test_tabpfn_runner_reports_real_and_baseline_metrics(monkeypatch):
    df = manufacturing(rows=500)
    profile, case = _case(df, "Manufacturing", "reduce failure downtime", "failure_next_30_days")
    features, _ = recommended_features(df, profile, case, suspected_leakage_columns(df, case.target))
    fake = types.ModuleType("tabpfn_client")
    fake.set_access_token = lambda token: None
    fake.TabPFNClassifier = lambda: LogisticRegression(max_iter=2000)
    fake.TabPFNRegressor = lambda: Ridge()
    monkeypatch.setitem(sys.modules, "tabpfn_client", fake)
    result = run_tabpfn_benchmark(df, profile, case, features, "test-token")
    assert set(result.metrics) == {"Naive baseline", "Logistic regression", "TabPFN"}
    assert "ROC-AUC" in result.metrics["TabPFN"]
    assert result.plan.test_rows == 100
