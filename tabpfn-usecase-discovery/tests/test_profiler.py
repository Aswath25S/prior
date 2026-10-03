import numpy as np
import pandas as pd

from src.data_profiler import infer_task_type, profile_dataset


def test_profile_types_missing_duplicates_and_targets():
    df = pd.DataFrame({
        "customer_id": ["a", "b", "b"],
        "event_date": ["2025-01-01", "2025-01-02", "2025-01-02"],
        "value": [1.0, np.nan, np.nan],
        "churn_flag": [0, 1, 1],
    })
    profile = profile_dataset(df)
    assert profile.missing_total == 2
    assert profile.duplicate_rows == 1
    assert "event_date" in profile.datetime_columns
    assert "customer_id" in profile.identifier_columns
    assert "churn_flag" in profile.target_candidates


def test_task_inference():
    assert infer_task_type(pd.Series([0, 1, 0, 1]), "failure") == "binary classification"
    assert infer_task_type(pd.Series(range(100)), "units_sold") == "regression"
    assert infer_task_type(pd.Series(["a", "b", "c"]), "outcome") == "multiclass classification"


def test_empty_profile_rejected():
    try:
        profile_dataset(pd.DataFrame())
        assert False
    except ValueError:
        pass
