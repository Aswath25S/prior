from pathlib import Path

import pytest
import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest

from scripts import generate_sample_data
from src.benchmark import prepare_benchmark, recommended_features
from src.data_profiler import profile_dataset
from src.objective_refinement import RefinedObjective, route_brief


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ["Manufacturing — equipment failure", "Retail — product demand", "Financial Services — daily card payments (time series)"]


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_catalog_is_available_and_loads(name):
    app = AppTest.from_file(str(ROOT / "app.py")).run(timeout=30)
    assert app.selectbox[0].options == EXAMPLES
    app.selectbox[0].select(name).run()
    next(button for button in app.button if button.label == "Load synthetic example").click().run(timeout=30)
    assert not app.exception
    assert any(name in message.value for message in app.success)
    assert any("Why this dataset was generated:" in item.value for item in app.markdown)
    assert next(field for field in app.text_area if field.label == "Business objective").value == ""
    if "Financial Services" in name:
        assert next(field for field in app.selectbox if field.label == "Industry").value == "Financial Services"


def test_generator_creates_only_catalog_csv_files(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_sample_data, "ROOT", tmp_path)
    generate_sample_data.main()
    assert {path.name for path in (tmp_path / "data/samples").iterdir()} == {
        "synthetic_manufacturing.csv", "synthetic_retail.csv", "synthetic_financial_payments.csv",
    }


def test_retail_daily_sales_and_opening_conditions():
    frame = generate_sample_data.retail()
    pd.testing.assert_frame_equal(frame, generate_sample_data.retail())
    assert frame.shape == (4800, 10)
    assert not frame.duplicated(["store_id", "product_id", "date"]).any()
    assert frame.groupby("date").size().eq(20).all()
    assert frame.date.min() == "2025-01-01"
    assert frame.date.max() == "2025-08-28"
    assert frame.units_sold.ge(0).all()
    assert frame.units_sold.le(frame.opening_stock_units).all()
    assert frame.units_sold.eq(frame.opening_stock_units).any()
    assert "inventory_level" not in frame
    assert frame.expected_store_visitors.gt(0).all()
    assert frame.price.gt(0).all()
    assert set(frame.discount_pct) == {0, 10, 15, 20, 25}


def test_retail_sales_history_uses_only_completed_prior_days():
    frame = generate_sample_data.retail(days=50, stores=2, products=2)
    # Extending the simulation must not change any previously generated record.
    earlier = generate_sample_data.retail(days=20, stores=2, products=2)
    pd.testing.assert_frame_equal(frame.head(len(earlier)), earlier)
    for _, series in frame.groupby(["store_id", "product_id"]):
        sales = series.units_sold
        expected = sales.shift(1).rolling(7).mean().round(1)
        np.testing.assert_allclose(series.avg_units_sold_previous_7_days.iloc[7:], expected.iloc[7:])
        expected_weekday = pd.concat([sales.shift(lag) for lag in (7, 14, 21, 28)], axis=1).mean(axis=1).round(1)
        np.testing.assert_allclose(series.avg_units_sold_same_weekday_previous_4_weeks.iloc[28:], expected_weekday.iloc[28:])
        assert series.avg_units_sold_previous_7_days.iloc[0] > 0


def test_financial_payments_history_and_forecasting_design():
    frame = generate_sample_data.financial_payments()
    pd.testing.assert_frame_equal(frame, generate_sample_data.financial_payments())
    assert frame.shape == (731, 10)
    dates = pd.to_datetime(frame.date)
    assert dates.min() == pd.Timestamp("2024-01-01")
    assert dates.max() == pd.Timestamp("2025-12-31")
    assert dates.tolist() == list(pd.date_range("2024-01-01", "2025-12-31", freq="D"))
    assert frame.transaction_count.gt(0).all()
    np.testing.assert_array_equal(frame.transaction_count_previous_day.iloc[1:], frame.transaction_count.shift(1).iloc[1:])
    np.testing.assert_array_equal(frame.transaction_count_same_weekday_last_week.iloc[7:], frame.transaction_count.shift(7).iloc[7:])
    for window in (7, 28):
        expected = frame.transaction_count.shift(1).rolling(window).mean().round(1)
        np.testing.assert_allclose(frame[f"avg_transaction_count_previous_{window}_days"].iloc[window:], expected.iloc[window:])
    profile = profile_dataset(frame)
    assert "date" not in profile.identifier_columns
    brief = RefinedObjective(
        objective="Forecast the next day's card-payment count", business_decision="Plan capacity and staffing",
        route="forecasting", target="transaction_count", relevant_columns=list(frame.columns[:-1]),
        rationale="Dated daily transaction history", deliverable="Daily count predictions",
        success_criteria=[], questions=[],
    )
    case = route_brief(brief, frame, profile)
    features, excluded = recommended_features(frame, profile, case)
    assert "date" in features
    assert "date" not in excluded
    prepared = prepare_benchmark(frame, profile, case, features)
    assert dates.loc[prepared.y_train.index].max() < dates.loc[prepared.y_test.index].min()
    assert prepared.plan.train_rows == 584
    assert prepared.plan.test_rows == 147
