import sys
import types
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from streamlit.testing.v1 import AppTest

from scripts.generate_sample_data import manufacturing, retail
from src.analysis import run_tabpfn_analysis
from src.benchmark import BenchmarkError, prepare_benchmark, run_tabpfn_benchmark
from src.data_profiler import profile_dataset
from src.schemas import SupportLevel, UseCase
from src.objective_refinement import RefinedObjective


@pytest.fixture
def fake_client(monkeypatch):
    fits = []

    class Regressor(Ridge):
        def fit(self, X, y):
            fits.append(set(y.index))
            self.spread = max(float(np.std(y)), .01)
            return super().fit(X, y)

        def predict(self, X, output_type="mean", quantiles=None):
            values = super().predict(X)
            if output_type == "quantiles":
                return [values + (q - .5) * 2 * self.spread for q in quantiles]
            return values

    class Classifier(LogisticRegression):
        def fit(self, X, y):
            fits.append(set(y.index))
            return super().fit(X, y)

    fake = types.ModuleType("tabpfn_client")
    fake.set_access_token = lambda token: None
    fake.TabPFNClassifier = lambda: Classifier(max_iter=2000)
    fake.TabPFNRegressor = Regressor
    monkeypatch.setitem(sys.modules, "tabpfn_client", fake)
    return fits


def sensor_data():
    rng = np.random.default_rng(9)
    x = rng.normal(20, 2, 300).round(1)
    df = pd.DataFrame({"temperature": x, "load": (2 * x + rng.normal(0, 1, 300)).round(1),
                       "state": np.where(x > 20, "high", "low")})
    df.loc[::13, "load"] = np.nan
    return df


@pytest.mark.parametrize("capability", ["imputation", "anomaly_detection", "synthetic_generation"])
def test_targetless_results_use_reference_rows_only(capability, fake_client):
    df = sensor_data()
    result = run_tabpfn_analysis(df, profile_dataset(df), capability, list(df), generated_rows=40)
    assert not result.metrics.empty
    if capability == "imputation":
        held_out = result.artifacts["original"]
        assert result.artifacts["completed"].notna().all().all()
        flags = result.artifacts["missing_flags"]
        assert flags.equals(held_out.isna())
        unchanged = ~flags
        assert (result.artifacts["completed"].where(unchanged).astype(str) == held_out.where(unchanged).astype(str)).all().all()
    elif capability == "anomaly_detection":
        queue = result.artifacts["queue"]
        held_out = queue
        assert queue["Unusualness"].between(0, 1).all()
        assert queue["Unusualness"].is_monotonic_decreasing
        assert len(result.artifacts["contributions"]) == len(queue)
    else:
        held_out = result.artifacts["observed"]
        generated = result.artifacts["generated"]
        assert generated.shape == (40, 3)
        assert generated.notna().all().all()
        assert set(generated["state"]) <= {"low", "high"}
        assert set(result.artifacts["fidelity"]["Model"]) == {"TabPFN conditional sampler", "Resampled baseline"}
    assert all(not fitted.intersection(held_out.index) for fitted in fake_client)


def test_boolean_synthetic_fields_preserve_type(fake_client):
    df = sensor_data().drop(columns="state")
    df["enabled"] = df["temperature"] > 20
    result = run_tabpfn_analysis(df, profile_dataset(df), "synthetic_generation", list(df), generated_rows=40)
    assert pd.api.types.is_bool_dtype(result.artifacts["generated"]["enabled"])


def test_limits_fail_before_an_api_call(fake_client):
    df = sensor_data()
    with pytest.raises(BenchmarkError, match="between 2 and 8"):
        run_tabpfn_analysis(df, profile_dataset(df), "imputation", ["load"])
    assert not fake_client


def supervised_result(kind):
    if kind == "classification":
        df = manufacturing(rows=500)
        case = UseCase("test", "Failure", "Predict failure", "failure_next_30_days", "binary classification", SupportLevel.SUPPORTED, "Maintenance")
        features = ["temperature_c", "vibration_level", "maintenance_count_90d"]
    else:
        df = retail(days=20, stores=2, products=2)
        case = UseCase("test", "Demand", "Estimate demand", "units_sold", "regression", SupportLevel.SUPPORTED, "Inventory", forecasting=kind == "forecasting")
        features = ["store_id", "product_id", "date", "price", "promotion_flag"]
    return df, profile_dataset(df), case, features


@pytest.mark.parametrize("kind", ["classification", "regression", "forecasting"])
def test_prediction_artifacts_and_interpretation(kind, fake_client):
    df, profile, case, features = supervised_result(kind)
    result = run_tabpfn_benchmark(df, profile, case, features, explain=True)
    assert len(result.predictions) == result.plan.test_rows
    assert set(result.feature_importance["Field"]) == set(features)
    assert result.case_explanations["Record"].nunique() <= 5
    for _, explanation in result.case_explanations.groupby("Record"):
        assert explanation["Base value"].iloc[0] + explanation["SHAP value"].sum() == pytest.approx(explanation["Prediction"].iloc[0])
        assert explanation["Method"].str.contains("SHAP").all()
    if kind == "classification":
        assert result.probabilities.shape == (result.plan.test_rows, 2)
        assert np.allclose(result.probabilities.sum(axis=1), 1)
    else:
        assert "Lower estimate" in result.predictions
    if kind == "forecasting":
        assert "Date" in result.predictions and "Series" in result.predictions


def test_forecast_never_splits_a_date_between_learning_and_evaluation():
    df, profile, case, features = supervised_result("forecasting")
    # An irregular panel would put the row-based cutoff in the middle of a date.
    df = pd.concat([df, df.iloc[-5:]], ignore_index=True)
    prepared = prepare_benchmark(df, profile, case, features)
    train_dates = pd.to_datetime(df.loc[prepared.y_train.index, "date"])
    test_dates = pd.to_datetime(df.loc[prepared.y_test.index, "date"])
    assert train_dates.max() < test_dates.min()


@pytest.mark.parametrize("kind", ["classification", "regression", "forecasting", "imputation", "anomaly_detection", "synthetic_generation"])
def test_result_views_render_measured_artifacts(kind, fake_client, monkeypatch):
    if kind in {"classification", "regression", "forecasting"}:
        df, profile, case, features = supervised_result(kind)
        result = run_tabpfn_benchmark(df, profile, case, features, explain=True)
        view = f"render_benchmark(result, forecasting={kind == 'forecasting'})"
    else:
        df = sensor_data()
        result = run_tabpfn_analysis(df, profile_dataset(df), kind, list(df), generated_rows=40)
        view = "render_analysis(result)"
    module = types.ModuleType("evaluation_test_artifact")
    module.result = result
    monkeypatch.setitem(sys.modules, "evaluation_test_artifact", module)
    app = AppTest.from_string("from evaluation_test_artifact import result\nfrom src.result_views import render_analysis, render_benchmark\n" + view).run(timeout=20)
    assert not app.exception
    assert len(app.get("plotly_chart")) <= 1
    assert len(app.dataframe) == 1
    if kind in {"classification", "regression", "forecasting"}:
        assert len(app.get("plotly_chart")) == 1
        chart = json.loads(app.get("plotly_chart")[0].proto.spec)
        assert chart["data"][0]["type"] == "waterfall"
        assert "Measure to compare" not in [select.label for select in app.selectbox]
        assert "Inspect an individual result" in [select.label for select in app.selectbox]
        assert "Download prediction explanations" in [button.label for button in app.get("download_button")]
    assert "How to interpret this evaluation" in [expander.label for expander in app.expander]
    assert len(app.get("download_button")) >= 1


def test_multiclass_shap_explains_each_records_predicted_class(fake_client):
    rng = np.random.default_rng(11)
    a, b = rng.normal(size=(2, 240)).round(1)
    df = pd.DataFrame({"reading": a, "load": b, "outcome": np.where(a < -.5, "low", np.where(a > .5, "high", "normal"))})
    case = UseCase("multi", "Operating state", "Estimate state", "outcome", "multiclass classification", SupportLevel.SUPPORTED, "Operational review")
    result = run_tabpfn_benchmark(df, profile_dataset(df), case, ["reading", "load"], explain=True, include_global_importance=False)
    for record, explanation in result.case_explanations.groupby("Record"):
        predicted = result.predictions.loc[result.predictions["Record"] == record, "Predicted"].iloc[0]
        assert explanation["Explained outcome"].iloc[0] == str(predicted)
        assert explanation["Prediction"].iloc[0] == pytest.approx(result.probabilities.loc[int(record), str(predicted)])


def click(app, label):
    next(button for button in app.button if button.label == label).click().run(timeout=30)
    assert not app.exception


@pytest.mark.parametrize("route", ["classification", "regression", "forecasting", "interpretability", "imputation", "anomaly_detection", "synthetic_generation"])
def test_step_five_runs_and_exports_results_for_each_route(route, fake_client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("TABPFN_TOKEN", "test")
    targetless = route in {"imputation", "anomaly_detection", "synthetic_generation"}
    target = None if targetless else "units_sold" if route == "forecasting" else "operating_load" if route == "regression" else "failure_next_30_days"
    columns = ["store_id", "product_id", "date", "price", "promotion_flag"] if route == "forecasting" else ["temperature_c", "vibration_level", "maintenance_count_90d"]
    brief = RefinedObjective(objective="Evaluate the proposed operational analysis", business_decision="Support operational review", route=route,
                             target=target, relevant_columns=columns, rationale="Matches the stated goal", deliverable="Reviewable outputs",
                             success_criteria=[], questions=[])
    monkeypatch.setattr("src.objective_refinement.refine_objective", lambda *args, **kwargs: brief)
    app = AppTest.from_file("../app.py").run(timeout=30)
    if route == "forecasting":
        app.selectbox[0].select("Retail — product demand").run()
    click(app, "Load synthetic example")
    app.text_area[0].input("Evaluate the proposed operational analysis").run()
    click(app, "Refine objective")
    click(app, "Use this objective and capability")
    click(app, "Save answers and continue")
    click(app, "Continue to evaluation")
    next(box for box in app.checkbox if "permitted" in box.label).check().run()
    click(app, "Run TabPFN evaluation")
    assert app.session_state["benchmark_result"] is not None
    if not targetless:
        assert app.session_state["benchmark_result"].feature_importance is None
        assert app.session_state["benchmark_result"].case_explanations is not None
    assert any("Evaluation completed" in success.value for success in app.success)
    assert len(app.get("download_button")) >= 1
    # A chart selection should rerun the UI without running the API again.
    calls = len(fake_client)
    app.run()
    assert len(fake_client) == calls
    click(app, "If interested, see official next steps")
    links = {button.label: button.url for button in app.get("link_button")}
    assert links["Documentation overview ↗"] == "https://docs.priorlabs.ai/overview"
    assert links["Quickstart ↗"] == "https://docs.priorlabs.ai/quickstart"
    assert links["Manage API keys ↗"] == "https://platform.priorlabs.ai/account/api-keys"
    assert links["Book a demo with Prior Labs ↗"] == "https://priorlabs.ai/contact"
    assert "Open the recommended capability guide ↗" in links
    fields = next(select for select in app.multiselect if select.label in {"Fields to use", "Fields to analyze"})
    fields.set_value(fields.value[:2]).run(timeout=30)
    assert not app.exception
    assert not any("Evaluation completed" in success.value for success in app.success)
    assert len(fake_client) == calls
