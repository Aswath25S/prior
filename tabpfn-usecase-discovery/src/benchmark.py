from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, label_binarize

from .schemas import BenchmarkPlan, BenchmarkResult, DatasetProfile, UseCase


class BenchmarkError(ValueError):
    """The requested benchmark cannot be run safely."""


@dataclass
class _PreparedBenchmark:
    plan: BenchmarkPlan
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_test: pd.Series
    test_rows: pd.DataFrame
    preprocessor: ColumnTransformer


def recommended_features(
    df: pd.DataFrame, profile: DatasetProfile, use_case: UseCase, leakage_columns: list[str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    """Return conservative defaults; the user can review and override them in the UI."""
    excluded: dict[str, str] = {}
    leakage = set(leakage_columns or [])
    for column in df.columns:
        name = str(column)
        if name == use_case.target:
            excluded[name] = "prediction target"
        elif name in profile.identifier_columns:
            non_null = df[name].dropna()
            uniqueness = non_null.nunique() / max(len(non_null), 1)
            if uniqueness >= 0.5:
                excluded[name] = "identifier-like field"
        elif name in leakage:
            excluded[name] = "potential post-outcome or leakage field"
        elif name in profile.constant_columns:
            excluded[name] = "constant field"
    included = [str(c) for c in df.columns if str(c) not in excluded]
    return included, excluded


def _build_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = list(frame.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [c for c in frame.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), categorical))
    if not transformers:
        raise BenchmarkError("No usable feature columns remain.")
    return ColumnTransformer(transformers, remainder="drop")


def prepare_benchmark(
    df: pd.DataFrame, profile: DatasetProfile, use_case: UseCase,
    feature_columns: list[str], test_size: float = 0.2, random_state: int = 42,
) -> _PreparedBenchmark:
    target = use_case.target
    if not target or target not in df.columns:
        raise BenchmarkError("A confirmed target is required before benchmarking.")
    if use_case.task_type not in {"binary classification", "multiclass classification", "regression"}:
        raise BenchmarkError(f"Unsupported task type: {use_case.task_type}.")
    if use_case.task_type == "regression" and not pd.api.types.is_numeric_dtype(df[target]):
        raise BenchmarkError("A numeric outcome is needed to estimate a value. Choose category/event analysis for this outcome.")
    features = list(dict.fromkeys(feature_columns))
    if target in features:
        raise BenchmarkError("The target cannot be included as a feature.")
    unknown = [c for c in features if c not in df.columns]
    if unknown:
        raise BenchmarkError("Unknown feature columns: " + ", ".join(unknown))
    if not features:
        raise BenchmarkError("Select at least one feature.")

    group_candidates = []
    if not use_case.forecasting:
        for column in profile.identifier_columns:
            if column == target or column not in df.columns:
                continue
            non_null = df[column].dropna()
            if 1 < non_null.nunique() < len(non_null) * 0.5:
                group_candidates.append(column)
    group_column = group_candidates[0] if group_candidates else None
    required_columns = list(dict.fromkeys(features + [target] + ([group_column] if group_column else [])))
    labeled = df.loc[df[target].notna(), required_columns].copy()
    original_rows = labeled.copy()
    # Preserve time order as a numeric feature instead of treating each date as an unrelated category.
    for date_column in [c for c in profile.datetime_columns if c in features]:
        parsed = pd.to_datetime(labeled[date_column], errors="coerce")
        labeled[date_column] = parsed.map(lambda value: value.toordinal() if pd.notna(value) else np.nan)
    if len(labeled) < 50:
        raise BenchmarkError("At least 50 labeled rows are required for this initial benchmark.")

    split_strategy: str
    if use_case.forecasting:
        date_columns = [c for c in profile.datetime_columns if c in labeled.columns]
        if not date_columns:
            raise BenchmarkError("Forecasting requires a recognized date column for chronological validation.")
        date_column = date_columns[0]
        dates = pd.to_datetime(original_rows.loc[labeled.index, date_column], errors="coerce")
        if dates.isna().any():
            raise BenchmarkError(f"`{date_column}` contains dates that could not be parsed.")
        order = np.argsort(dates.to_numpy(), kind="stable")
        labeled = labeled.iloc[order]
        # Keep all records from a date together: a panel must not cross the time cutoff.
        unique_dates = np.sort(dates.unique())
        if len(unique_dates) < 2:
            raise BenchmarkError("Forecasting needs at least two distinct dates.")
        cutoff = unique_dates[max(1, min(len(unique_dates) - 1, int(len(unique_dates) * (1 - test_size))))]
        train = labeled.loc[dates.loc[labeled.index] < cutoff]
        test = labeled.loc[dates.loc[labeled.index] >= cutoff]
        split_strategy = f"Chronological 80/20 holdout ordered by `{date_column}`"
    elif group_column:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_index, test_index = next(splitter.split(labeled, groups=labeled[group_column]))
        train, test = labeled.iloc[train_index], labeled.iloc[test_index]
        if "classification" in use_case.task_type and set(test[target].unique()) != set(train[target].unique()):
            raise BenchmarkError(f"Grouped holdout by `{group_column}` did not preserve every target class; more entities are needed.")
        split_strategy = f"Grouped 80/20 holdout by `{group_column}` so entities do not cross splits"
    else:
        stratify = labeled[target] if "classification" in use_case.task_type else None
        try:
            train, test = train_test_split(
                labeled, test_size=test_size, random_state=random_state, stratify=stratify,
            )
        except ValueError as exc:
            raise BenchmarkError(f"Could not create a defensible holdout: {exc}") from exc
        split_strategy = "Stratified random 80/20 holdout" if stratify is not None else "Random 80/20 holdout"

    preprocessor = _build_preprocessor(train[features])
    X_train = np.asarray(preprocessor.fit_transform(train[features]), dtype=np.float32)
    X_test = np.asarray(preprocessor.transform(test[features]), dtype=np.float32)
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise BenchmarkError("Preprocessing produced non-finite values.")
    plan = BenchmarkPlan(
        target=target, task_type=use_case.task_type, feature_columns=features,
        excluded_columns={}, split_strategy=split_strategy,
        train_rows=len(train), test_rows=len(test),
    )
    return _PreparedBenchmark(plan, X_train, X_test, train[target], test[target], original_rows.loc[test.index], preprocessor)


def _classification_metrics(y_true: pd.Series, predictions: np.ndarray, probabilities: np.ndarray | None) -> dict[str, float]:
    labels = np.unique(y_true)
    metrics = {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "Balanced accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "F1 (macro)": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
    }
    if len(labels) == 2:
        positive = y_true.value_counts().idxmin()
        metrics.update({
            "Precision (minority class)": float(precision_score(y_true, predictions, pos_label=positive, zero_division=0)),
            "Recall (minority class)": float(recall_score(y_true, predictions, pos_label=positive, zero_division=0)),
        })
        if probabilities is not None:
            positive_index = int(np.where(labels == positive)[0][0])
            scores = probabilities[:, positive_index]
            binary_true = (np.asarray(y_true) == positive).astype(int)
            metrics["ROC-AUC"] = float(roc_auc_score(binary_true, scores))
            metrics["PR-AUC"] = float(average_precision_score(binary_true, scores))
    elif probabilities is not None:
        encoded = label_binarize(y_true, classes=labels)
        metrics["ROC-AUC (macro OVR)"] = float(roc_auc_score(encoded, probabilities, average="macro", multi_class="ovr"))
    return metrics


def _regression_metrics(y_true: pd.Series, predictions: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, predictions))),
        "R²": float(r2_score(y_true, predictions)),
    }


def _fit_baselines(prepared: _PreparedBenchmark) -> dict[str, dict[str, float]]:
    if "classification" in prepared.plan.task_type:
        models = {
            "Naive baseline": DummyClassifier(strategy="prior"),
            "Logistic regression": LogisticRegression(max_iter=2_000, random_state=42),
        }
        output: dict[str, dict[str, float]] = {}
        for name, model in models.items():
            model.fit(prepared.X_train, prepared.y_train)
            probabilities = model.predict_proba(prepared.X_test)
            output[name] = _classification_metrics(prepared.y_test, model.predict(prepared.X_test), probabilities)
        return output
    models = {"Naive baseline": DummyRegressor(strategy="mean"), "Ridge regression": Ridge(alpha=1.0)}
    return {
        name: _regression_metrics(prepared.y_test, model.fit(prepared.X_train, prepared.y_train).predict(prepared.X_test))
        for name, model in models.items()
    }


def run_tabpfn_benchmark(
    df: pd.DataFrame, profile: DatasetProfile, use_case: UseCase,
    feature_columns: list[str], access_token: str | None = None, *, explain: bool = False,
    include_global_importance: bool = True,
) -> BenchmarkResult:
    """Run real cloud inference. Import is lazy so discovery works without this optional dependency."""
    prepared = prepare_benchmark(df, profile, use_case, feature_columns)
    try:
        import tabpfn_client
        from tabpfn_client import TabPFNClassifier, TabPFNRegressor
    except ImportError as exc:
        raise BenchmarkError("Install `tabpfn-client` to run the benchmark.") from exc
    if access_token:
        tabpfn_client.set_access_token(access_token)

    try:
        metrics = _fit_baselines(prepared)
    except Exception:
        raise BenchmarkError("The baseline evaluation could not run. Check the selected fields and outcome values.") from None
    probabilities = None
    try:
        if "classification" in use_case.task_type:
            model = TabPFNClassifier()
            model.fit(prepared.X_train, prepared.y_train)
            predictions = model.predict(prepared.X_test)
            probabilities = model.predict_proba(prepared.X_test)
            metrics["TabPFN"] = _classification_metrics(prepared.y_test, predictions, probabilities)
        else:
            model = TabPFNRegressor()
            model.fit(prepared.X_train, prepared.y_train)
            predictions = model.predict(prepared.X_test)
            metrics["TabPFN"] = _regression_metrics(prepared.y_test, predictions)
    except Exception:
        raise BenchmarkError("TabPFN evaluation failed. Check the token, access, quota, connection, and field types, then retry.") from None

    prediction_rows = pd.DataFrame({"Record": prepared.y_test.index.astype(str),
                                    "Observed": np.asarray(prepared.y_test), "Predicted": predictions})
    for column in feature_columns:
        prediction_rows[f"Input: {column}"] = prepared.test_rows[column].to_numpy()
    for column in profile.identifier_columns:
        prediction_rows[f"Identifier: {column}"] = df.loc[prepared.y_test.index, column].to_numpy()
    if use_case.forecasting:
        date_column = next(c for c in profile.datetime_columns if c in prepared.test_rows)
        prediction_rows["Date"] = pd.to_datetime(prepared.test_rows[date_column]).to_numpy()
        series_fields = [c for c in profile.identifier_columns if c in prepared.test_rows and c != date_column]
        if series_fields:
            prediction_rows["Series"] = prepared.test_rows[series_fields].astype(str).agg(" · ".join, axis=1).to_numpy()
    probability_rows = None
    if probabilities is not None:
        probability_rows = pd.DataFrame(probabilities, columns=[str(c) for c in model.classes_], index=prepared.y_test.index)
    importance, cases = None, None
    notes = [
        "Metrics are measured on one frozen holdout and are not proof of production performance.",
        "All models used the same locally preprocessed features and holdout rows.",
        "TabPFN inference was performed through the cloud client.",
    ]
    if probabilities is None:
        try:
            quantiles = np.asarray(model.predict(prepared.X_test, output_type="quantiles", quantiles=[.1, .9]))
            if quantiles.shape != (2, len(prediction_rows)) or not np.isfinite(quantiles).all():
                raise ValueError("Unexpected prediction intervals")
            prediction_rows["Lower estimate"] = np.minimum(quantiles[0], quantiles[1])
            prediction_rows["Upper estimate"] = np.maximum(quantiles[0], quantiles[1])
            coverage = ((prediction_rows["Observed"] >= prediction_rows["Lower estimate"]) & (prediction_rows["Observed"] <= prediction_rows["Upper estimate"])).mean()
            notes.append(f"The model's 80% prediction range covered {coverage:.1%} of observed holdout values. This is measured coverage, not a guarantee for future records.")
        except Exception:
            notes.append("Point estimates succeeded, but prediction ranges were unavailable for this run.")
    if explain:
        try:
            importance, cases = explain_predictions(model, prepared, include_global_importance=include_global_importance)
        except Exception:
            notes.append("Predictions succeeded, but the extra explanation calls failed. Run again to retry interpretation.")
    return BenchmarkResult(
        plan=prepared.plan,
        metrics=metrics,
        notes=notes, predictions=prediction_rows, probabilities=probability_rows,
        feature_importance=importance, case_explanations=cases,
    )


def explain_predictions(model, prepared: _PreparedBenchmark, *, include_global_importance=True) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """Held-out permutation importance and one-field-at-a-time sensitivity, not SHAP."""
    rng = np.random.default_rng(42)
    count = min(150 if include_global_importance else 5, len(prepared.X_test))
    chosen = np.sort(rng.choice(len(prepared.X_test), count, replace=False))
    X = prepared.X_test[chosen].copy()
    y = np.asarray(prepared.y_test)[chosen]
    classification = "classification" in prepared.plan.task_type
    score = (lambda pred: balanced_accuracy_score(y, pred)) if classification else (lambda pred: -mean_absolute_error(y, pred))
    base_score = score(model.predict(X)) if include_global_importance else None
    # Numeric and categorical preprocessing each preserve one dimension per field.
    numeric = list(prepared.preprocessor.transformers_[0][2]) if prepared.preprocessor.transformers_[0][0] == "numeric" else []
    ordered = numeric + [c for c in prepared.plan.feature_columns if c not in numeric]
    if len(ordered) != X.shape[1]:
        raise BenchmarkError("Cannot map the explanation back to the original fields.")
    if len(ordered) > 12:
        raise BenchmarkError("Select at most 12 fields to run this bounded explanation.")
    importance = []
    for j, name in enumerate(ordered if include_global_importance else []):
        drops = []
        for _ in range(2):
            shuffled = X.copy()
            shuffled[:, j] = rng.permutation(shuffled[:, j])
            drops.append(base_score - score(model.predict(shuffled)))
        importance.append({"Field": name, "Importance": float(np.mean(drops)), "Variation": float(np.std(drops)), "Records checked": count})
    case_X = X[:min(5, count)].copy()
    if classification:
        class_index = int(np.argmin(prepared.y_train.value_counts().reindex(model.classes_).to_numpy()))
        outcome = str(model.classes_[class_index])
        get_output = lambda values: np.asarray(model.predict_proba(values))[:, class_index]
    else:
        outcome = "numeric outcome"
        get_output = lambda values: np.asarray(model.predict(values))
    original = get_output(case_X)
    replacements = []
    for j in range(case_X.shape[1]):
        values = prepared.X_train[:, j]
        if j < len(numeric):
            replacements.append(float(np.median(values)))
        else:
            unique, counts = np.unique(values, return_counts=True)
            replacements.append(float(unique[np.argmax(counts)]))
    sensitivity = []
    for j, name in enumerate(ordered):
        changed = case_X.copy()
        changed[:, j] = replacements[j]
        without = get_output(changed)
        for i in range(len(case_X)):
            sensitivity.append({"Record": str(prepared.y_test.index[chosen[i]]), "Field": name,
                                "Change": float(original[i] - without[i]), "Original output": float(original[i]),
                                "Reference output": float(without[i]), "Explained outcome": outcome})
    importance_frame = pd.DataFrame(importance).sort_values("Importance", ascending=False) if importance else None
    return importance_frame, pd.DataFrame(sensitivity)
