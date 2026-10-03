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
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
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
        dates = pd.to_datetime(labeled[date_column], errors="coerce")
        if dates.isna().any():
            raise BenchmarkError(f"`{date_column}` contains dates that could not be parsed.")
        order = np.argsort(dates.to_numpy(), kind="stable")
        labeled = labeled.iloc[order]
        cut = int(len(labeled) * (1 - test_size))
        train = labeled.iloc[:cut]
        test = labeled.iloc[cut:]
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
    return _PreparedBenchmark(plan, X_train, X_test, train[target], test[target])


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
    feature_columns: list[str], access_token: str | None = None,
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

    metrics = _fit_baselines(prepared)
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
    except Exception as exc:
        raise BenchmarkError(f"TabPFN benchmark failed: {exc}") from exc

    return BenchmarkResult(
        plan=prepared.plan,
        metrics=metrics,
        notes=[
            "Metrics are measured on one frozen holdout and are not proof of production performance.",
            "All models used the same locally preprocessed features and holdout rows.",
            "TabPFN inference was performed through the cloud client.",
        ],
    )
