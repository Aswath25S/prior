from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from .schemas import DatasetProfile


TARGET_TERMS = (
    "target", "label", "outcome", "default", "fraud", "churn", "failure",
    "units_sold", "demand", "risk", "readmitted", "response", "revenue",
)
ID_PATTERN = re.compile(r"(^id$|_id$|^id_|uuid|identifier|account_number|serial_number)", re.I)
DATE_PATTERN = re.compile(r"date|time|timestamp|month|year", re.I)


def _looks_datetime(series: pd.Series) -> bool:
    if is_datetime64_any_dtype(series):
        return True
    if is_numeric_dtype(series) or not DATE_PATTERN.search(str(series.name)):
        return False
    sample = series.dropna().astype(str).head(100)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean()) >= 0.8


def profile_dataset(df: pd.DataFrame) -> DatasetProfile:
    if df.empty or len(df.columns) == 0:
        raise ValueError("A non-empty dataset is required for profiling.")

    datetimes = [str(c) for c in df.columns if _looks_datetime(df[c])]
    numeric = [str(c) for c in df.columns if is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])]
    categorical = [str(c) for c in df.columns if str(c) not in numeric and str(c) not in datetimes]
    type_map = {
        str(c): "datetime" if str(c) in datetimes else "numeric" if str(c) in numeric else "categorical"
        for c in df.columns
    }
    identifiers: list[str] = []
    high_cardinality: list[str] = []
    constants: list[str] = []
    near_constants: list[str] = []
    target_candidates: list[str] = []
    rows = len(df)

    for col in df.columns:
        name = str(col)
        series = df[col]
        non_null = series.dropna()
        unique = int(non_null.nunique())
        ratio = unique / max(len(non_null), 1)
        if ID_PATTERN.search(name) or (rows >= 20 and ratio > 0.98 and unique == len(non_null)):
            identifiers.append(name)
        if name in categorical and unique >= 20 and ratio > 0.2:
            high_cardinality.append(name)
        if unique <= 1:
            constants.append(name)
        elif not non_null.empty and float(non_null.value_counts(normalize=True).iloc[0]) >= 0.98:
            near_constants.append(name)
        lowered = name.lower()
        post_event = any(term in lowered for term in ("resolution", "resolved", "reason", "date"))
        if any(term in lowered for term in TARGET_TERMS) and name not in identifiers and not post_event:
            target_candidates.append(name)

    # Preserve column order and prefer conventional outcome-like names.
    target_candidates = list(dict.fromkeys(target_candidates))
    return DatasetProfile(
        rows=rows,
        columns=len(df.columns),
        numeric_columns=numeric,
        categorical_columns=categorical,
        datetime_columns=datetimes,
        missing_by_column={str(k): int(v) for k, v in df.isna().sum().items()},
        missing_total=int(df.isna().sum().sum()),
        duplicate_rows=int(df.duplicated().sum()),
        identifier_columns=identifiers,
        high_cardinality_columns=high_cardinality,
        constant_columns=constants,
        near_constant_columns=near_constants,
        target_candidates=target_candidates,
        type_by_column=type_map,
    )


def infer_task_type(series: pd.Series, name: str = "") -> str:
    values = series.dropna()
    unique = int(values.nunique())
    lowered = name.lower()
    if unique < 2:
        return "unknown"
    if is_bool_dtype(values) or unique == 2:
        return "binary classification"
    if not is_numeric_dtype(values):
        return "multiclass classification"
    if any(k in lowered for k in ("units", "amount", "value", "price", "demand", "revenue", "duration")):
        return "regression"
    if unique <= min(20, max(5, int(len(values) * 0.05))) and np.allclose(values, np.round(values)):
        return "multiclass classification"
    return "regression"
