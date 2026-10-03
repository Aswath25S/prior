from __future__ import annotations

import re

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .schemas import DatasetProfile, ReadinessAssessment, ReadinessStatus, UseCase
from .capabilities import CAPABILITIES


POST_EVENT_TERMS = ("resolution", "resolved", "after", "post_", "close_date", "closed_at", "actual_outcome", "failure_date")


def suspected_leakage_columns(df: pd.DataFrame, target: str | None) -> list[str]:
    if not target or target not in df.columns:
        return []
    suspected: list[str] = []
    target_series = df[target]
    target_tokens = {token for token in re.split(r"[_\W]+", target.lower()) if len(token) > 3}
    for col in df.columns:
        if col == target:
            continue
        lowered = str(col).lower()
        if any(term in lowered for term in POST_EVENT_TERMS) or (target_tokens and any(t in lowered for t in target_tokens) and any(x in lowered for x in ("date", "status", "reason"))):
            suspected.append(str(col))
            continue
        pair = pd.concat([target_series, df[col]], axis=1).dropna()
        if len(pair) >= 20 and pair.iloc[:, 0].nunique() > 1:
            if pair.iloc[:, 1].equals(pair.iloc[:, 0]):
                suspected.append(str(col))
            elif is_numeric_dtype(pair.iloc[:, 0]) and is_numeric_dtype(pair.iloc[:, 1]):
                corr = pair.corr(numeric_only=True).iloc[0, 1]
                if np.isfinite(corr) and abs(corr) > 0.995:
                    suspected.append(str(col))
    return list(dict.fromkeys(suspected))


def detect_leakage_warnings(df: pd.DataFrame, target: str | None, forecasting: bool = False) -> list[str]:
    warnings: list[str] = []
    if not target or target not in df.columns:
        return warnings
    target_tokens = {token for token in re.split(r"[_\W]+", target.lower()) if len(token) > 3}
    for col in suspected_leakage_columns(df, target):
        lowered = str(col).lower()
        if any(term in lowered for term in POST_EVENT_TERMS) or (target_tokens and any(t in lowered for t in target_tokens) and any(x in lowered for x in ("date", "status", "reason"))):
            warnings.append(f"`{col}` may contain post-outcome information; confirm its availability at prediction time.")
            continue
        pair = pd.concat([df[target], df[col]], axis=1).dropna()
        if pair.iloc[:, 1].equals(pair.iloc[:, 0]):
            warnings.append(f"`{col}` duplicates the target and should be excluded.")
        else:
            warnings.append(f"`{col}` is almost perfectly correlated with the target; investigate possible leakage.")
    if forecasting:
        warnings.append("A random split would leak future patterns; use a chronological holdout and backtesting.")
    return list(dict.fromkeys(warnings))


def assess_readiness(df: pd.DataFrame, profile: DatasetProfile, use_case: UseCase) -> ReadinessAssessment:
    if use_case.capability != "prediction":
        guide = CAPABILITIES[use_case.capability]
        usable = [c for c in df if c not in profile.identifier_columns and c not in profile.constant_columns]
        blocked = len(usable) < 2 or profile.rows < 50
        risks = [guide["risk"]]
        if guide["requires_target"] and (not use_case.target or use_case.target not in df or df[use_case.target].dropna().nunique() < 2):
            blocked = True
            risks.append("A varying outcome is required to explain a supervised model.")
        if len(usable) < 2 or profile.rows < 50:
            risks.append("This initial evaluation needs at least 50 records and two informative fields.")
        if use_case.capability == "imputation" and not profile.missing_total:
            risks.append("No missing cells were observed. Confirm a future missing-data scenario before proceeding.")
        return ReadinessAssessment(
            ReadinessStatus.BLOCKED if blocked else ReadinessStatus.INVESTIGATE,
            [f"Dataset contains {profile.rows:,} rows and {profile.columns} columns.",
             f"Observed {profile.missing_total:,} missing cells and {profile.duplicate_rows:,} duplicate rows.",
             "A fixed outcome label is not required for this capability." if not guide["requires_target"] else f"Proposed explanatory outcome: {use_case.target or 'not specified'}"],
            risks, [guide["design"], "Confirm the intended action and agree on evaluation criteria.", "Review the initial evaluation method and validate it against the full intended workflow."],
            [], ["Who will use the output, and what decision will it support?", guide["success"],
                 "Which columns and records are appropriate for this analysis?"],
        )
    facts = [f"Dataset contains {profile.rows:,} rows and {profile.columns} columns."]
    risks: list[str] = []
    actions: list[str] = []
    summary: dict[str, object] = {}
    target = use_case.target

    if not target or target not in df.columns:
        return ReadinessAssessment(
            status=ReadinessStatus.BLOCKED,
            observed_facts=facts + ["No usable outcome column is available for this candidate."],
            risks=["A supervised technical evaluation cannot measure performance without historical labels."],
            actions=["Define the outcome, prediction horizon, and label-generation process; then collect historical outcomes."],
            leakage_warnings=[], deployment_questions=_deployment_questions(), target_summary=summary,
        )

    values = df[target]
    non_null = values.dropna()
    missing = int(values.isna().sum())
    missing_pct = missing / len(values)
    classes = int(non_null.nunique())
    facts.append(f"Target `{target}` has {missing:,} missing values ({missing_pct:.1%}).")
    facts.append(f"Target contains {classes:,} distinct observed value(s).")
    summary.update({"target": target, "missing_count": missing, "missing_rate": missing_pct, "unique_values": classes, "task_type": use_case.task_type})

    critical = False
    if classes < 2:
        critical = True
        risks.append("The target has fewer than two observed values, so a predictive evaluation is not possible.")
        actions.append("Collect examples covering multiple target outcomes.")
    if len(non_null) < 50:
        risks.append("Fewer than 50 labeled rows are available; results would be highly unstable.")
        actions.append("Collect more labeled examples before drawing performance conclusions.")
    elif len(non_null) < 200:
        risks.append("The labeled sample is small; use uncertainty intervals and treat findings as directional.")
    if missing_pct > 0.5:
        risks.append("More than half of target labels are missing.")
        actions.append("Investigate why labels are missing and whether missingness is systematic.")
    elif missing_pct > 0:
        risks.append("Missing labels require an explicit exclusion or label-recovery policy.")

    if "classification" in use_case.task_type and classes >= 2:
        proportions = non_null.value_counts(normalize=True)
        minority = float(proportions.min())
        summary["class_distribution"] = {str(k): int(v) for k, v in non_null.value_counts().items()}
        summary["minority_rate"] = minority
        facts.append(f"Smallest class represents {minority:.1%} of labeled rows.")
        if minority < 0.1:
            risks.append("The target is materially imbalanced; accuracy alone would be misleading.")
            actions.append("Use stratified evaluation and review PR-AUC, recall, precision, and error costs.")
    if use_case.task_type == "regression" and classes / max(len(non_null), 1) < 0.02:
        risks.append("The numeric outcome has few distinct values; confirm whether regression is the intended framing.")

    if profile.duplicate_rows:
        risks.append(f"{profile.duplicate_rows:,} duplicate rows could bias evaluation or cross split boundaries.")
        actions.append("Resolve duplicates before splitting data.")
    if profile.identifier_columns:
        risks.append("Identifier-like fields may memorize entities instead of generalizing.")
    high_missing = [c for c, n in profile.missing_by_column.items() if n / profile.rows > 0.3 and c != target]
    if high_missing:
        risks.append("Features above 30% missingness: " + ", ".join(high_missing[:5]) + ".")
    if profile.constant_columns:
        risks.append("Constant columns add no predictive information: " + ", ".join(profile.constant_columns[:5]) + ".")

    leakage = detect_leakage_warnings(df, target, use_case.forecasting)
    actions.extend(["Verify every feature existed at the stated prediction time."] if leakage else [])
    if use_case.forecasting:
        actions.append("Sort by date and reserve the latest period for evaluation; consider rolling backtests.")
    status = ReadinessStatus.BLOCKED if critical else ReadinessStatus.INVESTIGATE if (risks or leakage or not use_case.target) else ReadinessStatus.READY
    if not actions and status == ReadinessStatus.READY:
        actions.append("Confirm target semantics and establish a leakage-safe baseline evaluation.")
    return ReadinessAssessment(status, facts, risks, list(dict.fromkeys(actions)), leakage, _deployment_questions(), summary)


def _deployment_questions() -> list[str]:
    return [
        "Are all proposed features available when the prediction will be made?",
        "Is the target definition stable and consistently recorded?",
        "Does this history represent expected production conditions?",
        "Are there privacy, regulatory, or data-access restrictions?",
        "How frequently are predictions needed, and what action follows each prediction?",
    ]
