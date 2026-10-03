from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SupportLevel(str, Enum):
    SUPPORTED = "Supported candidate"
    CLARIFY = "Needs clarification"
    DATA_GAP = "Data gap"


class ReadinessStatus(str, Enum):
    READY = "Ready for initial evaluation"
    INVESTIGATE = "Needs investigation"
    BLOCKED = "Blocked by a critical data gap"


@dataclass
class DatasetProfile:
    rows: int
    columns: int
    numeric_columns: list[str]
    categorical_columns: list[str]
    datetime_columns: list[str]
    missing_by_column: dict[str, int]
    missing_total: int
    duplicate_rows: int
    identifier_columns: list[str]
    high_cardinality_columns: list[str]
    constant_columns: list[str]
    near_constant_columns: list[str]
    target_candidates: list[str]
    type_by_column: dict[str, str]


@dataclass
class BusinessContext:
    industry: str = "Other"
    objective: str = ""
    row_definition: str = ""
    prediction_timing: str = ""
    selected_target: str | None = None
    data_description: str = ""


@dataclass
class UseCase:
    key: str
    name: str
    problem_statement: str
    target: str | None
    task_type: str
    support: SupportLevel
    business_rationale: str
    evidence: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    forecasting: bool = False
    capability: str = "prediction"
    objective_match: bool = False
    deliverable: str = ""
    success_criteria: list[str] = field(default_factory=list)
    suggested_columns: list[str] = field(default_factory=list)


@dataclass
class ReadinessAssessment:
    status: ReadinessStatus
    observed_facts: list[str]
    risks: list[str]
    actions: list[str]
    leakage_warnings: list[str]
    deployment_questions: list[str]
    target_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class PriorityResult:
    use_case: UseCase
    business_importance: str
    data_feasibility: str
    implementation_complexity: str
    decision_readiness: str
    recommendation_reason: str
    score: int


@dataclass
class BenchmarkPlan:
    target: str
    task_type: str
    feature_columns: list[str]
    excluded_columns: dict[str, str]
    split_strategy: str
    train_rows: int
    test_rows: int


@dataclass
class BenchmarkResult:
    plan: BenchmarkPlan
    metrics: dict[str, dict[str, float]]
    notes: list[str]
