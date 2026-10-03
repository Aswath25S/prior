from __future__ import annotations

import pandas as pd

from .data_profiler import infer_task_type
from .capabilities import CAPABILITIES
from .schemas import BusinessContext, DatasetProfile, SupportLevel, UseCase


CATALOG = [
    {
        "key": "equipment_failure", "name": "Equipment failure prediction",
        "industries": {"manufacturing"}, "objective": ("failure", "downtime", "maintenance", "machine", "equipment"),
        "targets": ("failure", "breakdown", "fault"), "features": ("vibration", "temperature", "operating", "maintenance", "machine"),
        "statement": "Predict whether equipment will fail within a defined future window.",
        "rationale": "Earlier risk signals could help operations prioritize inspections and maintenance.",
    },
    {
        "key": "demand", "name": "Product demand prediction",
        "industries": {"retail"}, "objective": ("demand", "inventory", "sales", "stock", "forecast"),
        "targets": ("units_sold", "demand", "sales_quantity"), "features": ("price", "promotion", "holiday", "store", "product", "date"),
        "statement": "Estimate future product demand at the relevant product and location level.",
        "rationale": "Demand estimates can inform replenishment, allocation, and inventory decisions.", "forecasting": True,
    },
    {
        "key": "credit_default", "name": "Credit-default prediction",
        "industries": {"financial services", "banking"}, "objective": ("credit", "default", "loan", "risk"),
        "targets": ("default", "delinquent", "charged_off"), "features": ("income", "loan", "debt", "credit", "payment"),
        "statement": "Estimate whether an applicant or account will default within a defined horizon.",
        "rationale": "A calibrated risk signal could support underwriting or portfolio review decisions.",
    },
    {
        "key": "fraud", "name": "Fraud detection",
        "industries": {"financial services", "retail"}, "objective": ("fraud", "abuse", "transaction"),
        "targets": ("fraud", "chargeback"), "features": ("transaction", "merchant", "amount", "device"),
        "statement": "Identify transactions or events requiring fraud review.",
        "rationale": "Risk ranking could focus manual review on higher-risk events.",
    },
    {
        "key": "churn", "name": "Customer churn prediction",
        "industries": set(), "objective": ("churn", "retention", "cancel", "customer"),
        "targets": ("churn", "cancelled", "renewed"), "features": ("customer", "usage", "tenure", "subscription"),
        "statement": "Predict whether a customer will stop using or renewing a service.",
        "rationale": "Churn risk could support targeted retention decisions.",
    },
    {
        "key": "clinical", "name": "Clinical outcome prediction",
        "industries": {"healthcare"}, "objective": ("patient", "clinical", "readmission", "diagnosis"),
        "targets": ("readmitted", "outcome", "diagnosis"), "features": ("patient", "age", "lab", "diagnosis", "treatment"),
        "statement": "Estimate a clearly defined future clinical outcome for an eligible patient population.",
        "rationale": "A validated risk signal might support prioritization, subject to clinical and regulatory review.",
    },
]


def _find_target(columns: list[str], terms: tuple[str, ...]) -> str | None:
    exact = {c.lower(): c for c in columns}
    for term in terms:
        if term in exact:
            return exact[term]
    outcome_hints = ("next", "target", "label", "outcome", "flag", "is_")
    prohibited = ("resolution", "reason", "date", "timestamp")
    matches = [col for col in columns if any(term in col.lower() for term in terms) and not any(x in col.lower() for x in prohibited)]
    matches.sort(key=lambda col: (not any(hint in col.lower() for hint in outcome_hints), columns.index(col)))
    for col in matches:
        if any(term in col.lower() for term in terms):
            return col
    return None


def discover_use_cases(df: pd.DataFrame, profile: DatasetProfile, context: BusinessContext) -> list[UseCase]:
    columns = [str(c) for c in df.columns]
    searchable = " ".join(columns).lower()
    objective = context.objective.lower()
    description = context.data_description.lower()
    industry = context.industry.lower()
    candidates: list[tuple[int, UseCase]] = []

    if context.selected_target and context.selected_target in columns:
        task = infer_task_type(df[context.selected_target], context.selected_target)
        generic = UseCase(
            key="user_defined", name=f"Predict {context.selected_target.replace('_', ' ')}",
            problem_statement=f"Predict `{context.selected_target}` at the stated decision point.",
            target=context.selected_target, task_type=task, support=SupportLevel.CLARIFY,
            business_rationale=context.objective or "The business decision and value should be confirmed with the owner.",
            evidence=["Target explicitly selected by the user."],
            uncertainties=["Confirm that the target is measured after, and all features are available before, prediction time."],
            objective_match=True,
        )
        candidates.append((8, generic))

    for item in CATALOG:
        col_hits = [term for term in item["features"] if term in searchable]
        objective_hits = [term for term in item["objective"] if term in objective]
        target = _find_target(columns, item["targets"])
        industry_hit = industry in item["industries"] if item["industries"] else False
        description_hits = [term for term in item["features"] if term in description]
        score = len(col_hits) + 2 * len(objective_hits) + (3 if target else 0) + (1 if industry_hit else 0) + bool(description_hits)
        if score < 2:
            continue
        if target:
            task = infer_task_type(df[target], target)
            support = SupportLevel.SUPPORTED if task != "unknown" else SupportLevel.CLARIFY
        else:
            task = "to be determined"
            support = SupportLevel.DATA_GAP
        evidence = []
        if target:
            evidence.append(f"A plausible outcome column, `{target}`, is present; its meaning and timing require confirmation.")
        if col_hits:
            evidence.append("Relevant fields observed: " + ", ".join(f"`{x}`" for x in col_hits[:5]) + ".")
        if objective_hits:
            evidence.append("The user-provided objective contains relevant business concepts.")
        if description_hits:
            evidence.append("The user-provided data description references relevant operational concepts; these are not verified facts.")
        uncertainties = ["This is a heuristic candidate, not a claim that the organization has this business problem."]
        if not target:
            uncertainties.append("No defensible historical outcome column was found; outcome data would be required.")
        if item.get("forecasting"):
            uncertainties.append("Forecasting requires chronological validation and confirmation that planned features are known in advance.")
        candidates.append((score, UseCase(
            key=item["key"], name=item["name"], problem_statement=item["statement"],
            target=target, task_type=task, support=support, business_rationale=item["rationale"],
            evidence=evidence, uncertainties=uncertainties, forecasting=bool(item.get("forecasting")),
            objective_match=bool(objective_hits),
        )))

    for capability, guide in CAPABILITIES.items():
        intent = any(term in objective for term in guide["terms"])
        # Describing records as synthetic or incomplete is not itself a request
        # to generate or repair them. Require an explicit analysis intent.
        described = any(phrase in description for phrase in ("goal is", "want to", "need to", "used to")) and any(term in description for term in guide["terms"])
        if not intent and not described:
            continue
        target = None
        if guide["requires_target"]:
            target = context.selected_target or next((c.target for _, c in candidates if c.target), None)
        support = SupportLevel.DATA_GAP if guide["requires_target"] and not target else SupportLevel.CLARIFY
        candidates.append((100 if intent else 10, UseCase(
            key=capability, name=guide["name"], problem_statement=guide["purpose"],
            target=target, task_type=capability.replace("_", " "), support=support,
            business_rationale=f"Proposed contribution to your objective: {context.objective or 'Clarify the intended business decision.'}",
            evidence=["The business objective requests this type of analysis." if intent else "The data description suggests this analysis; confirm that it serves the business objective."],
            uncertainties=[guide["risk"], "Experimental TabPFN extension; available here for scoping, with a separate execution integration required."],
            capability=capability, objective_match=intent,
        )))

    # Deduplicate a generic selected-target candidate when a catalog candidate uses the same target.
    candidates.sort(key=lambda value: value[0], reverse=True)
    result: list[UseCase] = []
    for _, candidate in candidates:
        if any(c.key == candidate.key for c in result):
            continue
        if candidate.key == "user_defined" and any(c.target == candidate.target and c.key != "user_defined" for _, c in candidates):
            continue
        result.append(candidate)
        if len(result) == 3:
            break
    return result
