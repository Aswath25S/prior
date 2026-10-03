from __future__ import annotations

from .schemas import PriorityResult, ReadinessAssessment, ReadinessStatus, SupportLevel, UseCase


LEVEL_SCORE = {"Low": 0, "Medium": 1, "High": 2}


def prioritize(
    opportunities: list[UseCase], assessments: dict[str, ReadinessAssessment],
    business_priority: str = "Medium", operational_importance: str = "Medium",
) -> list[PriorityResult]:
    results: list[PriorityResult] = []
    for use_case in opportunities:
        assessment = assessments[use_case.key]
        importance_score = LEVEL_SCORE.get(business_priority, 1) + LEVEL_SCORE.get(operational_importance, 1)
        feasibility_score = {ReadinessStatus.READY: 3, ReadinessStatus.INVESTIGATE: 2, ReadinessStatus.BLOCKED: 0}[assessment.status]
        complexity = "High" if use_case.forecasting or use_case.support == SupportLevel.DATA_GAP else "Medium" if assessment.leakage_warnings else "Low"
        complexity_penalty = {"Low": 0, "Medium": 1, "High": 2}[complexity]
        ready = "Sufficient for scoping" if assessment.status != ReadinessStatus.BLOCKED else "Insufficient—resolve the outcome data gap"
        score = importance_score + feasibility_score - complexity_penalty
        reason = (
            f"User-rated priority is {business_priority.lower()} and operational importance is {operational_importance.lower()}; "
            f"data status is {assessment.status.value.lower()}."
        )
        results.append(PriorityResult(use_case, business_priority, assessment.status.value, complexity, ready, reason, score))
    return sorted(results, key=lambda result: result.score, reverse=True)


def recommendation(results: list[PriorityResult]) -> str:
    viable = [r for r in results if r.use_case.support != SupportLevel.DATA_GAP and r.data_feasibility != ReadinessStatus.BLOCKED.value]
    if not viable:
        return "Do not launch a model POC yet. First validate the target definition and obtain usable historical outcomes."
    winner = viable[0]
    return f"Investigate **{winner.use_case.name}** first. {winner.recommendation_reason} Resolve its listed risks before treating it as an approved POC."
