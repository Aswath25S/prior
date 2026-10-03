from __future__ import annotations

from .schemas import PriorityResult, ReadinessAssessment, ReadinessStatus, SupportLevel, UseCase


def prioritize(
    opportunities: list[UseCase], assessments: dict[str, ReadinessAssessment],
) -> list[PriorityResult]:
    results: list[PriorityResult] = []
    for use_case in opportunities:
        assessment = assessments[use_case.key]
        importance_score = 4 if use_case.objective_match else 0
        feasibility_score = {ReadinessStatus.READY: 3, ReadinessStatus.INVESTIGATE: 2, ReadinessStatus.BLOCKED: 0}[assessment.status]
        complexity = "High" if use_case.capability != "prediction" or use_case.forecasting or use_case.support == SupportLevel.DATA_GAP else "Medium" if assessment.leakage_warnings else "Low"
        complexity_penalty = {"Low": 0, "Medium": 1, "High": 2}[complexity]
        ready = "Sufficient for scoping" if assessment.status != ReadinessStatus.BLOCKED else "Insufficient—resolve data gaps"
        score = importance_score + feasibility_score - complexity_penalty
        reason = (
            f"{'Matches the stated objective' if use_case.objective_match else 'Suggested by the available data; business relevance needs confirmation'}; "
            f"data status is {assessment.status.value.lower()}; implementation complexity is {complexity.lower()}. "
            "Objective alignment is a heuristic, not a measured business-value score."
        )
        results.append(PriorityResult(use_case, "Objective aligned" if use_case.objective_match else "Needs confirmation", assessment.status.value, complexity, ready, reason, score))
    return sorted(results, key=lambda result: result.score, reverse=True)


def recommendation(results: list[PriorityResult]) -> str:
    viable = [r for r in results if r.use_case.support != SupportLevel.DATA_GAP and r.data_feasibility != ReadinessStatus.BLOCKED.value]
    if not viable:
        return "First resolve the listed data gaps and confirm the intended analysis before launching a POC."
    winner = viable[0]
    return f"Investigate **{winner.use_case.name}** first. {winner.recommendation_reason} Resolve its listed risks before treating it as an approved POC."
