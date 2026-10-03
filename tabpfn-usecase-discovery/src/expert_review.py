from __future__ import annotations

from .schemas import ExpertReview, ExpertReviewResult, ReadinessAssessment, ReadinessStatus, UseCase


def _plain_risk(risk: str) -> str:
    """Translate automated checks into language a business domain expert can assess."""
    lowered = risk.lower()
    if "imbalance" in lowered or "minority" in lowered:
        return "The outcome of interest is uncommon, so overall accuracy could be misleading."
    if "identifier" in lowered or "unique" in lowered:
        return "Some fields appear to identify individual records and should not be used as evidence."
    if "missing" in lowered and ("target" in lowered or "label" in lowered or "outcome" in lowered):
        return "Some records do not contain the outcome needed for evaluation."
    if "duplicate" in lowered:
        return "The data contains repeated records that could make results look better than they are."
    if "leak" in lowered or "after" in lowered:
        return "Some fields may contain information that would only be known after the outcome."
    if "small" in lowered or "few" in lowered or "row" in lowered:
        return "There may be too little history for a dependable evaluation."
    return "An automated data check found a pattern that needs care during evaluation."


def _plain_action(action: str) -> str:
    lowered = action.lower()
    if "metric" in lowered or "imbalance" in lowered:
        return "Judge the approach by how well it finds the important outcome, not by overall accuracy alone."
    if "identifier" in lowered:
        return "Keep record identifiers out of the analysis."
    if "duplicate" in lowered:
        return "Remove repeated records or keep related records together during evaluation."
    if "leak" in lowered or "exclude" in lowered:
        return "Keep fields that are created after the outcome out of the analysis."
    if "collect" in lowered or "more" in lowered:
        return "Collect more representative history before relying on the result."
    return "Use a controlled holdout evaluation and review the findings with the data owner."


def evaluate_expert_review(
    use_case: UseCase,
    assessment: ReadinessAssessment,
    review: ExpertReview,
) -> ExpertReviewResult:
    """Combine computed checks with answers only a domain expert can provide."""
    reasons: list[str] = []
    next_steps: list[str] = []

    if assessment.status == ReadinessStatus.BLOCKED:
        reasons.extend(assessment.risks or ["The dataset is missing something required for this analysis."])
        next_steps.extend(assessment.actions)
        return ExpertReviewResult(
            ReadinessStatus.BLOCKED,
            "The data is not ready for this evaluation yet",
            reasons,
            next_steps,
        )

    requires_target = use_case.capability == "prediction" or use_case.capability == "interpretability"
    if requires_target and review.target_meaning == "No":
        return ExpertReviewResult(
            ReadinessStatus.BLOCKED,
            "The proposed outcome does not match the business objective",
            ["The selected outcome must represent the event or value the organization actually cares about."],
            ["Choose a more appropriate outcome column or collect the required historical outcome."],
        )

    if requires_target and review.target_meaning == "Not sure":
        reasons.append("The meaning of the proposed outcome still needs confirmation.")
        next_steps.append("Confirm how and when the outcome is recorded with its data owner.")
    if review.feature_timing == "Some are unavailable":
        reasons.append("Some fields may not exist when the result is needed.")
        next_steps.append("Exclude fields that are created after the decision or outcome.")
    elif review.feature_timing == "Not sure":
        reasons.append("Feature availability at decision time is not yet confirmed.")
        next_steps.append("Ask the data owner when each proposed input becomes available.")
    if review.missing_values == "Collection problem":
        reasons.append("Missing values may reflect a data-collection problem.")
        next_steps.append("Investigate the collection process before relying on those fields.")
    elif review.missing_values == "Not sure":
        reasons.append("The reason for missing values is not yet understood.")
        next_steps.append("Ask the data owner why values are missing and whether the pattern is systematic.")
    if review.representative_history == "No":
        reasons.append("The historical data may not reflect the conditions where the analysis will be used.")
        next_steps.append("Add more representative periods, locations, equipment, customers, or products.")
    elif review.representative_history == "Not sure":
        reasons.append("Coverage of expected future conditions is not confirmed.")
        next_steps.append("Confirm that the history covers normal operations and important edge cases.")
    if review.actionability == "No":
        reasons.append("There is no confirmed action or review process for the output.")
        next_steps.append("Define who receives the output and what they can do with it.")
    elif review.actionability == "Not sure":
        reasons.append("The owner and action for the output are not confirmed.")
        next_steps.append("Identify the person or team that will use the result and the decision it supports.")

    if assessment.status == ReadinessStatus.INVESTIGATE:
        reasons.extend(_plain_risk(risk) for risk in assessment.risks)
        next_steps.extend(_plain_action(action) for action in assessment.actions)
        if not reasons:
            reasons.append("The automated review found a data pattern that needs care during evaluation.")
        if not next_steps:
            next_steps.append("Use a controlled holdout evaluation and review the findings with the data owner.")
    if reasons:
        return ExpertReviewResult(
            ReadinessStatus.INVESTIGATE,
            "You can continue, with a few points to validate",
            list(dict.fromkeys(reasons)),
            list(dict.fromkeys(next_steps)),
        )
    return ExpertReviewResult(
        ReadinessStatus.READY,
        "The data appears ready for an initial evaluation",
        ["The required business meaning, timing, coverage, and action have been confirmed."],
        ["Proceed with a controlled evaluation and review the results with the domain owner."],
    )
