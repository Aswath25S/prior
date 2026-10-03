from src.expert_review import evaluate_expert_review
from src.schemas import (
    ExpertReview, ReadinessAssessment, ReadinessStatus, SupportLevel, UseCase,
)


def assessment(status=ReadinessStatus.READY):
    return ReadinessAssessment(status, ["100 rows"], [], [], [], [])


def prediction():
    return UseCase("x", "Failure risk", "Predict failure", "failure", "binary classification",
                   SupportLevel.SUPPORTED, "Reduce downtime")


def confirmed_review(**changes):
    values = dict(target_meaning="Yes", feature_timing="All are available",
                  missing_values="Expected and understood", representative_history="Yes",
                  actionability="Yes")
    values.update(changes)
    return ExpertReview(**values)


def test_confirmed_expertise_can_mark_clean_data_ready():
    result = evaluate_expert_review(prediction(), assessment(), confirmed_review())
    assert result.status == ReadinessStatus.READY


def test_wrong_target_blocks_progress():
    result = evaluate_expert_review(prediction(), assessment(), confirmed_review(target_meaning="No"))
    assert result.status == ReadinessStatus.BLOCKED
    assert "outcome" in result.headline.lower()


def test_uncertain_answers_generate_plain_actions():
    result = evaluate_expert_review(prediction(), assessment(), ExpertReview())
    assert result.status == ReadinessStatus.INVESTIGATE
    assert any("data owner" in item for item in result.next_steps)


def test_computed_critical_gap_cannot_be_overridden():
    blocked = assessment(ReadinessStatus.BLOCKED)
    blocked.risks = ["No historical labels"]
    result = evaluate_expert_review(prediction(), blocked, confirmed_review())
    assert result.status == ReadinessStatus.BLOCKED
    assert "No historical labels" in result.reasons
