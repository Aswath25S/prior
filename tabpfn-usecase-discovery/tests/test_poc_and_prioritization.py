from scripts.generate_sample_data import manufacturing
from src.data_profiler import profile_dataset
from src.prioritization import prioritize, recommendation
from src.readiness import assess_readiness
from src.schemas import BusinessContext
from src.use_case_discovery import discover_use_cases


def pipeline():
    df = manufacturing(rows=600)
    profile = profile_dataset(df)
    context = BusinessContext("Manufacturing", "Reduce downtime", "one machine observation", "weekly before maintenance", "failure_next_30_days")
    cases = discover_use_cases(df, profile, context)
    assessments = {case.key: assess_readiness(df, profile, case) for case in cases}
    return df, profile, context, cases, assessments


def test_prioritization_explains_user_inputs():
    _, _, _, cases, assessments = pipeline()
    ranked = prioritize(cases, assessments)
    assert ranked
    assert "objective" in ranked[0].recommendation_reason
    assert "Proceed with" in recommendation(ranked)
