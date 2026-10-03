import pandas as pd
import pytest

from src.data_profiler import profile_dataset
from src.schemas import BusinessContext, ReadinessStatus
from src.use_case_discovery import discover_use_cases
from src.readiness import assess_readiness
from src.prioritization import prioritize
from src.poc_generator import generate_poc_markdown


@pytest.mark.parametrize('objective,capability,phrase', [
    ('Investigate unusual records to focus review effort', 'anomaly_detection', 'review budget'),
    ('Recover missing readings to improve reporting', 'imputation', 'masking'),
    ('Generate synthetic test data for development', 'synthetic_generation', 'fidelity'),
])
def test_targetless_objectives_have_specific_plans(objective, capability, phrase):
    df = pd.DataFrame({'temperature': [20 + i % 13 for i in range(120)], 'load': [None if i % 9 == 0 else i % 17 for i in range(120)]})
    profile = profile_dataset(df)
    context = BusinessContext(industry='Manufacturing', objective=objective, data_description='Daily sensor readings from one production line.')
    cases = discover_use_cases(df, profile, context)
    case = next(c for c in cases if c.capability == capability)
    assessment = assess_readiness(df, profile, case)
    assert case.target is None
    assert assessment.status != ReadinessStatus.BLOCKED
    ranked = prioritize(cases, {c.key: assess_readiness(df, profile, c) for c in cases})
    assert ranked[0].use_case.capability == capability
    proposal = generate_poc_markdown('Sensor history', profile, context, case, assessment)
    assert objective in proposal and context.data_description in proposal
    assert phrase in proposal
    assert 'runner is not integrated' in proposal
    assert 'No single prediction target is required' in proposal


def test_explanations_need_an_outcome_but_do_not_claim_causality():
    df = pd.DataFrame({'a': [i % 9 for i in range(100)], 'b': [i % 4 for i in range(100)]})
    profile = profile_dataset(df)
    context = BusinessContext(objective='Understand the root cause of quality problems')
    case = discover_use_cases(df, profile, context)[0]
    assessment = assess_readiness(df, profile, case)
    assert case.capability == 'interpretability'
    assert assessment.status == ReadinessStatus.BLOCKED
    assert any('causality' in risk for risk in assessment.risks)


def test_synthetic_provenance_does_not_request_generation():
    df = pd.DataFrame({'a': [i % 9 for i in range(100)], 'b': [i % 4 for i in range(100)]})
    cases = discover_use_cases(df, profile_dataset(df), BusinessContext(objective='Find unusual observations', data_description='These records are synthetic.'))
    assert not any(c.capability == 'synthetic_generation' for c in cases)
