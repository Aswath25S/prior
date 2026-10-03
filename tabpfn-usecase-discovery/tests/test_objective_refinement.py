import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.data_profiler import profile_dataset
from src.objective_refinement import RefinedObjective, RefinementError, refine_objective, route_brief, _api_error_message
from src.schemas import BusinessContext, SupportLevel


def fixture_data():
    df = pd.DataFrame({'sensor': [i % 7 for i in range(100)], 'failure': [i % 2 for i in range(100)],
                       'private_note': ['secret-row-value'] * 100})
    return df, profile_dataset(df), BusinessContext(objective='Prioritize maintenance', data_description='Machine snapshots')


def brief(**updates):
    fields = dict(objective='Prioritize inspections using equipment risk', business_decision='Which machines to inspect first',
                  route='classification', target='failure', relevant_columns=['sensor'], rationale='Historical failures are recorded.',
                  deliverable='Inspection queue', success_criteria=['Agree acceptable missed failures'], questions=['When are readings available?'])
    fields.update(updates)
    return RefinedObjective(**fields)


def fake_client(output, status='completed'):
    return SimpleNamespace(responses=SimpleNamespace(parse=Mock(return_value=SimpleNamespace(status=status, output_parsed=output))))


def test_request_uses_structured_output_no_rows_and_does_not_execute_tabpfn():
    df, profile, context = fixture_data()
    client = fake_client(brief())
    result = refine_objective(context, profile, api_key='fake-key', client=client)
    kwargs = client.responses.parse.call_args.kwargs
    assert kwargs['store'] is False
    assert kwargs['text_format'] is RefinedObjective
    serialized = json.dumps(kwargs['input'])
    assert 'secret-row-value' not in serialized and 'fake-key' not in serialized
    case = route_brief(result, df, profile)
    assert case.capability == 'prediction' and case.task_type == 'binary classification'
    assert case.deliverable == 'Inspection queue'
    assert case.business_rationale == 'Which machines to inspect first'


@pytest.mark.parametrize('route', ['anomaly_detection', 'imputation', 'synthetic_generation', 'interpretability'])
def test_extension_route_is_preserved(route):
    df, profile, _ = fixture_data()
    case = route_brief(brief(route=route, target='failure' if route == 'interpretability' else None), df, profile)
    assert case.capability == route
    assert case.task_type != 'regression'


@pytest.mark.parametrize('change', [{'target':'invented'}, {'relevant_columns':['invented']}, {'route':'imputation'}])
def test_invalid_references_and_targetless_routes_rejected(change):
    _, profile, context = fixture_data()
    with pytest.raises(RefinementError):
        refine_objective(context, profile, api_key='fake', client=fake_client(brief(**change)))


def test_missing_outcome_remains_a_data_gap_and_clarification_does_not_route():
    df, profile, _ = fixture_data()
    assert route_brief(brief(target=None), df, profile).support == SupportLevel.DATA_GAP
    assert route_brief(brief(route='needs_clarification', target=None), df, profile) is None


def test_refusal_incomplete_and_api_errors_are_handled_without_leaking_details():
    _, profile, context = fixture_data()
    for output, status in [(None, 'completed'), (brief(), 'incomplete')]:
        with pytest.raises(RefinementError):
            refine_objective(context, profile, api_key='fake', client=fake_client(output, status))
    client = fake_client(None)
    client.responses.parse.side_effect = RuntimeError('secret-row-value fake-key')
    with pytest.raises(RefinementError) as exc:
        refine_objective(context, profile, api_key='fake', client=client)
    assert 'secret-row-value' not in str(exc.value) and 'fake-key' not in str(exc.value)


def test_forecast_route_and_non_numeric_regression_validation():
    df, profile, _ = fixture_data()
    assert route_brief(brief(route='forecasting'), df, profile).forecasting
    with pytest.raises(RefinementError):
        route_brief(brief(route='regression', target='private_note'), df, profile)


def test_selected_target_is_respected():
    _, profile, context = fixture_data()
    context.selected_target = 'sensor'
    with pytest.raises(RefinementError, match='differs'):
        refine_objective(context, profile, api_key='fake', client=fake_client(brief()))


@pytest.mark.parametrize('status,code,expected', [
    (401, None, 'authentication failed'),
    (403, None, 'access denied'),
    (429, 'insufficient_quota', 'quota is exhausted'),
    (429, 'rate_limit_exceeded', 'rate or usage limit'),
    (404, None, 'model unavailable'),
    (400, None, 'rejected the request'),
    (503, None, 'server error'),
])
def test_safe_actionable_api_errors(status, code, expected):
    exc = RuntimeError('secret-token-sensitive-server-body')
    exc.status_code = status
    exc.code = code
    message = _api_error_message(exc)
    assert expected in message
    assert 'secret-token' not in message
