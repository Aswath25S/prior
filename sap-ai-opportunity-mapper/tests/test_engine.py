import copy
import pytest
from src.enterprise_profiles import load_profiles,custom_profile
from src.opportunity_catalog import load_catalog
from src.opportunity_engine import generate,recommendation
from src.models import Challenge
from src.financial_models import estimate
from src.prioritization import normalize_weights,DEFAULT_WEIGHTS
from src.report_generator import proposal,executive_brief
from src.roadmap import build_roadmap

@pytest.fixture
def manufacturer():return load_profiles()[0]

def find(items,id):return next(a for a in items if a.use_case['id']==id)

def test_profiles():
    profiles=load_profiles()
    assert len(profiles)==3
    assert profiles[0].revenue==7e9
    profiles[0].challenges['supplier_delay'].owner='Changed'
    assert load_profiles()[0].challenges['supplier_delay'].owner!='Changed'

def test_matching(manufacturer):
    a=find(generate(manufacturer,load_catalog()),'supplier_delay')
    assert 'supplier reliability' in a.reason
    manufacturer.challenges['supplier_delay'].enabled=False
    assert 'supplier_delay' not in [a.use_case['id'] for a in generate(manufacturer,load_catalog())]

def test_missing_outcomes(manufacturer):
    manufacturer.challenges['supplier_delay'].evidence['target']=False
    a=find(generate(manufacturer,load_catalog()),'supplier_delay')
    assert a.gates and a.status.startswith('Blocked')
    assert a.scores['feasibility']<100
    assert 'supplier_delay'!=generate(manufacturer,load_catalog())[0].use_case['id']

def test_stockout_sensitivity():
    e=load_profiles()[1]
    first=find(generate(e,load_catalog()),'stockout')
    e.challenges['stockout'].financial['unit_cost']*=2
    second=find(generate(e,load_catalog()),'stockout')
    assert second.finance['ceiling']==2*first.finance['ceiling']
    assert second.scores['impact']>first.scores['impact']

def test_incumbent_gate(manufacturer):
    ch=manufacturer.challenges['supplier_delay'];ch.incumbent='Already performs satisfactorily';ch.gap=''
    a=find(generate(manufacturer,load_catalog()),'supplier_delay')
    assert any('incremental' in x for x in a.gates)
    ch.gap='Investigate performance at fixed escalation capacity'
    assert not find(generate(manufacturer,load_catalog()),'supplier_delay').gates

def test_priorities_weights_and_complexity(manufacturer):
    before=find(generate(manufacturer,load_catalog()),'equipment')
    manufacturer.priorities=['equipment uptime']
    after=find(generate(manufacturer,load_catalog()),'equipment')
    assert after.scores['impact']>before.scores['impact'] and after.reason!=before.reason
    impact_only=dict(impact=100,feasibility=0,readiness=0,incremental=0)
    assert find(generate(manufacturer,load_catalog(),impact_only),'equipment').score==after.scores['impact']
    ch=manufacturer.challenges['supplier_delay'];old=find(generate(manufacturer,load_catalog()),'supplier_delay').score
    ch.complexity=5
    assert find(generate(manufacturer,load_catalog()),'supplier_delay').score<old

def test_unknown():
    e=custom_profile();e.challenges={'supplier_delay':Challenge()}
    a=generate(e,load_catalog())[0]
    assert not a.finance['known'] and a.gates
    assert a.scores['feasibility']==0
    assert 'No initial' in recommendation([a])
    assert 'Not enough information' in executive_brief(e,[a])

def test_normalization():
    assert sum(normalize_weights(dict(impact=10,feasibility=20,readiness=5,incremental=15)).values())==100
    assert normalize_weights(dict.fromkeys(DEFAULT_WEIGHTS,0))==DEFAULT_WEIGHTS
    with pytest.raises(ValueError):normalize_weights({'impact':-1})

def test_event_calculation():
    f=estimate('event',dict(volume=1000,rate=.2,unit_cost=100,addressable=.5,adoption=.8,effectiveness=.5))
    assert f['baseline']==20000 and f['ceiling']==10000 and f['base']==4000
    assert f['conservative']<=f['base']<=f['optimistic']<=f['ceiling']
    assert 'net' not in f

def test_payment_carrying_cost():
    f=estimate('payment',dict(volume=1000,rate=.2,unit_cost=1000,days_late=30,days_reduced=10,cost_of_capital=.1,addressable=1,adoption=1,effectiveness=1))
    assert f['baseline']==pytest.approx(200000*30/365*.1)
    assert f['base']==pytest.approx(200000*10/365*.1)

def test_zero_and_missing():
    inputs=dict(volume=0,rate=0,unit_cost=0,addressable=0,adoption=0,effectiveness=0)
    assert estimate('event',inputs)['base']==0
    assert not estimate('event',{})['known']
    inputs['rate']=2
    with pytest.raises(ValueError):estimate('event',inputs)

def test_costs():
    inputs=dict(volume=1000,rate=1,unit_cost=100,addressable=1,adoption=1,effectiveness=1,operating_cost=20000,implementation_cost=40000)
    f=estimate('event',inputs)
    assert f['net']==80000 and f['payback_months']==6
    inputs['operating_cost']=200000
    assert estimate('event',inputs)['payback_months'] is None

def test_outputs_all_accounts():
    for e in load_profiles():
        items=generate(e,load_catalog())
        assert 3<=len(items)<=8
        for a in items:
            doc=proposal(e,a)
            assert '14. Decision after POC' in doc and a.use_case['target'] in doc
            assert 'Success criteria to be agreed' in doc
            assert len(build_roadmap(a))==4
        assert '12. Open questions' in executive_brief(e,items)

def test_timing_action_access_gates(manufacturer):
    ch=manufacturer.challenges['supplier_delay']
    for field in ['timing','features','action','access','history','horizon','structured']:
        original=copy.deepcopy(ch.evidence)
        ch.evidence[field]=None
        assert find(generate(manufacturer,load_catalog()),'supplier_delay').gates
        ch.evidence=original

def test_insufficient_examples_need_validation(manufacturer):
    manufacturer.challenges['supplier_delay'].evidence['enough_examples']=None
    a=find(generate(manufacturer,load_catalog()),'supplier_delay')
    assert a.status=='Needs validation'
    assert 'Enough examples' in a.next_step

def test_recommendation_changes_with_priority_and_weights(manufacturer):
    first=generate(manufacturer,load_catalog())
    assert first[0].use_case['id']=='supplier_delay'
    manufacturer.priorities=['inventory carrying costs']
    manufacturer.challenges['inventory'].severity=5
    weighted=generate(manufacturer,load_catalog(),dict(impact=100,feasibility=0,readiness=0,incremental=0))
    assert weighted[0].use_case['id']=='inventory'
    assert recommendation(weighted)!=recommendation(first)


def test_no_enabled_challenges():
    e=custom_profile()
    assert generate(e,load_catalog())==[]
    assert 'No initial' in recommendation([])
    assert 'Enable a challenge' in executive_brief(e,[])
