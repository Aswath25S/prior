from pathlib import Path
from streamlit.testing.v1 import AppTest

APP=Path(__file__).resolve().parents[1]/'app.py'

def test_six_screens_and_accounts():
    at=AppTest.from_file(str(APP),default_timeout=20).run()
    assert not at.exception
    for account in ['Northstar Manufacturing','Meridian Retail Group','Harborview Financial','Custom enterprise']:
        at.selectbox(key="account_selector").select(account).run()
        assert not at.exception
        for page in ['Business Landscape','Opportunity Portfolio','Prioritization','Adoption Roadmap','Executive Brief','Enterprise Profile']:
            at.radio[0].set_value(page).run()
            assert not at.exception, str(at.exception)

def test_live_financial_and_data_edits():
    at=AppTest.from_file(str(APP),default_timeout=20).run()
    at.radio[0].set_value('Business Landscape').run()
    # Change the supplier-delay unit cost, then verify recalculation outside the UI.
    prefix=str(at.session_state['revision'])+'_supplier_delay_'
    at.number_input(key=prefix+'f_unit_cost').set_value(1800.).run()
    assert not at.exception
    from src.opportunity_catalog import load_catalog
    from src.opportunity_engine import generate
    supplier=next(a for a in generate(at.session_state['enterprise'],load_catalog()) if a.use_case['id']=='supplier_delay')
    assert supplier.finance['baseline']==80000*.18*1800
    at.selectbox(key=prefix+'target').select('No').run()
    supplier=next(a for a in generate(at.session_state['enterprise'],load_catalog()) if a.use_case['id']=='supplier_delay')
    assert supplier.gates
    at.radio[0].set_value('Opportunity Portfolio').run()
    assert not at.exception
