import copy
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.models import Challenge
from src.enterprise_profiles import load_profiles, custom_profile
from src.opportunity_catalog import load_catalog
from src.opportunity_engine import generate, recommendation
from src.feasibility import EVIDENCE
from src.prioritization import DEFAULT_WEIGHTS, normalize_weights
from src.sap_landscape import FUNCTIONS, landscape_rows
from src.roadmap import build_roadmap
from src.report_generator import proposal, executive_brief, financial_text

st.set_page_config(page_title='SAP Enterprise AI Opportunity Mapper',page_icon='◈',layout='wide')
st.markdown('''<style>
html,body,[class*="css"]{font-family:Inter,sans-serif}
.stApp{background:#f7f8fa} [data-testid="stSidebar"]{background:#edf1f5}
h1{letter-spacing:-1.4px!important;color:#142a40!important} h2,h3{color:#20394f!important}
[data-testid="stMetric"]{background:white;border:1px solid #dde4eb;border-radius:10px;padding:16px}
.stButton button,.stDownloadButton button{border-radius:7px}
.block-container{padding-top:2rem;max-width:1450px}
</style>''',unsafe_allow_html=True)
catalog=load_catalog()
profiles=load_profiles()

with st.sidebar:
    st.markdown('### ◈ Opportunity Mapper')
    st.caption('ENTERPRISE AI · STRATEGY WORKSPACE')
    account=st.selectbox('Fictional account',[p.name for p in profiles]+['Custom enterprise'],key='account_selector')
    if st.session_state.get('account')!=account:
        st.session_state.account=account
        st.session_state.enterprise=copy.deepcopy(next((p for p in profiles if p.name==account),custom_profile()))
        st.session_state.revision=st.session_state.get('revision',0)+1
    page=st.radio('Workspace',['Enterprise Profile','Business Landscape','Opportunity Portfolio','Prioritization','Adoption Roadmap','Executive Brief'])
    st.divider()
    st.caption('Independent, unofficial prototype. All accounts and demo assumptions are fictional. No SAP integration, customer data access or model inference.')
    if st.button('Reset this account'):
        del st.session_state.account
        st.rerun()
e=st.session_state.enterprise
rev=st.session_state.revision

def key(name): return f'{rev}_{name}'

def evidence_controls(c,ch):
    st.caption('Unknown is distinct from No. Demo Yes values are assumptions, not verified evidence.')
    cols=st.columns(2)
    for i,(field,label) in enumerate(EVIDENCE.items()):
        value=ch.evidence.get(field)
        options=['Unknown','Yes','No']
        choice=cols[i%2].selectbox(label,options,index=0 if value is None else (1 if value else 2),key=key(c['id']+'_'+field))
        ch.evidence[field]={'Unknown':None,'Yes':True,'No':False}[choice]

def financial_controls(c,ch):
    f=ch.financial
    st.caption('Annual values in USD. Effectiveness is an editable intervention assumption, not TabPFN performance.')
    known=st.checkbox('Enough information for a financial scenario',value=f.get('volume') is not None,key=key(c['id']+'_finance_known'))
    if not known:
        ch.financial={}
        ch.exposure=None
        return
    if not f: f=ch.financial=copy.deepcopy(c['default_financial'])
    use_override=st.checkbox('Use estimated annual exposure instead of volume model',value=ch.exposure is not None,key=key(c['id']+'_override'))
    if use_override:
        label='Annual carrying-cost exposure ($), not outstanding balances' if c['financial_model']=='payment' else 'Annual business-loss exposure ($)'
        ch.exposure=st.number_input(label,min_value=0.,value=float(ch.exposure or 0),step=10000.,key=key(c['id']+'_exposure'))
    else:
        ch.exposure=None
    cols=st.columns(3)
    fields=[('volume','Invoices per year' if c['financial_model']=='payment' else 'Orders / events / decisions per year',1.),('rate','Historical adverse-event rate',.01),('unit_cost','Average invoice balance ($)' if c['financial_model']=='payment' else 'Cost / lost contribution margin per event ($)',100.)]
    if not use_override:
        for i,(field,label,step) in enumerate(fields):
            f[field]=cols[i].number_input(label,min_value=0.,max_value=1. if field=='rate' else None,value=float(f.get(field,0)),step=step,key=key(c['id']+'_f_'+field))
        if c['financial_model']=='payment':
            for i,(field,label,step) in enumerate([('days_late','Average days late',1.),('days_reduced','Assumed days reduced',1.),('cost_of_capital','Annual cost of capital',.01)]):
                f[field]=cols[i].number_input(label,min_value=0.,max_value=1. if field=='cost_of_capital' else None,value=float(f.get(field,0)),step=step,key=key(c['id']+'_f_'+field))
    for i,(field,label) in enumerate([('addressable','Addressable fraction'),('adoption','Operational adoption'),('effectiveness','Intervention effectiveness')]):
        f[field]=cols[i].slider(label,0.,1.,float(f.get(field,.3)),.05,key=key(c['id']+'_f_'+field))
    costs=st.checkbox('Include user-entered operating and implementation costs',value=f.get('operating_cost') is not None,key=key(c['id']+'_costs'))
    if costs:
        f['operating_cost']=cols[0].number_input('Annual operating cost ($)',min_value=0.,value=float(f.get('operating_cost') or 0),step=1000.,key=key(c['id']+'_opex'))
        implementation=cols[1].checkbox('One-time implementation cost known',value=f.get('implementation_cost') is not None,key=key(c['id']+'_capex_known'))
        f['implementation_cost']=cols[2].number_input('Implementation cost ($)',min_value=0.,value=float(f.get('implementation_cost') or 0),step=1000.,key=key(c['id']+'_capex')) if implementation else None
    else:
        f.pop('operating_cost',None); f.pop('implementation_cost',None)

def candidate_select(items,label='Inspect opportunity'):
    return st.selectbox(label,items,format_func=lambda a:a.use_case['name'],key=key(page+'_selected'))

def show_finance(a):
    f=a.finance
    if not f['known']: st.info(f['reason']); return
    cols=st.columns(4)
    for col,label,field in zip(cols,['Baseline exposure','Conservative','Base scenario','Optimistic'],['baseline','conservative','base','optimistic']):
        col.metric(label,f"${f[field]:,.0f}")
    st.caption(f"Addressable annual ceiling: ${f['ceiling']:,.0f}. Scenarios are unvalidated. Conservative = 50% of base; optimistic increases adoption 20% and effectiveness 30%, each capped at 100%.")
    st.write(financial_text(a))
    with st.expander('Calculation, assumptions and sensitivity'):
        st.write('Event exposure = annual volume × adverse-event rate × cost per event. Payment exposure = invoices × late-payment rate × average balance × days late / 365 × annual cost of capital. Payment ceiling also scales by the fraction of late days assumed recoverable. An exposure override is already an annual cost estimate. Base benefit = ceiling × adoption × effectiveness.')
        st.json({'inputs':f['assumptions'],'exposure_override':f.get('exposure_override')})
        st.dataframe(pd.DataFrame({'Scenario':list(f['sensitivity']),'Annual benefit':list(f['sensitivity'].values())}),hide_index=True,use_container_width=True)

def detail(a):
    c=a.use_case
    st.subheader(c['name'])
    st.write(a.reason)
    st.info(a.status)
    left,right=st.columns(2)
    left.markdown('**Business decision**'); left.write(c['action'])
    left.markdown('**Prediction target & timing**'); left.write(c['target']+' · '+c['prediction_type']); left.write(c['timing'])
    right.markdown('**Required historical data**'); right.write(', '.join(c['required_data']))
    right.markdown('**Feature categories**'); right.write(', '.join(c['features']))
    right.markdown('**Possible SAP process context**'); right.write(c['sap_context']+'. Customer-specific access and schema must be confirmed.')
    st.markdown('**Existing solution / incremental value**')
    st.write(a.challenge.incumbent+' · '+(a.challenge.gap or 'Incremental-value hypothesis has not been established.'))
    st.write('Baseline: '+c['baseline'])
    if a.gates: st.warning('Required before technical evaluation: '+'; '.join(a.gates))
    st.write('**Next validation step:** '+a.next_step)
    show_finance(a)
    with st.expander('Technical risks, metrics and deployment questions'):
        st.write('Metrics: '+c['metrics']); st.write('; '.join(c['data_risks'])); st.write('; '.join(c['deployment_questions']))

st.caption('STRATEGY / '+page.upper())
st.title(page)
st.write(f'{e.name} · Fictional enterprise assessment')

if page=='Enterprise Profile':
    overview=st.columns(3)
    overview[0].metric('Annual revenue · fictional',f'${e.revenue/1e9:,.1f}bn')
    overview[1].metric('Employees',f'{e.employees:,}')
    overview[2].metric('Declared SAP products',len(e.sap_products))
    st.write('**Strategic priorities:** '+('; '.join(e.priorities) or 'Not defined'))
    st.write('**Operational challenges:** '+('; '.join(c['pain'] for c in catalog if c['id'] in e.challenges and e.challenges[c['id']].enabled) or 'Select challenges in Business Landscape'))
    st.caption('SAP landscape: '+(', '.join(e.sap_products) or 'Unknown'))
    st.markdown('#### Define the account and the decisions that matter')
    with st.form(key('profile')):
        cols=st.columns(2)
        name=cols[0].text_input('Company name',e.name)
        inds=['Industrial manufacturing','Retail','Financial services','Other']
        industry=cols[1].selectbox('Industry',inds,index=inds.index(e.industry) if e.industry in inds else 3)
        revenue=cols[0].number_input('Annual revenue ($)',min_value=0.,value=float(e.revenue),step=100000000.)
        employees=cols[1].number_input('Employees',min_value=0,value=e.employees)
        geography=cols[0].text_input('Geography',e.geography)
        sap=cols[1].multiselect('Declared SAP products',['S/4HANA','Ariba','IBP','Sales Cloud','BTP'],default=e.sap_products)
        priorities=st.text_area('Business priorities (one per line)','\n'.join(e.priorities))
        ml=st.text_input('Existing ML capabilities',e.ml_capabilities)
        data=st.text_area('Data availability assumptions',e.data_assumptions)
        resources=st.text_input('Available implementation resources',e.resources)
        if st.form_submit_button('Save enterprise profile',type='primary'):
            e.name=name;e.industry=industry;e.revenue=revenue;e.employees=employees;e.geography=geography;e.sap_products=sap;e.priorities=[p.strip() for p in priorities.splitlines() if p.strip()];e.ml_capabilities=ml;e.data_assumptions=data;e.resources=resources
            st.rerun()
    st.caption('Company size provides account context; it does not estimate TabPFN revenue. Select and edit challenges in Business Landscape.')

elif page=='Business Landscape':
    st.write('Select operational challenges, then make the evidence and value assumptions explicit.')
    for function in FUNCTIONS:
        available=[c for c in catalog if c['function']==function and (e.industry in c['industries'] or e.industry=='Other' or c['id'] in e.challenges)]
        if not available: continue
        st.subheader(function)
        for c in available:
            existing=e.challenges.get(c['id'])
            with st.expander(c['pain']+' → '+c['name'],expanded=False):
                enabled=st.checkbox('Include this challenge',value=bool(existing and existing.enabled),key=key(c['id']+'_enabled'))
                if existing is None and enabled:
                    existing=e.challenges[c['id']]=Challenge()
                if existing: existing.enabled=enabled
                st.write(c['problem']); st.write('Possible decision: '+c['action'])
                st.caption('Possible SAP context: '+c['sap_context']+'; availability and schema must be confirmed.')
                if not enabled: continue
                ch=existing
                tabs=st.tabs(['Business & incumbent','Data & technical evidence','Financial scenario'])
                with tabs[0]:
                    ch.severity=st.slider('Pain-point severity',1,5,ch.severity,key=key(c['id']+'_severity'))
                    ch.owner=st.text_input('Business owner role',ch.owner,key=key(c['id']+'_owner'))
                    ch.complexity=st.slider('Implementation complexity (sources, dependencies, integration, security, workflow)',1,5,ch.complexity,key=key(c['id']+'_complexity'))
                    statuses=['Known gap requiring investigation','Existing capability; incremental value unproven','Already performs satisfactorily','No incumbent identified','Information missing']
                    ch.incumbent=st.selectbox('Existing solution status',statuses,index=statuses.index(ch.incumbent),key=key(c['id']+'_incumbent'))
                    ch.gap=st.text_area('Explicit incremental-value hypothesis (clear this if no credible gap)',ch.gap,key=key(c['id']+'_gap'))
                    st.caption(c['overlap'])
                with tabs[1]: evidence_controls(c,ch)
                with tabs[2]: financial_controls(c,ch)
    assessments=generate(e,catalog,st.session_state.get('weights'))
    st.subheader('Operational map')
    st.dataframe(pd.DataFrame(landscape_rows(assessments)),hide_index=True,use_container_width=True)

else:
    weights=st.session_state.get('weights',DEFAULT_WEIGHTS.copy())
    if page=='Prioritization':
        st.write('Weights are illustrative. Hard decision gates determine eligibility before the provisional score.')
        cols=st.columns(4)
        weights={k:cols[i].number_input(label,min_value=0,max_value=100,value=int(weights[k]),key=key('weight_'+k)) for i,(k,label) in enumerate(zip(DEFAULT_WEIGHTS,['Business impact','Data / technical feasibility','Implementation / adoption','Incremental value']))}
        st.session_state.weights=weights
        st.caption('Normalized weights: '+', '.join(f'{k} {v:.0f}%' for k,v in normalize_weights(weights).items())+('. All-zero inputs use default weights.' if sum(weights.values())==0 else ''))
    assessments=generate(e,catalog,weights)
    if not assessments:
        st.info('No enabled challenges. Add operational challenges in Business Landscape to generate opportunities.')
        st.stop()
    if page in ['Opportunity Portfolio','Prioritization']:
        ready=sum(a.status=='Feasible candidate for discovery/POC planning' for a in assessments)
        cols=st.columns(3)
        cols[0].metric('Relevant opportunities',len(assessments));cols[1].metric('Discovery / planning candidates',ready);cols[2].metric('Require hard-gate validation',sum(bool(a.gates) for a in assessments))
        st.info(recommendation(assessments))
        if page=='Opportunity Portfolio':
            fig=go.Figure()
            for blocked,color,symbol in [(False,'#147d92','circle'),(True,'#a67437','diamond-open')]:
                items=[a for a in assessments if bool(a.gates)==blocked]
                if items:
                    fig.add_trace(go.Scatter(x=[a.scores['feasibility'] for a in items],y=[a.scores['impact'] for a in items],mode='markers+text',text=[a.use_case['name'] for a in items],textposition='top center',customdata=[a.use_case['id'] for a in items],marker=dict(size=17,color=color,symbol=symbol),name='Provisional / gated' if blocked else 'Planning candidate',hovertemplate='%{text}<br>Feasibility %{x:.0f}/100<br>Impact %{y:.0f}/100<extra></extra>'))
            fig.add_hline(y=60,line_dash='dot',line_color='#b4bfcb');fig.add_vline(x=60,line_dash='dot',line_color='#b4bfcb')
            for x,y,label in [(25,94,'Strategic · groundwork'),(78,94,'Initial POC candidates*'),(25,6,'Defer'),(78,6,'Optional experiments')]:
                fig.add_annotation(x=x,y=y,text=label,showarrow=False,font=dict(color='#7a8998',size=12))
            fig.update_layout(height=460,paper_bgcolor='#f7f8fa',plot_bgcolor='white',margin=dict(l=10,r=10,t=15,b=20),xaxis=dict(title='Feasibility / ease of evaluation (illustrative)',range=[-5,110]),yaxis=dict(title='Potential business impact (illustrative)',range=[-5,105]),legend=dict(orientation='h',y=-.22))
            st.plotly_chart(fig,use_container_width=True)
            st.caption('*Quadrants guide discussion. Hard gates still apply. Select below to inspect evidence and assumptions; all positions remain provisional.')
        st.dataframe(pd.DataFrame([{'Opportunity':a.use_case['name'],'Score':round(a.score),'Impact':round(a.scores['impact']),'Feasibility':round(a.scores['feasibility']),'Readiness':round(a.scores['readiness']),'Incremental':round(a.scores['incremental']),'Status':a.status,'Next step':a.next_step} for a in assessments[:8]]),hide_index=True,use_container_width=True)
        with st.expander('How scores and tradeoffs work'):
            st.write('Impact = 55% scenario value + 30% severity + 15% strategic priority alignment. Scenario value uses a capped log scale ($10k → 0; $10m → 100). Unknown benefit scores zero for that component. Feasibility averages data and technical evidence; unknown and No contribute zero but remain separately labeled. Readiness combines owner/sponsor/KPI/action evidence with inverse implementation complexity. Incremental value = 70 for an explicit hypothesis, 40 for no identified incumbent, otherwise 0. Hypotheses require validation. Scores are illustrative, not probability estimates.')
        a=candidate_select(assessments)
        detail(a)
        st.caption('What would move this position? Update exposure, severity or priorities for impact; validate outcome labels, access and point-in-time features for feasibility; reduce complexity and confirm ownership for readiness.')
        if page=='Prioritization':
            for label,items in [('Initial evaluation',[x for x in assessments if not x.gates and x.status.startswith('Feasible')]),('Groundwork / data preparation',[x for x in assessments if x.gates]),('Needs further validation',[x for x in assessments if not x.gates and not x.status.startswith('Feasible')])]:
                st.markdown('**'+label+'**')
                st.write('; '.join(x.use_case['name']+' — '+x.next_step for x in items) or 'None currently')
    elif page=='Adoption Roadmap':
        a=candidate_select(assessments)
        if a.gates: st.warning('Discovery first. Technical evaluation is conditional on resolving: '+'; '.join(a.gates))
        st.caption('Illustrative 90-day phases, not a guaranteed delivery schedule. Missing outcomes may require a longer collection period.')
        for r in build_roadmap(a):
            with st.container(border=True):
                st.subheader(r['weeks']+' · '+r['phase']);st.write(r['objective'])
                for activity in r['activities']: st.markdown('- '+activity)
                st.caption('Owner: '+r['owner']);st.write('**Deliverable:** '+r['deliverable']);st.write('**Dependencies:** '+r['dependencies']);st.write('**Exit criteria:** '+r['exit'])
        success=st.text_area('Success criteria to agree with stakeholders','Agree minimum useful precision / recall or forecast error, maximum intervention burden, and minimum incremental value against the incumbent.',key=key('success'))
        poc=proposal(e,a,success)
        st.download_button('Download POC proposal',poc,file_name=a.use_case['id']+'_poc_proposal.md',mime='text/markdown',type='primary')
        with st.expander('Preview full POC proposal'):st.markdown(poc)
    elif page=='Executive Brief':
        st.info(recommendation(assessments))
        a=candidate_select(assessments,'Roadmap opportunity for the brief')
        report=executive_brief(e,assessments,weights,a)
        st.download_button('Download executive brief',report,file_name='enterprise_ai_assessment.md',mime='text/markdown',type='primary')
        st.download_button('Download selected POC proposal',proposal(e,a),file_name=a.use_case['id']+'_poc_proposal.md',mime='text/markdown')
        st.markdown(report)
