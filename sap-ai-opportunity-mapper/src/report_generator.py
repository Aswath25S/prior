from .roadmap import build_roadmap
from .opportunity_engine import recommendation
from .prioritization import normalize_weights, DEFAULT_WEIGHTS

def money(value): return f'${value:,.0f}'

def financial_text(a):
    f=a.finance
    if not f['known']: return f['reason']
    s=f"Baseline annual exposure {money(f['baseline'])}; addressable ceiling {money(f['ceiling'])}; conservative / base / optimistic {money(f['conservative'])} / {money(f['base'])} / {money(f['optimistic'])}."
    if 'net' in f: s+=f" Net annual base scenario {money(f['net'])}."
    if 'payback_months' in f: s+=f" Illustrative payback: {round(f['payback_months'],1) if f['payback_months'] is not None else 'No positive net benefit'} months."
    return s+' Effectiveness is an assumption, not measured model performance.'

def roadmap_markdown(a):
    return '\n\n'.join(f"### {r['weeks']} · {r['phase']}\n{r['objective']}\n\n"+'\n'.join('- '+v for v in r['activities'])+f"\n\nOwner: {r['owner']}\n\nDeliverable: {r['deliverable']}\n\nDependencies: {r['dependencies']}\n\nExit criteria: {r['exit']}" for r in build_roadmap(a))

def proposal(e,a,success='Agree business-relevant precision, recall/error, intervention capacity and minimum incremental value with the business owner.'):
    c=a.use_case
    sections=[('Executive summary',f"Investigate {c['name']}. Current status: {a.status}. No model has been trained or evaluated."),('Enterprise context',f'{e.name}, {e.industry}; declared SAP products: {", ".join(e.sap_products) or "Unknown"}. All inputs are fictional or user-entered.'),('Business problem',c['problem']),('Prediction objective',f"{c['target']} ({c['prediction_type']}). Timing: {c['timing']}. Action: {c['action']}"),('Potential business value',financial_text(a)+f'\n\nAssumptions: {a.finance["assumptions"]}; exposure override: {a.challenge.exposure}.'),('Data requirements',', '.join(c['required_data'])+'\n\nFeature categories: '+', '.join(c['features'])+'\n\nUnconfirmed requirements: '+('; '.join(a.feasibility['missing']) or 'Verify all demo assumptions.')),('Proposed baseline',c['baseline']+f". Incumbent status: {a.challenge.incumbent}. Gap hypothesis: {a.challenge.gap or 'Not established'}."),('Evaluation metrics',c['metrics']+'; use chronological evaluation and entity grouping when appropriate.'),('Relevant technical risks','; '.join(c['data_risks'])),('Deployment/security questions','; '.join(c['deployment_questions'])),('Stakeholder roles',f"Business: {a.challenge.owner or 'Unassigned'}; data owner: {'assumed identified' if a.challenge.evidence.get('data_owner') is True else 'confirm'}; data science, IT/security and executive sponsor required."),('Timeline',roadmap_markdown(a)),('Success criteria to be agreed',success),('Decision after POC','Proceed only if business utility, incremental value, operating constraints and security meet agreed thresholds. Otherwise defer or stop.')]
    return '# POC proposal · '+c['name']+'\n\nUnofficial discussion draft. Fictional / illustrative assumptions; no SAP integration or model inference.\n\n'+'\n\n'.join(f'## {i}. {title}\n\n{body}' for i,(title,body) in enumerate(sections,1))

def executive_brief(e,assessments,weights=None,selected=None):
    text=f'# Enterprise AI Opportunity Assessment\n\nUnofficial prototype · fictional / user-entered data · provisional assumptions · no validated savings.\n\n## 1. Enterprise overview\n\n{e.name} · {e.industry} · revenue {money(e.revenue)} · {e.employees:,} employees · {e.geography}\n\n## 2. Strategic priorities\n\n'+', '.join(e.priorities)+f'\n\nResources: {e.resources}. Existing ML: {e.ml_capabilities}.\n\n## 3. SAP landscape assumptions\n\n'+', '.join(e.sap_products)+f'. {e.data_assumptions} SAP context does not prove integration or data availability.\n\n## 4. Identified predictive use cases\n\n'
    text+='\n'.join(f'- {a.use_case["name"]}: {a.reason}' for a in assessments)
    text+='\n\n## 5. Opportunity comparison\n\n| Opportunity | Provisional score | Impact | Feasibility | Readiness | Incremental | Status |\n|---|---:|---:|---:|---:|---:|---|\n'
    text+='\n'.join(f'| {a.use_case["name"]} | {a.score:.0f} | '+ ' | '.join(f'{a.scores[k]:.0f}' for k in DEFAULT_WEIGHTS)+f' | {a.status} |' for a in assessments)
    text+=f'\n\n## 6. Prioritization rationale\n\nIllustrative normalized weights: {normalize_weights(weights or DEFAULT_WEIGHTS)}. Impact blends annual scenario benefit (55%), severity (30%) and priority alignment (15%). Value uses a capped log scale from $10k to $10m; unknown financial value scores zero for that component. Unknown evidence scores zero, but remains labeled unknown. Readiness combines organization and inverse complexity. Incremental score is 70 for an explicit hypothesis, 40 for no identified incumbent, otherwise zero. Gates determine eligibility before scores.\n\n## 7. Financial scenarios and assumptions\n\n'
    text+='\n\n'.join(f'### {a.use_case["name"]}\n{financial_text(a)}\n\nInputs: {a.finance["assumptions"]}; annual exposure override: {a.challenge.exposure}.' for a in assessments)
    text+='\n\n## 8. Technical/data risks\n\n'+'\n'.join(f'- {a.use_case["name"]}: '+ '; '.join(a.gates+a.use_case['data_risks']) for a in assessments)
    text+='\n\n## 9. Current-solution overlap\n\n'+'\n'.join(f'- {a.use_case["name"]}: {a.challenge.incumbent}; hypothesis: {a.challenge.gap or "Not established"}; baseline: {a.use_case["baseline"]}.' for a in assessments)
    text+='\n\n## 10. Recommended initial evaluation\n\n'+recommendation(assessments)
    text+='\n\n## 11. 90-day roadmap\n\n'+(roadmap_markdown(selected or assessments[0]) if assessments else 'Enable a challenge and complete discovery first.')
    text+='\n\n## 12. Open questions\n\n'+'\n'.join(f'- {a.use_case["name"]}: {a.next_step}' for a in assessments)
    return text
