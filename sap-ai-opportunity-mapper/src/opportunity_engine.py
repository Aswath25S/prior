import math
from .models import Assessment
from .financial_models import estimate
from .feasibility import assess
from .prioritization import DEFAULT_WEIGHTS, weighted_score, rank

def generate(enterprise,catalog,weights=None):
    output=[]
    for c in catalog:
        ch=enterprise.challenges.get(c['id'])
        if ch is None or not ch.enabled: continue
        strategic=any(c['priority'].lower() in p.lower() or p.lower() in c['priority'].lower() for p in enterprise.priorities if p.strip())
        f=estimate(c['financial_model'],ch.financial,ch.exposure)
        readiness=assess(ch)
        # Log scaling: $10k = 0; $10m = 100. Revenue is never model revenue or a score input.
        value=min(100,max(0,(math.log10(max(f['base'],1))-4)/3*100)) if f['known'] else 0
        impact=.55*value+.30*ch.severity*20+.15*(100 if strategic else 0)
        incremental=70 if ch.gap.strip() else (40 if ch.incumbent=='No incumbent identified' else 0)
        scores=dict(impact=impact,feasibility=readiness['feasibility'],readiness=readiness['readiness'],incremental=incremental)
        reason=f"Matches enabled challenge: {c['pain']}. " + (f"Aligns with priority: {c['priority']}." if strategic else 'No direct match to a declared strategic priority.')
        reason+=' SAP products are context hypotheses; data access is unverified.'
        next_step=readiness['gates'][0] if readiness['gates'] else (readiness['missing'][0] if readiness['missing'] else f"Confirm {c['baseline']} and agree evaluation thresholds with {ch.owner}.")
        output.append(Assessment(c,ch,reason,f,readiness,scores,weighted_score(scores,weights or DEFAULT_WEIGHTS),readiness['gates'],readiness['status'],next_step))
    return rank(output)

def recommendation(assessments):
    ready=[a for a in assessments if not a.gates and a.status=='Feasible candidate for discovery/POC planning']
    if not ready:
        return 'No initial technical evaluation candidate is ready. Resolve the listed evidence gaps before committing to a POC.'
    a=ready[0]
    benefit=f"${a.finance['base']:,.0f} in illustrative annual benefit" if a.finance['known'] else 'financial benefit remains unknown'
    return f"Begin discovery and POC planning for {a.use_case['name']}. {a.reason} Base scenario: {benefit}. Confirm the assumed evidence and benchmark the incremental-value hypothesis: {a.challenge.gap or 'No incumbent identified; verify existing functionality first.'}"
