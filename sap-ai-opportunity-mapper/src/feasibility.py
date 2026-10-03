EVIDENCE = {
    'history':'Usable historical records exist', 'target':'Historical outcomes / labels are recorded',
    'enough_examples':'Enough examples and positive outcomes for evaluation', 'features':'Required features available before the decision',
    'access':'Legal and operational data access confirmed', 'depth':'Historical depth known',
    'timing':'Prediction timing defined', 'horizon':'Prediction horizon defined',
    'structured':'Suitable structured classification / regression framing', 'action':'Operational business action defined',
    'data_owner':'Data owner identified', 'kpi':'Current business KPI known',
    'sponsor':'Stakeholder sponsorship confirmed', 'poc':'Willingness and resources to run a POC'}

def assess(challenge):
    ev=challenge.evidence
    def score(keys): return sum(100 if ev.get(k) is True else 0 for k in keys)/len(keys)
    data=score(['history','target','enough_examples','features','access','depth'])
    technical=score(['structured','features','timing','horizon'])
    organization=(score(['data_owner','kpi','sponsor','poc','action'])*5+(100 if challenge.owner.strip() else 0))/6
    gates=[]
    for key,label in [('target','Define target and validate recorded historical outcomes'),('history','Validate usable historical outcome data'),('features','Confirm features exist at decision time'),('timing','Define prediction timing'),('horizon','Define prediction horizon'),('structured','Review structured prediction formulation'),('action','Define the business action'),('access','Confirm authorized data access')]:
        if ev.get(key) is not True: gates.append(label + (' (unknown)' if ev.get(key) is None else ' (not met)'))
    if not challenge.owner.strip(): gates.append('Identify business owner')
    if challenge.incumbent in ['Existing capability; incremental value unproven','Already performs satisfactorily','Information missing'] and not challenge.gap.strip():
        gates.append('Justify incremental value against the incumbent / confirm current solution')
    missing=[label for k,label in EVIDENCE.items() if ev.get(k) is not True]
    feasibility=(data+technical)/2
    readiness=(organization+(6-challenge.complexity)*20)/2
    status='Blocked by critical missing requirements' if gates else ('Feasible candidate for discovery/POC planning' if not missing and challenge.owner.strip() else 'Needs validation')
    return dict(data=data,technical=technical,organization=organization,feasibility=feasibility,readiness=readiness,gates=gates,missing=missing,status=status)
