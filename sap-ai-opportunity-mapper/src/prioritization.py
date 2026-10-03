DEFAULT_WEIGHTS={'impact':35,'feasibility':30,'readiness':20,'incremental':15}

def normalize_weights(weights):
    if any(v<0 for v in weights.values()): raise ValueError('Weights cannot be negative')
    total=sum(weights.values())
    return {k:100*v/total for k,v in weights.items()} if total else DEFAULT_WEIGHTS.copy()

def weighted_score(scores,weights):
    normalized=normalize_weights(weights)
    return sum(scores[k]*normalized[k]/100 for k in DEFAULT_WEIGHTS)

def rank(assessments):
    # Gates are categorical: even immense potential value cannot override a blocker.
    return sorted(assessments,key=lambda a:(not a.gates,a.status=='Feasible candidate for discovery/POC planning',a.score),reverse=True)
