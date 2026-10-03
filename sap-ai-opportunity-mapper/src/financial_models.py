import math

def estimate(model, inputs, exposure=None):
    """Annual loss/carrying cost scenarios. None remains unknown; zero is valid."""
    required = ['addressable', 'adoption', 'effectiveness']
    if exposure is None:
        required += ['volume', 'rate', 'unit_cost']
        if model == 'payment': required += ['days_late', 'days_reduced', 'cost_of_capital']
    if any(inputs.get(k) is None for k in required):
        return {'known': False, 'reason': 'Not enough information to estimate', 'assumptions': dict(inputs)}
    for k in required:
        if not math.isfinite(inputs[k]) or inputs[k] < 0: raise ValueError(f'Invalid {k}')
    for k in ['rate','addressable','adoption','effectiveness','cost_of_capital']:
        if inputs.get(k) is not None and not 0 <= inputs[k] <= 1: raise ValueError(f'{k} must be between 0 and 1')
    if exposure is not None and (not math.isfinite(exposure) or exposure < 0): raise ValueError('Invalid exposure')
    baseline = exposure if exposure is not None else inputs['volume'] * inputs['rate'] * inputs['unit_cost']
    fraction = inputs['addressable']
    if model == 'payment' and exposure is None:
        baseline *= inputs['days_late'] / 365 * inputs['cost_of_capital']
        fraction *= min(inputs['days_reduced'] / inputs['days_late'], 1) if inputs['days_late'] else 0
    ceiling = baseline * fraction
    base = ceiling * inputs['adoption'] * inputs['effectiveness']
    result = dict(known=True, baseline=baseline, ceiling=ceiling, conservative=base*.5, base=base,
                  optimistic=ceiling*min(inputs['adoption']*1.2,1)*min(inputs['effectiveness']*1.3,1),
                  assumptions=dict(inputs), exposure_override=exposure,
                  sensitivity={'Unit cost / exposure −20%':base*.8,'Base assumptions':base,'Unit cost / exposure +20%':base*1.2})
    operating = inputs.get('operating_cost')
    implementation = inputs.get('implementation_cost')
    if operating is not None:
        if operating < 0: raise ValueError('Invalid operating cost')
        result['net'] = base-operating
        if implementation is not None:
            if implementation < 0: raise ValueError('Invalid implementation cost')
            result['payback_months'] = implementation / result['net'] * 12 if result['net'] > 0 else None
    return result
