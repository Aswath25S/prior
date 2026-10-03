FUNCTIONS=['Finance','Procurement','Supply chain','Operations','Sales / customer management']

def landscape_rows(assessments):
    return [{'Function':a.use_case['function'],'Pain point':a.use_case['pain'],'Prediction':a.use_case['target'],
             'Business action':a.use_case['action'],'Data categories':', '.join(a.use_case['required_data']),
             'Business owner':a.challenge.owner or 'Unknown','Possible SAP context':a.use_case['sap_context'],
             'Missing information':'; '.join(a.feasibility['missing']) or 'Demo assumptions require confirmation'} for a in assessments]
