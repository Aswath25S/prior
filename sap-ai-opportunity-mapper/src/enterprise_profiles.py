import json
from .models import Enterprise, Challenge
from .opportunity_catalog import ROOT

def load_profiles():
    rows = json.loads((ROOT / 'data/fictional_accounts.json').read_text())
    return [Enterprise(**{**row, 'challenges': {k: Challenge(**v) for k, v in row['challenges'].items()}}) for row in rows]

def custom_profile():
    return Enterprise('New enterprise', 'Industrial manufacturing', 0, 0, '', [], [], {})
