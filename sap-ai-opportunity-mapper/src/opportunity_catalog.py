import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_catalog():
    return json.loads((ROOT / 'data/opportunity_catalog.json').read_text())
