from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Challenge:
    enabled: bool = True
    severity: int = 3
    owner: str = ''
    exposure: Optional[float] = None
    evidence: dict = field(default_factory=dict)
    financial: dict = field(default_factory=dict)
    incumbent: str = 'Information missing'
    gap: str = ''
    complexity: int = 3

@dataclass
class Enterprise:
    name: str
    industry: str
    revenue: float
    employees: int
    geography: str
    sap_products: list
    priorities: list
    challenges: dict
    ml_capabilities: str = 'Unknown'
    data_assumptions: str = 'Customer-specific access and schema require confirmation.'
    resources: str = 'Unknown'

@dataclass
class Assessment:
    use_case: dict
    challenge: Challenge
    reason: str
    finance: dict
    feasibility: dict
    scores: dict
    score: float
    gates: list
    status: str
    next_step: str
