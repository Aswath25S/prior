"""Scoping guides for capabilities with bounded client-based evaluations.

Source: https://github.com/PriorLabs/tabpfn-extensions
These guides describe the complete POC, beyond the app's initial evaluation.
"""

CAPABILITIES = {
    "anomaly_detection": {
        "name": "Unusual-record detection and review",
        "terms": ("anomal", "outlier", "unusual", "abnormal"),
        "requires_target": False,
        "purpose": "Rank unusual records for expert review and identify which cases deserve investigation.",
        "deliverable": "A review queue of unusual records, a review protocol, and an agreed intervention process.",
        "design": "Reserve unseen records or the latest period; fit density estimation on representative history. Review top-ranked records with domain experts.",
        "baseline": "Isolation Forest and existing review rules",
        "metrics": "Expert-confirmed precision at the review budget; alert volume and stability. Measure recall only if a representative labeled audit set exists.",
        "risk": "Unusual records are not necessarily failures or fraud; review labels are needed to measure usefulness.",
        "success": "Agree how many alerts the team can review and the minimum fraction that must warrant action.",
    },
    "imputation": {
        "name": "Missing-data recovery",
        "terms": ("imput", "missing", "incomplete", "data quality", "fill gaps"),
        "requires_target": False,
        "purpose": "Recover missing feature values to improve the usability of operational records.",
        "deliverable": "An imputation policy with per-column validation, uncertainty review, and preserved original missing-value flags.",
        "design": "Hold out observed cells with representative masking patterns. Fit on training rows only; compare recovered cells with their known values across repeated masks.",
        "baseline": "Training-set median/mode imputation",
        "metrics": "Per-column MAE for numeric values and accuracy for categorical values; downstream utility and sensitivity to missingness patterns.",
        "risk": "Artificial masking may not represent real missingness. Structurally absent values should not be filled automatically.",
        "success": "Agree recovery error tolerances by column and demonstrate improvement over median/mode without hiding uncertainty.",
    },
    "interpretability": {
        "name": "Predictive factor analysis",
        "terms": ("explain", "driver", "factor", "understand", "feature importance", "root cause"),
        "requires_target": True,
        "purpose": "Identify which recorded factors contribute to model predictions and guide further operational investigation.",
        "deliverable": "A validated predictive model with global and case-level explanations and a shortlist of hypotheses for stakeholder review.",
        "design": "Validate a supervised model on unseen rows first, then assess SHAP or permutation explanations across resamples and operational segments.",
        "baseline": "Linear-model explanations and permutation importance",
        "metrics": "Predictive holdout performance, explanation stability across resamples, and expert assessment of usefulness.",
        "risk": "Predictive associations do not establish causality or prove that changing a feature improves the outcome.",
        "success": "Agree explanation stability and review criteria; validate proposed interventions separately before acting on them.",
    },
    "synthetic_generation": {
        "name": "Synthetic tabular data generation",
        "terms": ("synthetic", "generate data", "augment", "test data", "simulate records"),
        "requires_target": False,
        "purpose": "Generate representative artificial records for a defined testing or development workflow.",
        "deliverable": "A synthetic dataset with a fidelity report, constraint checks, and documented allowed uses.",
        "design": "Reserve real holdout rows before fitting a generator. Evaluate marginal and joint distributions, constraint violations, duplication, and downstream utility on untouched real data.",
        "baseline": "Empirical resampling of training records",
        "metrics": "Distribution fidelity, constraint-violation rate, nearest-neighbor similarity, and downstream task utility where labels exist.",
        "risk": "Generated records can reproduce sensitive patterns; synthetic generation does not guarantee privacy or create independent evidence.",
        "success": "Agree required fidelity and allowed uses, and review disclosure risk before sharing generated records.",
    },
}


def execution_note(capability: str) -> str:
    if capability == "prediction":
        return "Prediction benchmark available in this app."
    return "Initial client-based evaluation available in Step 5; confirm the method and its limitations before treating it as a full POC."
