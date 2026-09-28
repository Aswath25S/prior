"""Generate structured call preparation from verified lead data."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

CALL_PREP_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "meeting_objective": {"type": "string"},
        "buying_signals": {"type": "array", "items": {"type": "string"}},
        "use_case_hypothesis": {"type": "string"},
        "information_gaps": {"type": "array", "items": {"type": "string"}},
        "discovery_questions": {"type": "array", "items": {"type": "string"}},
        "suggested_agenda": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "desired_next_step": {"type": "string"},
    },
    "required": [
        "summary",
        "meeting_objective",
        "buying_signals",
        "use_case_hypothesis",
        "information_gaps",
        "discovery_questions",
        "suggested_agenda",
        "risks",
        "desired_next_step",
    ],
    "additionalProperties": False,
}


def _text(value, default="Unknown"):
    if value is None:
        return default
    rendered = str(value).strip()
    return rendered if rendered and rendered.lower() != "nan" else default


def _integer(value, default=0):
    try:
        if value is None:
            return default
        normalized = str(value).strip().replace(",", "").lstrip("~")
        return max(int(float(normalized)), 0)
    except (TypeError, ValueError, OverflowError):
        return default


def call_prep_context(lead):
    """Select and normalize only the lead fields needed for call preparation."""
    return {
        "lead_id": _text(lead.get("lead_id")),
        "name": _text(lead.get("name")),
        "job_title": _text(lead.get("job_title")),
        "company": _text(lead.get("company")),
        "company_size": _integer(lead.get("company_size", 0)),
        "company_type": _text(lead.get("account_type")),
        "persona": _text(lead.get("persona")),
        "source": _text(lead.get("normalized_source", lead.get("source"))),
        "inbound_message": _text(lead.get("inbound_text")),
        "recent_company_activities": _integer(lead.get("account_activity_count", 0)),
        "same_company_users": _integer(lead.get("same_company_users", 0)),
        "use_case_stage": _text(lead.get("use_case_maturity")),
        "assigned_salesperson": _text(lead.get("owner")),
        "meeting_status": _text(lead.get("meeting_status")),
        "meeting_start_at": _text(lead.get("meeting_start_at")),
    }


def call_prep_fingerprint(lead):
    encoded = json.dumps(call_prep_context(lead), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contains(message, terms):
    return any(term in message for term in terms)


def build_fallback_call_prep(lead):
    """Create a deterministic brief when the OpenAI API is not configured."""
    context = call_prep_context(lead)
    message = context["inbound_message"].lower()
    signals = []
    if context["same_company_users"] >= 3:
        signals.append(
            f"{context['same_company_users']} people from {context['company']} have engaged, suggesting broader company interest."
        )
    if context["recent_company_activities"] >= 5:
        signals.append(
            f"The company has {context['recent_company_activities']} recent activities, indicating active evaluation."
        )
    if _contains(message, ["pricing", "license", "contract", "buy", "procurement"]):
        signals.append("The inbound message includes commercial or purchasing language.")
    if _contains(message, ["production", "deploy", "vpc", "on-prem", "security", "dpa"]):
        signals.append("The lead is considering production, deployment, or security requirements.")
    if _contains(message, ["benchmark", "baseline", "evaluate", "pilot", "poc", "xgb", "lightgbm"]):
        signals.append("The lead is comparing TabPFN with an existing approach or formal evaluation.")
    if not signals:
        signals.append("The booked meeting is the clearest current signal of active interest.")

    gaps = []
    if not _contains(message, ["accuracy", "auc", "f1", "mae", "rmse", "metric"]):
        gaps.append("The evaluation success metric is not yet defined.")
    if not _contains(message, ["xgb", "lightgbm", "baseline", "current model", "existing model"]):
        gaps.append("The current model or baseline is unknown.")
    if not _contains(message, ["row", "sample", "record", "column", "feature", "dataset size"]):
        gaps.append("Dataset size and structure have not been confirmed.")
    if not _contains(message, ["api", "cloud", "vpc", "on-prem", "self-host", "deployment"]):
        gaps.append("The preferred deployment environment is unknown.")
    if not _contains(message, ["quarter", "month", "week", "deadline", "timeline"]):
        gaps.append("The evaluation and purchasing timeline is unclear.")
    if not _contains(message, ["budget", "procurement", "buyer", "decision maker"]):
        gaps.append("The commercial decision-maker and budget owner are unknown.")

    questions_by_gap = {
        "The evaluation success metric is not yet defined.":
            "Which metric and target would make this evaluation successful?",
        "The current model or baseline is unknown.":
            "What model or workflow is currently used, and how well does it perform?",
        "Dataset size and structure have not been confirmed.":
            "How many rows and features are involved, and what is the prediction target?",
        "The preferred deployment environment is unknown.":
            "Where would TabPFN need to run: API, private cloud, or your own infrastructure?",
        "The evaluation and purchasing timeline is unclear.":
            "What is the target date for completing the evaluation and making a decision?",
        "The commercial decision-maker and budget owner are unknown.":
            "Who else will participate in the technical and commercial decision?",
    }
    questions = [questions_by_gap[gap] for gap in gaps[:5]]
    if len(questions) < 5:
        questions.append("What business decision will this TabPFN model support?")
    if len(questions) < 5:
        questions.append("What would be the most useful next step after this call?")

    risks = []
    if context["use_case_stage"] in {"Exploratory / unclear", "Technical exploration", "Non-commercial"}:
        risks.append("The use case may still be exploratory and lack a committed production path.")
    if context["same_company_users"] <= 1:
        risks.append("Only one known company user is involved, creating single-threaded engagement risk.")
    if not _contains(message, ["budget", "procurement", "pricing", "license"]):
        risks.append("Commercial ownership has not been established.")
    if not risks:
        risks.append("Confirm that technical interest is connected to a funded business priority.")

    summary = (
        f"{context['name']}, {context['job_title']} at {context['company']}, booked a meeting after "
        f"engaging through {context['source']}. Their message points to a {context['use_case_stage'].lower()} "
        f"TabPFN use case, with {context['recent_company_activities']} recent company activities and "
        f"{context['same_company_users']} known company user(s)."
    )
    objective = (
        "Confirm the business problem, define measurable evaluation criteria, and agree on the owner and next step for a TabPFN evaluation."
    )
    hypothesis = (
        f"Hypothesis: {context['company']} is exploring TabPFN for the problem described in the inbound message: "
        f"“{context['inbound_message']}” The exact dataset, baseline, decision process, and production requirements still need confirmation."
    )

    return {
        "summary": summary,
        "meeting_objective": objective,
        "buying_signals": signals,
        "use_case_hypothesis": hypothesis,
        "information_gaps": gaps,
        "discovery_questions": questions[:6],
        "suggested_agenda": [
            "Confirm the business problem and desired outcome",
            "Review the current model, dataset, and evaluation criteria",
            "Discuss TabPFN fit and deployment requirements",
            "Identify stakeholders, timeline, and commercial process",
            "Agree on a concrete next step and owner",
        ],
        "risks": risks,
        "desired_next_step": (
            "Leave the call with an agreed evaluation plan: dataset owner, baseline, success metric, deployment path, and follow-up date."
        ),
    }


def generate_call_prep(lead):
    """Generate an AI brief, falling back safely when configuration is unavailable."""
    fallback = build_fallback_call_prep(lead)
    generated_at = datetime.now(timezone.utc).isoformat()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_CALL_PREP_MODEL", "gpt-6-astra").strip()
    if not api_key:
        return fallback, {
            "mode": "rule-based",
            "model": None,
            "generated_at": generated_at,
            "notice": "Set OPENAI_API_KEY to enable AI-generated call preparation.",
        }

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            store=False,
            instructions=(
                "You prepare concise B2B sales call briefs for Prior Labs and TabPFN. "
                "Use only the supplied lead data. Never invent news, company facts, people, metrics, "
                "competitors, product claims, or previous conversations. Explicitly label uncertain "
                "interpretations as hypotheses and treat missing information as a discovery gap. "
                "Make every recommendation specific, practical, and brief."
            ),
            input=json.dumps({
                "task": "Prepare a sales call brief for this booked meeting.",
                "lead": call_prep_context(lead),
            }),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "call_preparation",
                    "strict": True,
                    "schema": CALL_PREP_SCHEMA,
                }
            },
        )
        brief = json.loads(response.output_text)
        return brief, {
            "mode": "ai",
            "model": model,
            "generated_at": generated_at,
            "notice": "AI-generated from the verified lead fields shown on this page.",
        }
    except Exception as exc:
        return fallback, {
            "mode": "rule-based",
            "model": None,
            "generated_at": generated_at,
            "notice": f"AI generation was unavailable ({type(exc).__name__}); showing a fact-based brief.",
        }
