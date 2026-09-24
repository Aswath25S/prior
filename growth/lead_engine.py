"""Shared lead scoring and sales-capacity allocation for Streamlit pages."""

from pathlib import Path

import pandas as pd


APP_DIR = Path(__file__).resolve().parent
DATA_VERSION = "2026-09-24-calendar-sync-v4"
STATUS_ORDER = ["In Progress", "Success", "Rejection"]
MEETING_STATUS_ORDER = [
    "Invite sent", "Scheduled", "In progress", "Completed", "No-show", "Cancelled"
]
SELF_SERVE_QUALITY_THRESHOLD = 30
SELF_SERVE_WAIT_HOURS = 14 * 24


def load_leads():
    return pd.read_csv(APP_DIR / "sample_leads.csv")


def load_sales_team():
    return pd.read_csv(APP_DIR / "sales_team.csv")


def sync_meeting_statuses(leads, now=None):
    """Derive meeting status from Calendly event times on every app refresh."""
    workflow_defaults = {
        "meeting_status": "Not invited",
        "meeting_start_at": "",
        "meeting_end_at": "",
        "meeting_join_url": "",
    }
    for column, default in workflow_defaults.items():
        if column not in leads.columns:
            leads[column] = default
        else:
            leads[column] = leads[column].fillna(default)

    current_time = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("UTC")
    else:
        current_time = current_time.tz_convert("UTC")

    changes = 0
    for lead_index, lead in leads.iterrows():
        if lead["meeting_status"] in {"Cancelled", "No-show"}:
            continue
        start_at = pd.to_datetime(lead["meeting_start_at"], utc=True, errors="coerce")
        if pd.isna(start_at):
            continue
        end_at = pd.to_datetime(lead["meeting_end_at"], utc=True, errors="coerce")
        if pd.isna(end_at):
            end_at = start_at + pd.Timedelta(minutes=30)
            leads.at[lead_index, "meeting_end_at"] = end_at.isoformat()

        if current_time < start_at:
            derived_status = "Scheduled"
        elif current_time <= end_at:
            derived_status = "In progress"
        else:
            derived_status = "Completed"

        if lead["meeting_status"] != derived_status:
            leads.at[lead_index, "meeting_status"] = derived_status
            changes += 1

    return changes


def record_calendly_invites(leads, lead_view, sales_team):
    """Record one demo Calendly email whenever a lead receives a sales owner."""
    workflow_defaults = {
        "meeting_status": "Not invited",
        "calendly_sent_at": "",
        "calendly_owner_id": "",
        "meeting_start_at": "",
        "meeting_end_at": "",
        "meeting_join_url": "",
    }
    for column, default in workflow_defaults.items():
        if column not in leads.columns:
            leads[column] = default
        else:
            leads[column] = leads[column].fillna(default)

    calendly_by_owner = dict(zip(sales_team["salesperson_id"], sales_team["calendly_url"]))
    salesperson_by_owner = dict(zip(sales_team["salesperson_id"], sales_team["name"]))
    sent_at = pd.Timestamp.now(tz="UTC").isoformat()
    outbox = []

    for _, assigned_lead in lead_view.iterrows():
        owner_id = assigned_lead["owner_id"]
        if assigned_lead["engagement"] != "Sales assigned" or owner_id not in calendly_by_owner:
            continue

        lead_matches = leads.index[leads["lead_id"] == assigned_lead["lead_id"]]
        if len(lead_matches) == 0:
            continue
        lead_index = lead_matches[0]
        if leads.at[lead_index, "calendly_owner_id"] == owner_id:
            continue

        calendly_url = calendly_by_owner[owner_id]
        salesperson = salesperson_by_owner[owner_id]
        leads.at[lead_index, "meeting_status"] = "Invite sent"
        leads.at[lead_index, "calendly_sent_at"] = sent_at
        leads.at[lead_index, "calendly_owner_id"] = owner_id
        outbox.append({
            "lead_id": assigned_lead["lead_id"],
            "to": assigned_lead["email"],
            "subject": f"Book time with {salesperson} at Prior Labs",
            "calendly_url": calendly_url,
            "sent_at": sent_at,
            "delivery": "Demo email recorded",
        })

    sync_meeting_statuses(leads)
    return outbox


def normalize_source(source):
    source = str(source or "").lower()
    if "sap" in source:
        return "SAP"
    if any(term in source for term in ["demo", "website", "contact"]):
        return "Direct website"
    if "api" in source:
        return "Free API"
    if "github" in source or "open" in source:
        return "Open-source / GitHub"
    return "Other"


def classify_account_type(company_size):
    try:
        size = int(company_size)
    except (TypeError, ValueError):
        return "Unknown"
    if size >= 5000:
        return "Large enterprise"
    if size >= 500:
        return "Mid-market / enterprise"
    if size >= 50:
        return "SMB / scale-up"
    if size > 0:
        return "Small company"
    return "Individual / unknown"


def classify_persona(title):
    title = str(title or "").lower()
    if any(term in title for term in ["chief", "vp", "vice president", "head of", "director"]):
        return "Decision-maker / senior leader"
    if any(term in title for term in ["data scientist", "machine learning", "ml engineer", "research scientist", "engineer"]):
        return "Technical champion"
    if any(term in title for term in ["student", "intern", "research assistant"]):
        return "Student / academic"
    return "Other / unknown"


def classify_use_case(text):
    text = str(text or "").lower()
    noncommercial = any(term in text for term in [
        "school project", "class project", "student project", "personal project",
        "hobby", "coursework", "learning only", "toy dataset",
    ])
    production = any(term in text for term in [
        "production", "deploy", "deployment", "rollout", "enterprise",
        "procurement", "security review", "sla", "on-prem", "private cloud",
    ])
    evaluation = any(term in text for term in [
        "benchmark", "evaluate", "evaluation", "pilot", "poc", "proof of concept",
        "compare", "xgboost", "lightgbm", "baseline",
    ])
    business_problem = any(term in text for term in [
        "fraud", "credit", "default", "risk", "forecast", "demand", "maintenance",
        "clinical", "churn", "pricing", "claims", "propensity", "quality", "failure",
        "underwriting",
    ])
    commercial = any(term in text for term in [
        "pricing", "quote", "contract", "buy", "purchase", "sales", "demo",
        "security", "procurement", "license", "commercial",
    ])

    if noncommercial:
        maturity, maturity_score = "Non-commercial", 0
    elif production:
        maturity, maturity_score = "Production-ready", 25
    elif evaluation and business_problem:
        maturity, maturity_score = "Active evaluation", 21
    elif business_problem:
        maturity, maturity_score = "Defined business use case", 15
    elif evaluation:
        maturity, maturity_score = "Technical exploration", 9
    else:
        maturity, maturity_score = "Exploratory / unclear", 5

    return {
        "use_case_maturity": maturity,
        "use_case_score": maturity_score,
        "explicit_commercial_language": commercial,
        "production_signal": production,
        "evaluation_signal": evaluation,
    }


def score_lead(row):
    source = normalize_source(row.get("source", ""))
    account_type = classify_account_type(row.get("company_size", 0))
    persona = classify_persona(row.get("job_title", ""))
    use_case = classify_use_case(row.get("inbound_text", ""))
    size = int(row.get("company_size", 0) or 0)

    if size >= 5000:
        potential = 32
    elif size >= 500:
        potential = 24
    elif size >= 50:
        potential = 14
    elif size > 0:
        potential = 7
    else:
        potential = 1
    if source == "SAP" and size >= 500:
        potential += 8
    elif persona == "Decision-maker / senior leader" and size >= 500:
        potential += 5
    potential = min(potential, 40)

    intent = {"SAP": 15, "Direct website": 14, "Free API": 7, "Open-source / GitHub": 3, "Other": 5}[source]
    if use_case["explicit_commercial_language"]:
        intent += 10
    if use_case["production_signal"]:
        intent += 7
    elif use_case["evaluation_signal"]:
        intent += 5
    activity = int(row.get("account_activity_count", 1) or 1)
    same_company_users = int(row.get("same_company_users", 1) or 1)
    if activity >= 5:
        intent += 2
    if same_company_users >= 3:
        intent += 3
    intent = min(intent, 35)

    maturity = use_case["use_case_score"]
    raw_score = potential + intent + maturity
    completeness = "High"
    if not size or not row.get("job_title") or not row.get("inbound_text"):
        completeness = "Medium"
    if sum([bool(size), bool(row.get("job_title")), bool(row.get("inbound_text"))]) <= 1:
        completeness = "Low"

    return {
        "normalized_source": source,
        "account_type": account_type,
        "persona": persona,
        "use_case_maturity": use_case["use_case_maturity"],
        "commercial_potential": potential,
        "buying_intent": intent,
        "use_case_score": maturity,
        "score": raw_score,
        "confidence": completeness,
        "why": " | ".join([
            f"Commercial potential: {potential}/40 ({account_type})",
            f"Buying intent: {intent}/35 ({source}; {activity} recent company activities; {same_company_users} user(s))",
            f"Use-case maturity: {maturity}/25 ({use_case['use_case_maturity']})",
        ]),
    }


def waiting_points(hours):
    if hours < 2:
        return 0
    if hours < 6:
        return 2
    if hours < 12:
        return 4
    if hours < 24:
        return 7
    if hours < 48:
        return 11
    return 15


def format_wait(hours):
    total_hours = max(int(hours), 0)
    days, remaining_hours = divmod(total_hours, 24)
    return f"{days}d {remaining_hours}h" if days else f"{remaining_hours}h"


def build_lead_view(leads, sales_team):
    scored = leads.apply(lambda row: pd.Series(score_lead(row)), axis=1)
    result = pd.concat([leads.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    result["received_at"] = pd.to_datetime(result["inbound_time"], utc=True, errors="coerce")
    result["waiting_hours"] = (
        (pd.Timestamp.now(tz="UTC") - result["received_at"])
        .dt.total_seconds().div(3600).clip(lower=0).fillna(0)
    )
    result["time_points"] = result["waiting_hours"].apply(waiting_points)
    result["qualification_score"] = result["score"]
    result["qualification_points"] = (result["qualification_score"] * 0.85).round().astype(int)
    result["score"] = (result["qualification_points"] + result["time_points"]).clip(upper=100).astype(int)
    result["self_serve"] = (
        (result["qualification_score"] < SELF_SERVE_QUALITY_THRESHOLD)
        & (result["waiting_hours"] >= SELF_SERVE_WAIT_HOURS)
    )
    result["waiting"] = result["waiting_hours"].apply(format_wait)
    result = result.sort_values(
        ["score", "qualification_score", "received_at"], ascending=[False, False, True]
    ).copy()

    valid_owners = set(sales_team["salesperson_id"])
    result.loc[
        (result["status"] == "In Progress") | ~result["owner_id"].isin(valid_owners), "owner_id"
    ] = "UNASSIGNED"
    capacity_left = dict(zip(sales_team["salesperson_id"], sales_team["weekly_capacity"].astype(int)))
    for index, lead in result.iterrows():
        if lead["status"] != "In Progress" or lead["self_serve"]:
            continue
        available = [owner for owner, capacity in capacity_left.items() if capacity > 0]
        if not available:
            continue
        owner = sorted(available, key=lambda value: (-capacity_left[value], value))[0]
        result.at[index, "owner_id"] = owner
        capacity_left[owner] -= 1

    result["engagement"] = "Queued — unassigned"
    result.loc[result["owner_id"] != "UNASSIGNED", "engagement"] = "Sales assigned"
    result.loc[result["self_serve"], "engagement"] = "Self-serve"
    result.loc[result["status"].isin(["Success", "Rejection"]), "engagement"] = "Completed"
    owner_names = dict(zip(sales_team["salesperson_id"], sales_team["name"]))
    owner_names["UNASSIGNED"] = "Unassigned"
    result["owner"] = result["owner_id"].map(owner_names).fillna("Unassigned")
    calendly_links = dict(zip(sales_team["salesperson_id"], sales_team["calendly_url"]))
    result["calendly_url"] = result["owner_id"].map(calendly_links).fillna("")
    return result.reset_index(drop=True)
