"""Shared lead scoring and sales-capacity allocation for Streamlit pages."""

from pathlib import Path
import re

import pandas as pd


APP_DIR = Path(__file__).resolve().parent
DATA_VERSION = "2026-09-28-full-range-scoring-v22"
STATUS_ORDER = ["In Progress", "Success", "Rejected"]
SUCCESS_MILESTONES = ["Commercial win"]
REJECTION_REASONS = [
    "No commercial use case",
    "No budget / wrong timing",
    "Technical fit",
    "Deployment / security requirements",
    "Duplicate / existing account",
    "Unresponsive",
    "Other",
]
OPPORTUNITY_STAGES = ["Discovery", "Evaluation", "POC", "Procurement", "Closed won"]
VALUE_BANDS = ["Unknown", "< $50k", "$50k–$200k", "$200k+"]
MEETING_STATUS_ORDER = [
    "Invite sent", "Scheduled", "In progress", "Completed", "No-show", "Cancelled"
]
SELF_SERVE_QUALITY_THRESHOLD = 30
SELF_SERVE_WAIT_DAYS = 7
GENERIC_EMAIL_DOMAINS = {"gmail.com", "outlook.com", "hotmail.com", "yahoo.com", "icloud.com"}
INDIVIDUAL_ACCOUNT_LABELS = {"independent", "not provided"}


def load_leads():
    """Load fictional demo leads without altering their recorded inbound timestamps."""
    return pd.read_csv(APP_DIR / "sample_leads.csv")


def load_sales_team():
    return pd.read_csv(APP_DIR / "sales_team.csv")


def safe_text(value, default=""):
    if value is None or pd.isna(value):
        return default
    rendered = str(value).strip()
    return rendered if rendered else default


def safe_int(value, default=0):
    try:
        if value is None or pd.isna(value):
            return default
        normalized = str(value).strip().replace(",", "").lstrip("~")
        return max(int(float(normalized)), 0)
    except (TypeError, ValueError, OverflowError):
        return default


def ensure_lead_fields(leads):
    """Supply safe workflow defaults before scoring or rendering incomplete inbound data."""
    original_columns = set(leads.columns)
    defaults = {
        "lead_id": "",
        "name": "Unknown contact",
        "email": "",
        "company": "Not provided",
        "company_size": 0,
        "job_title": "Unknown role",
        "source": "Other",
        "inbound_time": "",
        "account_activity_count": 1,
        "inbound_text": "No inbound message provided.",
        "owner_id": "UNASSIGNED",
        "status": "In Progress",
    }
    for column, default in defaults.items():
        if column not in leads.columns:
            leads[column] = default

    if "missing_data_fields" not in leads.columns:
        tracked_fields = [
            "name", "email", "company", "company_size", "job_title", "source",
            "inbound_time", "inbound_text",
        ]

        def field_is_missing(row, field):
            if field not in original_columns:
                return True
            value = row.get(field)
            if field == "company_size":
                return safe_int(value) == 0
            if field == "inbound_time":
                return pd.isna(pd.to_datetime(value, utc=True, errors="coerce"))
            return not safe_text(value)

        leads["missing_data_fields"] = leads.apply(
            lambda row: ", ".join(
                field for field in tracked_fields if field_is_missing(row, field)
            ),
            axis=1,
        )

    used_ids = {safe_text(value) for value in leads["lead_id"] if safe_text(value)}
    for position, lead_index in enumerate(leads.index, start=1):
        if safe_text(leads.at[lead_index, "lead_id"]):
            continue
        candidate = f"AUTO-{position:03d}"
        while candidate in used_ids:
            position += 1
            candidate = f"AUTO-{position:03d}"
        leads.at[lead_index, "lead_id"] = candidate
        used_ids.add(candidate)

    text_defaults = {
        "name": "Unknown contact",
        "company": "Not provided",
        "job_title": "Unknown role",
        "source": "Other",
        "inbound_text": "No inbound message provided.",
        "owner_id": "UNASSIGNED",
        "status": "In Progress",
    }
    for column, default in text_defaults.items():
        leads[column] = leads[column].apply(lambda value: safe_text(value, default))
    leads["email"] = leads["email"].apply(safe_text)
    leads.loc[leads["status"] == "Rejection", "status"] = "Rejected"
    leads.loc[~leads["status"].isin(STATUS_ORDER), "status"] = "In Progress"
    return leads


def next_lead_id(leads):
    numeric_ids = []
    for value in leads.get("lead_id", []):
        match = re.fullmatch(r"L(\d+)", safe_text(value), flags=re.IGNORECASE)
        if match:
            numeric_ids.append(int(match.group(1)))
    return f"L{max(numeric_ids, default=0) + 1:03d}"


def derive_account_id(row):
    """Create a deterministic prototype account key; production should use CRM identity resolution."""
    company_label = safe_text(row.get("company")).lower()
    if company_label in INDIVIDUAL_ACCOUNT_LABELS:
        return f"individual:{safe_text(row.get('lead_id'), 'unknown')}"
    email = safe_text(row.get("email")).lower()
    domain = email.rsplit("@", 1)[-1] if "@" in email else ""
    if domain and domain not in GENERIC_EMAIL_DOMAINS and "." in domain:
        return f"domain:{domain}"
    company = re.sub(r"[^a-z0-9]+", "-", safe_text(row.get("company"), "unknown").lower()).strip("-")
    return f"company:{company}:{safe_text(row.get('lead_id'), 'unknown')}"


def ensure_account_fields(leads):
    ensure_lead_fields(leads)
    if "account_id" not in leads.columns:
        leads["account_id"] = ""
    missing = leads["account_id"].fillna("").astype(str).str.strip().eq("")
    leads.loc[missing, "account_id"] = leads.loc[missing].apply(derive_account_id, axis=1)
    individual_rows = leads["company"].apply(
        lambda value: safe_text(value).lower() in INDIVIDUAL_ACCOUNT_LABELS
    )
    leads.loc[individual_rows, "account_id"] = leads.loc[individual_rows].apply(
        derive_account_id, axis=1
    )
    account_names = leads.groupby("account_id")["company"].transform(
        lambda values: max((safe_text(value, "Unknown") for value in values), key=len)
    )
    leads["account_name"] = account_names
    return leads


def ensure_outcome_fields(leads):
    """Keep older demo data compatible with the structured outcome workflow."""
    defaults = {
        "qualification_outcome": "Pending",
        "pipeline_stage": "Initial follow-up",
        "success_milestone": "",
        "disposition_reason": "",
        "disposition_notes": "",
        "opportunity_created": False,
        "opportunity_stage": "",
        "estimated_value_band": "",
        "next_action": "",
        "next_action_due": "",
        "outcome_recorded_at": "",
        "qualification_score_at_outcome": None,
        "priority_score_at_outcome": None,
    }
    for column, default in defaults.items():
        if column not in leads.columns:
            leads[column] = default
        elif default is not None:
            leads[column] = leads[column].fillna(default)
    return leads


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

    processed_accounts = set()
    invite_candidates = lead_view.sort_values(
        ["score", "contact_qualification_score"], ascending=[False, False]
    )
    for _, assigned_lead in invite_candidates.iterrows():
        owner_id = assigned_lead["owner_id"]
        if assigned_lead["engagement"] != "Sales assigned" or owner_id not in calendly_by_owner:
            continue
        account_id = assigned_lead["account_id"]
        if account_id in processed_accounts:
            continue
        email = safe_text(assigned_lead.get("email"))
        if "@" not in email or email.lower() == "not provided":
            continue
        processed_accounts.add(account_id)

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
            "to": email,
            "subject": f"Book time with {salesperson} at Prior Labs",
            "calendly_url": calendly_url,
            "sent_at": sent_at,
            "delivery": "Demo email recorded",
        })

    sync_meeting_statuses(leads)
    return outbox


def normalize_source(source):
    source = safe_text(source).lower()
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
    size = safe_int(company_size)
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
    title = safe_text(title).lower()
    if any(term in title for term in ["chief", "vp", "vice president", "head of", "director"]):
        return "Decision-maker / senior leader"
    if any(term in title for term in ["data scientist", "machine learning", "ml engineer", "research scientist", "engineer"]):
        return "Technical champion"
    if any(term in title for term in ["student", "intern", "research assistant"]):
        return "Student / academic"
    return "Other / unknown"


def classify_use_case_maturity(text):
    """Classify technical and operational readiness without inferring purchase intent."""
    text = safe_text(text).lower()
    noncommercial = any(term in text for term in [
        "school project", "class project", "student project", "personal project",
        "hobby", "coursework", "learning only", "toy dataset",
    ])
    production = any(term in text for term in [
        "production", "deploy", "deployment", "rollout", "on-prem",
        "private cloud", "vpc", "private networking",
    ])
    evaluation = any(term in text for term in [
        "benchmark", "evaluate", "evaluation", "pilot", "poc", "proof of concept",
        "compare", "xgboost", "lightgbm", "baseline",
    ])
    business_problem = any(term in text for term in [
        "fraud", "credit", "default", "risk", "forecast", "demand", "maintenance",
        "clinical", "churn", "claims", "propensity", "quality", "failure",
        "underwriting",
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
    }


def detect_commercial_intent(text):
    """Return true only for explicit purchasing or commercial-process language."""
    text = safe_text(text).lower()
    return any(term in text for term in [
        "pricing", "price quote", "quote", "contract", "buy", "purchase",
        "procurement", "license",
    ])


def detect_negative_intent(text):
    """Detect language explicitly distancing the contact from an active purchase."""
    text = safe_text(text).lower()
    return any(term in text for term in [
        "no active project", "not buying", "no commercial use", "learning only",
        "personal side project", "just me exploring", "no specific prod project",
    ])


def score_lead(row):
    source = normalize_source(row.get("source", ""))
    account_type = classify_account_type(row.get("company_size", 0))
    persona = classify_persona(row.get("job_title", ""))
    inbound_text = row.get("inbound_text", "")
    use_case = classify_use_case_maturity(inbound_text)
    commercial_intent = detect_commercial_intent(inbound_text)
    negative_intent = detect_negative_intent(inbound_text)
    size = safe_int(row.get("company_size", 0))

    if size >= 5000:
        potential = 35
    elif size >= 500:
        potential = 25
    elif size >= 50:
        potential = 15
    elif size > 0:
        potential = 7
    else:
        potential = 1
    if persona == "Decision-maker / senior leader" and size >= 500:
        potential += 5
    potential = min(potential, 40)

    intent = {"SAP": 10, "Direct website": 8, "Free API": 5, "Open-source / GitHub": 2, "Other": 3}[source]
    if commercial_intent:
        intent += 10
    activity = safe_int(row.get("account_activity_signal", row.get("account_activity_count", 1)), 1)
    known_contacts = safe_int(row.get("account_contact_count", 1), 1)
    if activity >= 5:
        intent += 7
    if known_contacts >= 3:
        intent += 8
    if negative_intent:
        intent -= 10
    if use_case["use_case_maturity"] == "Non-commercial":
        intent = min(intent, 5)
    intent = max(0, min(intent, 35))

    maturity = use_case["use_case_score"]
    raw_score = potential + intent + maturity
    missing_fields = {
        field.strip() for field in safe_text(row.get("missing_data_fields")).split(",") if field.strip()
    }
    score_input_gaps = len(missing_fields.intersection({"company_size", "job_title", "inbound_text"}))
    completeness = "Low" if score_input_gaps >= 2 else "Medium" if score_input_gaps == 1 else "High"

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
            f"Buying intent: {intent}/35 ({source}; {activity} recent account activities; "
            f"{known_contacts} known contact(s))",
            f"Use-case maturity: {maturity}/25 ({use_case['use_case_maturity']})",
        ]),
    }


def format_wait(hours):
    total_minutes = max(int(float(hours) * 60), 0)
    days, remaining_minutes = divmod(total_minutes, 24 * 60)
    display_hours, minutes = divmod(remaining_minutes, 60)
    if days:
        return f"{days}d {display_hours}h"
    if display_hours:
        return f"{display_hours}h {minutes}m"
    return f"{minutes}m"


def waiting_points(days):
    """Add bounded urgency at day boundaries without allowing age to dominate quality."""
    if days < 1:
        return 0
    if days < 2:
        return 2
    if days < 3:
        return 4
    if days < 5:
        return 7
    if days < 7:
        return 11
    return 15


def route_for_score(score):
    if score >= 65:
        return "Engage now"
    if score >= 45:
        return "Human review"
    if score >= SELF_SERVE_QUALITY_THRESHOLD:
        return "Nurture / monitor"
    return "Self-serve"


def build_lead_view(leads, sales_team, now=None):
    ensure_lead_fields(leads)
    ensure_outcome_fields(leads)
    ensure_account_fields(leads)
    prepared = leads.reset_index(drop=True).copy()
    prepared["name"] = prepared["name"].apply(lambda value: safe_text(value, "Unknown contact"))
    prepared["email"] = prepared["email"].apply(lambda value: safe_text(value, "Not provided"))
    prepared["company"] = prepared["company"].apply(lambda value: safe_text(value, "Not provided"))
    prepared["job_title"] = prepared["job_title"].apply(lambda value: safe_text(value, "Unknown role"))
    prepared["source"] = prepared["source"].apply(lambda value: safe_text(value, "Other"))
    prepared["inbound_text"] = prepared["inbound_text"].apply(
        lambda value: safe_text(value, "No inbound message provided.")
    )
    for numeric_column in ["company_size", "account_activity_count"]:
        prepared[numeric_column] = prepared[numeric_column].apply(safe_int)
    prepared["account_contact_count"] = prepared.groupby("account_id")["lead_id"].transform("nunique")
    prepared["account_activity_signal"] = prepared.groupby("account_id")[
        "account_activity_count"
    ].transform(lambda values: max(safe_int(value, 1) for value in values))
    scored = prepared.apply(lambda row: pd.Series(score_lead(row)), axis=1)
    result = pd.concat([prepared, scored.reset_index(drop=True)], axis=1)
    result["received_at"] = pd.to_datetime(result["inbound_time"], utc=True, errors="coerce")
    current_time = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("UTC")
    else:
        current_time = current_time.tz_convert("UTC")
    outcome_times = pd.to_datetime(result["outcome_recorded_at"], utc=True, errors="coerce")
    waiting_end = pd.Series(current_time, index=result.index, dtype="datetime64[ns, UTC]")
    closed_with_time = result["status"].isin(["Success", "Rejected"]) & outcome_times.notna()
    waiting_end.loc[closed_with_time] = outcome_times.loc[closed_with_time]
    result["contact_waiting_hours"] = (
        (waiting_end - result["received_at"])
        .dt.total_seconds().div(3600).clip(lower=0).fillna(0)
    )
    result["contact_waiting"] = result["contact_waiting_hours"].apply(format_wait)
    result["waiting_hours"] = result.groupby("account_id")["contact_waiting_hours"].transform("max")
    result["waiting_days"] = result["waiting_hours"] / 24
    result["contact_qualification_score"] = result["score"].astype(int)
    result["qualification_score"] = result.groupby("account_id")[
        "contact_qualification_score"
    ].transform("max").astype(int)
    result["qualification_points"] = (result["qualification_score"] * 0.85).round().astype(int)
    result["time_points"] = result["waiting_days"].apply(waiting_points).astype(int)
    result["score"] = (
        result["qualification_points"] + result["time_points"]
    ).clip(upper=100).astype(int)
    result["priority_route"] = result["score"].apply(route_for_score)
    result["self_serve"] = (
        (result["qualification_score"] < SELF_SERVE_QUALITY_THRESHOLD)
        & (result["waiting_days"] >= SELF_SERVE_WAIT_DAYS)
    )
    result.loc[result["self_serve"], "priority_route"] = "Self-serve"
    result.loc[
        (result["priority_route"] == "Self-serve") & ~result["self_serve"],
        "priority_route",
    ] = "Nurture / monitor"
    result["waiting"] = result["waiting_hours"].apply(format_wait)
    result = result.sort_values(
        ["score", "qualification_score", "received_at"], ascending=[False, False, True]
    ).copy()

    valid_owners = set(sales_team["salesperson_id"])
    result.loc[~result["owner_id"].isin(valid_owners), "owner_id"] = "UNASSIGNED"
    existing_owner_by_account = {}
    for _, account_rows in result.groupby("account_id", sort=False):
        valid_account_owners = account_rows.loc[
            account_rows["owner_id"].isin(valid_owners), "owner_id"
        ]
        if valid_account_owners.empty:
            continue
        account_owner = valid_account_owners.iloc[0]
        existing_owner_by_account[account_rows.iloc[0]["account_id"]] = account_owner
        active_indices = account_rows.index[account_rows["status"] == "In Progress"]
        result.loc[active_indices, "owner_id"] = account_owner
    capacity_left = dict(zip(sales_team["salesperson_id"], sales_team["weekly_capacity"].astype(int)))

    # Score alone determines which eligible accounts consume the team's limited
    # capacity. Earliest inbound time is the deterministic tie-breaker.
    eligible_rows = result[
        (result["status"] == "In Progress") & ~result["self_serve"]
    ]
    eligible_accounts = (
        eligible_rows.groupby("account_id", as_index=False)
        .agg(score=("score", "max"), received_at=("received_at", "min"))
        .sort_values(
            ["score", "received_at", "account_id"],
            ascending=[False, True, True],
            na_position="last",
        )
    )
    total_capacity = sum(capacity_left.values())
    selected_accounts = eligible_accounts.head(total_capacity)["account_id"].tolist()

    active_mask = result["status"] == "In Progress"
    result.loc[active_mask, "owner_id"] = "UNASSIGNED"
    pending_accounts = []
    for account_id in selected_accounts:
        preferred_owner = existing_owner_by_account.get(account_id)
        if preferred_owner in capacity_left and capacity_left[preferred_owner] > 0:
            account_indices = result.index[active_mask & (result["account_id"] == account_id)]
            result.loc[account_indices, "owner_id"] = preferred_owner
            capacity_left[preferred_owner] -= 1
        else:
            pending_accounts.append(account_id)

    for account_id in pending_accounts:
        available = [owner for owner, capacity in capacity_left.items() if capacity > 0]
        if not available:
            continue
        owner = sorted(available, key=lambda value: (-capacity_left[value], value))[0]
        account_indices = result.index[active_mask & (result["account_id"] == account_id)]
        result.loc[account_indices, "owner_id"] = owner
        capacity_left[owner] -= 1

    result["engagement"] = "Queued — unassigned"
    result.loc[result["owner_id"] != "UNASSIGNED", "engagement"] = "Sales assigned"
    result.loc[
        result["self_serve"] & (result["owner_id"] == "UNASSIGNED"), "engagement"
    ] = "Self-serve"
    result.loc[result["status"].isin(["Success", "Rejected"]), "engagement"] = "Completed"
    owner_names = dict(zip(sales_team["salesperson_id"], sales_team["name"]))
    owner_names["UNASSIGNED"] = "Unassigned"
    result["owner"] = result["owner_id"].map(owner_names).fillna("Unassigned")
    calendly_links = dict(zip(sales_team["salesperson_id"], sales_team["calendly_url"]))
    result["calendly_url"] = result["owner_id"].map(calendly_links).fillna("")
    return result.reset_index(drop=True)


def build_account_view(lead_view):
    """Collapse contact-level rows into the current operating state of each account."""
    rows = []
    for _, account_rows in lead_view.groupby("account_id", sort=False):
        active_rows = account_rows[account_rows["status"] == "In Progress"]
        representative_pool = active_rows if not active_rows.empty else account_rows
        representative = representative_pool.sort_values(
            ["contact_qualification_score", "received_at"], ascending=[False, True]
        ).iloc[0].copy()
        representative["account_contact_count"] = int(account_rows["lead_id"].nunique())
        representative["received_at"] = account_rows["received_at"].min()
        representative["waiting_hours"] = account_rows["waiting_hours"].max()
        representative["waiting"] = format_wait(representative["waiting_hours"])
        representative["account_sources"] = ", ".join(
            sorted(account_rows["normalized_source"].dropna().unique())
        )
        if not active_rows.empty:
            representative["status"] = "In Progress"
        rows.append(representative)
    return pd.DataFrame(rows).sort_values(
        ["score", "received_at"], ascending=[False, True]
    ).reset_index(drop=True)


def persist_owner_assignments(leads, lead_view):
    """Persist the owner allocation produced by the current ranked-capacity pass."""
    owners_by_lead = lead_view.set_index("lead_id")["owner_id"]
    changes = 0
    for lead_index, lead in leads.iterrows():
        lead_id = lead["lead_id"]
        if lead_id not in owners_by_lead.index:
            continue
        owner_id = owners_by_lead.at[lead_id]
        if leads.at[lead_index, "owner_id"] != owner_id:
            leads.at[lead_index, "owner_id"] = owner_id
            changes += 1
    return changes


def update_lead_progress(
    leads,
    lead_snapshot,
    status,
    *,
    pipeline_stage="Initial follow-up",
    success_milestone="",
    disposition_reason="",
    disposition_notes="",
    opportunity_created=False,
    opportunity_stage="",
    estimated_value_band="",
    next_action="",
    next_action_due="",
    now=None,
):
    """Record an actionable status plus the outcome data needed for learning."""
    if status not in STATUS_ORDER:
        raise ValueError(f"Unsupported lead status: {status}")
    if status == "Success":
        success_milestone = "Commercial win"
        opportunity_created = True
        opportunity_stage = "Closed won"
    pipeline_stage = safe_text(pipeline_stage, "Initial follow-up")
    if status == "Success" and success_milestone not in SUCCESS_MILESTONES:
        raise ValueError("A success milestone is required for a successful outcome.")
    if status == "Rejected" and disposition_reason not in REJECTION_REASONS:
        raise ValueError("A disposition reason is required for a rejected outcome.")
    if opportunity_created and opportunity_stage not in OPPORTUNITY_STAGES:
        raise ValueError("An opportunity stage is required when an opportunity is created.")

    ensure_lead_fields(leads)
    ensure_account_fields(leads)
    ensure_outcome_fields(leads)
    lead_id = lead_snapshot["lead_id"]
    matches = leads.index[leads["lead_id"] == lead_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one lead with ID {lead_id}; found {len(matches)}.")
    lead_index = matches[0]
    account_id = safe_text(lead_snapshot.get("account_id"), leads.at[lead_index, "account_id"])
    account_indices = leads.index[leads["account_id"] == account_id]
    previous_status = leads.at[lead_index, "status"]
    previous_outcome_at = leads.at[lead_index, "outcome_recorded_at"]
    leads.loc[account_indices, "owner_id"] = lead_snapshot["owner_id"]
    leads.loc[account_indices, "status"] = status
    leads.loc[account_indices, "next_action"] = str(next_action or "").strip()
    leads.loc[account_indices, "next_action_due"] = str(next_action_due or "")

    def propagate_account_state():
        workflow_fields = [
            "qualification_outcome", "pipeline_stage", "success_milestone",
            "disposition_reason", "disposition_notes", "opportunity_created",
            "opportunity_stage", "estimated_value_band", "outcome_recorded_at",
            "qualification_score_at_outcome", "priority_score_at_outcome",
        ]
        for field in workflow_fields:
            leads.loc[account_indices, field] = leads.at[lead_index, field]

    if status == "In Progress":
        normalized_stage = pipeline_stage.lower()
        if any(term in normalized_stage for term in ["poc", "proof of concept", "pilot", "evaluation"]):
            in_progress_opportunity, in_progress_stage = True, "POC"
        elif any(term in normalized_stage for term in ["qualified", "discovery", "opportunity"]):
            in_progress_opportunity, in_progress_stage = True, "Discovery"
        else:
            in_progress_opportunity, in_progress_stage = False, ""
        reset_values = {
            "qualification_outcome": "Pending",
            "pipeline_stage": pipeline_stage,
            "success_milestone": "",
            "disposition_reason": "",
            "disposition_notes": "",
            "opportunity_created": in_progress_opportunity,
            "opportunity_stage": in_progress_stage,
            "estimated_value_band": estimated_value_band if in_progress_opportunity else "",
            "outcome_recorded_at": "",
            "qualification_score_at_outcome": None,
            "priority_score_at_outcome": None,
        }
        for column, value in reset_values.items():
            leads.at[lead_index, column] = value
        propagate_account_state()
        return

    first_close = previous_status not in {"Success", "Rejected"} or not previous_outcome_at
    leads.at[lead_index, "qualification_outcome"] = (
        "Qualified" if status == "Success" else "Not qualified"
    )
    leads.at[lead_index, "pipeline_stage"] = (
        "Customer" if status == "Success" else safe_text(pipeline_stage, "Closed")
    )
    leads.at[lead_index, "success_milestone"] = (
        success_milestone if status == "Success" else ""
    )
    leads.at[lead_index, "disposition_reason"] = (
        disposition_reason if status == "Rejected" else ""
    )
    leads.at[lead_index, "disposition_notes"] = str(disposition_notes or "").strip()
    leads.at[lead_index, "opportunity_created"] = bool(opportunity_created)
    leads.at[lead_index, "opportunity_stage"] = (
        opportunity_stage if opportunity_created else ""
    )
    leads.at[lead_index, "estimated_value_band"] = (
        estimated_value_band if opportunity_created else ""
    )
    if first_close:
        recorded_at = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
        if recorded_at.tzinfo is None:
            recorded_at = recorded_at.tz_localize("UTC")
        else:
            recorded_at = recorded_at.tz_convert("UTC")
        leads.at[lead_index, "outcome_recorded_at"] = recorded_at.isoformat()
        leads.at[lead_index, "qualification_score_at_outcome"] = int(
            lead_snapshot["qualification_score"]
        )
        leads.at[lead_index, "priority_score_at_outcome"] = int(lead_snapshot["score"])
    propagate_account_state()
