
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Prior Labs — Inbound Triage Copilot",
    page_icon="⚡",
    layout="wide",
)

# -----------------------------
# Prototype configuration
# -----------------------------
ROUTE_ORDER = ["P1 — Engage now", "P2 — Human review", "P3 — Nurture / monitor", "P4 — Self-serve"]
STATUS_ORDER = [
    "New", "Reviewing", "Needs information", "Contacted", "Meeting booked",
    "Nurture", "Self-serve", "Closed — not relevant",
]
APP_DIR = Path(__file__).resolve().parent
DATA_VERSION = "2026-09-23-wait-time-v1"
RESPONSE_TARGET_HOURS = {
    "P1 — Engage now": 4,
    "P2 — Human review": 12,
    "P3 — Nurture / monitor": 24,
    "P4 — Self-serve": 48,
}

def normalize_source(source: str) -> str:
    s = (source or "").lower()
    if "sap" in s:
        return "SAP"
    if "demo" in s or "website" in s or "contact" in s:
        return "Direct website"
    if "api" in s:
        return "Free API"
    if "github" in s or "open" in s:
        return "Open-source / GitHub"
    return "Other"

def classify_account_type(company_size):
    try:
        n = int(company_size)
    except Exception:
        return "Unknown"
    if n >= 5000:
        return "Large enterprise"
    if n >= 500:
        return "Mid-market / enterprise"
    if n >= 50:
        return "SMB / scale-up"
    if n > 0:
        return "Small company"
    return "Individual / unknown"

def classify_persona(title: str) -> str:
    t = (title or "").lower()
    if any(x in t for x in ["chief", "vp", "vice president", "head of", "director"]):
        return "Decision-maker / senior leader"
    if any(x in t for x in ["data scientist", "machine learning", "ml engineer", "research scientist", "engineer"]):
        return "Technical champion"
    if any(x in t for x in ["student", "intern", "research assistant"]):
        return "Student / academic"
    return "Other / unknown"

def classify_use_case(text: str):
    t = (text or "").lower()

    noncommercial = any(x in t for x in [
        "school project", "class project", "student project", "personal project",
        "hobby", "coursework", "learning only", "toy dataset"
    ])
    production = any(x in t for x in [
        "production", "deploy", "deployment", "rollout", "enterprise",
        "procurement", "security review", "sla", "on-prem", "private cloud"
    ])
    evaluation = any(x in t for x in [
        "benchmark", "evaluate", "evaluation", "pilot", "poc", "proof of concept",
        "compare", "xgboost", "lightgbm", "baseline"
    ])
    business_problem = any(x in t for x in [
        "fraud", "credit", "default", "risk", "forecast", "demand",
        "maintenance", "clinical", "churn", "pricing", "claims", "propensity",
        "quality", "failure", "underwriting"
    ])
    commercial = any(x in t for x in [
        "pricing", "quote", "contract", "buy", "purchase", "sales",
        "demo", "security", "procurement", "license", "commercial"
    ])

    if noncommercial:
        maturity = "Non-commercial"
        maturity_score = 0
    elif production:
        maturity = "Production-ready"
        maturity_score = 25
    elif evaluation and business_problem:
        maturity = "Active evaluation"
        maturity_score = 21
    elif business_problem:
        maturity = "Defined business use case"
        maturity_score = 15
    elif evaluation:
        maturity = "Technical exploration"
        maturity_score = 9
    else:
        maturity = "Exploratory / unclear"
        maturity_score = 5

    return {
        "use_case_maturity": maturity,
        "use_case_score": maturity_score,
        "explicit_commercial_language": commercial,
        "production_signal": production,
        "evaluation_signal": evaluation,
        "business_problem_signal": business_problem,
        "noncommercial_signal": noncommercial,
    }

def score_lead(row):
    source = normalize_source(row.get("source", ""))
    account_type = classify_account_type(row.get("company_size", 0))
    persona = classify_persona(row.get("job_title", ""))
    uc = classify_use_case(row.get("inbound_text", ""))

    # 1) Commercial potential (0-40)
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

    # Strategic enterprise / SAP context can add signal, capped at 40
    if source == "SAP" and size >= 500:
        potential += 8
    elif persona == "Decision-maker / senior leader" and size >= 500:
        potential += 5
    potential = min(potential, 40)

    # 2) Buying intent (0-35)
    source_points = {
        "SAP": 15,
        "Direct website": 14,
        "Free API": 7,
        "Open-source / GitHub": 3,
        "Other": 5,
    }[source]

    intent = source_points
    if uc["explicit_commercial_language"]:
        intent += 10
    if uc["production_signal"]:
        intent += 7
    elif uc["evaluation_signal"]:
        intent += 5

    activity = int(row.get("account_activity_count", 1) or 1)
    same_company_users = int(row.get("same_company_users", 1) or 1)
    if activity >= 5:
        intent += 2
    if same_company_users >= 3:
        intent += 3
    intent = min(intent, 35)

    # 3) Use-case maturity (0-25)
    maturity = uc["use_case_score"]

    raw_score = potential + intent + maturity

    # Hard rules for obvious extremes
    hard_override = None
    override_reason = None

    if source == "SAP" and size >= 5000 and not uc["noncommercial_signal"]:
        hard_override = "P1 — Engage now"
        override_reason = "Strategic SAP-sourced enterprise: protect against a high-cost false negative."
    elif (
        size >= 5000
        and source == "Direct website"
        and (uc["explicit_commercial_language"] or uc["production_signal"] or uc["evaluation_signal"])
    ):
        hard_override = "P1 — Engage now"
        override_reason = "Large enterprise with explicit direct commercial/evaluation intent."
    elif (
        size <= 5
        and source == "Open-source / GitHub"
        and uc["noncommercial_signal"]
        and same_company_users <= 1
    ):
        hard_override = "P4 — Self-serve"
        override_reason = "Obvious non-commercial open-source activity."

    if hard_override:
        route = hard_override
    elif raw_score >= 75:
        route = "P1 — Engage now"
    elif raw_score >= 55:
        route = "P2 — Human review"
    elif raw_score >= 30:
        route = "P3 — Nurture / monitor"
    else:
        route = "P4 — Self-serve"

    confidence = "High"
    if row.get("company_size", 0) in [0, "", None] or not row.get("job_title") or not row.get("inbound_text"):
        confidence = "Medium"
    if sum([
        bool(row.get("company_size", 0)),
        bool(row.get("job_title")),
        bool(row.get("inbound_text"))
    ]) <= 1:
        confidence = "Low"

    why = [
        f"Commercial potential: {potential}/40 ({account_type})",
        f"Buying intent: {intent}/35 ({source}; {activity} recent account activities; {same_company_users} user(s))",
        f"Use-case maturity: {maturity}/25 ({uc['use_case_maturity']})",
    ]
    if override_reason:
        why.append(f"Hard rule: {override_reason}")

    return {
        "normalized_source": source,
        "account_type": account_type,
        "persona": persona,
        "use_case_maturity": uc["use_case_maturity"],
        "commercial_potential": potential,
        "buying_intent": intent,
        "use_case_score": maturity,
        "score": raw_score,
        "route": route,
        "confidence": confidence,
        "why": " | ".join(why),
    }

def enrich_df(df):
    scored = df.apply(lambda r: pd.Series(score_lead(r)), axis=1)
    return pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)


def waiting_points(hours):
    """Add a bounded urgency boost as a lead waits longer in the queue."""
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
    days, hours_left = divmod(total_hours, 24)
    return f"{days}d {hours_left}h" if days else f"{hours_left}h"


def add_time_metrics(scored_df):
    result = scored_df.copy()
    result["received_at"] = pd.to_datetime(result["inbound_time"], utc=True, errors="coerce")
    now = pd.Timestamp.now(tz="UTC")
    result["waiting_hours"] = (
        (now - result["received_at"]).dt.total_seconds().div(3600).clip(lower=0).fillna(0)
    )
    result["time_points"] = result["waiting_hours"].apply(waiting_points)
    result["attention_score"] = (result["score"] + result["time_points"]).clip(upper=100)
    result["response_target_hours"] = result["route"].map(RESPONSE_TARGET_HOURS)
    result["response_status"] = result.apply(
        lambda row: (
            "Overdue"
            if row["waiting_hours"] >= row["response_target_hours"]
            else "Due soon"
            if row["waiting_hours"] >= row["response_target_hours"] * 0.75
            else "On track"
        ),
        axis=1,
    )
    result["waiting"] = result["waiting_hours"].apply(format_wait)
    return result

@st.cache_data
def load_leads():
    return pd.read_csv(APP_DIR / "sample_leads.csv")


@st.cache_data
def load_sales_team():
    return pd.read_csv(APP_DIR / "sales_team.csv")


def suggested_owner(leads, sales_team):
    """Assign the next call to the interchangeable salesperson with most capacity left."""
    assignment_counts = leads["owner_id"].value_counts()
    availability = sales_team[["salesperson_id", "weekly_capacity"]].copy()
    availability["assigned_leads"] = availability["salesperson_id"].map(assignment_counts).fillna(0)
    availability["capacity_left"] = availability["weekly_capacity"] - availability["assigned_leads"]
    availability = availability.sort_values(
        ["capacity_left", "salesperson_id"], ascending=[False, True]
    )
    return availability.iloc[0]["salesperson_id"]


if st.session_state.get("data_version") != DATA_VERSION:
    st.session_state.leads = load_leads()
    st.session_state.pending_lead = None
    st.session_state.data_version = DATA_VERSION
# Keep a running Streamlit session compatible when the underlying demo schema changes.
if "owner_id" not in st.session_state.leads.columns:
    st.session_state.leads["owner_id"] = "UNASSIGNED"
if "status" not in st.session_state.leads.columns:
    st.session_state.leads["status"] = "New"
if "inbound_time" not in st.session_state.leads.columns:
    st.session_state.leads["inbound_time"] = pd.Timestamp.now(tz="UTC").isoformat()
if "pending_lead" not in st.session_state:
    st.session_state.pending_lead = None

team = load_sales_team()
owner_names = dict(zip(team["salesperson_id"], team["name"]))
owner_names["UNASSIGNED"] = "Unassigned"
owner_ids = ["UNASSIGNED"] + team["salesperson_id"].tolist()

df = add_time_metrics(enrich_df(st.session_state.leads))
df["owner"] = df["owner_id"].map(owner_names).fillna("Unassigned")
df = df.sort_values(
    ["attention_score", "score", "received_at"], ascending=[False, False, True]
).reset_index(drop=True)

# A restrained visual layer keeps the hierarchy clear while retaining native Streamlit behavior.
st.markdown(
    """
    <style>
    .block-container {max-width: 1400px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stMetric"] {background: #f7f8fa; border: 1px solid #e7e9ee; padding: 1rem; border-radius: 12px;}
    [data-testid="stMetricLabel"] {color: #626b7a;}
    div[data-testid="stTabs"] button {font-weight: 600;}
    .eyebrow {color: #6255d9; font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;}
    .hero-title {font-size: 2.05rem; font-weight: 750; letter-spacing: -.035em; margin: .2rem 0;}
    .hero-copy {color: #5c6472; font-size: 1rem; margin-bottom: 1.3rem;}
    .priority-card {border: 1px solid #e5e7eb; border-left: 5px solid #6757e8; border-radius: 12px; padding: 1rem 1.15rem; margin: .5rem 0 1rem; background: #fbfbfe;}
    .priority-card h3 {margin: 0 0 .25rem; font-size: 1.15rem;}
    .muted {color: #697386; font-size: .9rem;}
    .team-card {border: 1px solid #e5e7eb; border-radius: 12px; padding: 1rem; min-height: 168px;}
    .initials {display: inline-flex; width: 36px; height: 36px; border-radius: 50%; align-items: center; justify-content: center; background: #ece9ff; color: #5146b8; font-weight: 700; margin-bottom: .5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">Prior Labs · Commercial operations</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Inbound command center</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy">Balance opportunity quality with response time, know who owns each lead, and move every account to a clear next step.</div>',
    unsafe_allow_html=True,
)

new_count = int((df["status"] == "New").sum())
p1_count = int((df["route"] == "P1 — Engage now").sum())
meeting_count = int((df["status"] == "Meeting booked").sum())
oldest_wait = df["waiting_hours"].max() if len(df) else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total inbound", len(df), help="All leads in the unified queue")
k2.metric("Engage now", p1_count, help="P1 leads requiring prompt attention")
k3.metric("New to review", new_count, help="Leads that have not yet been reviewed")
k4.metric("Oldest waiting", format_wait(oldest_wait), help="Time since the oldest lead arrived")

queue_tab, team_tab, add_tab, method_tab = st.tabs(
    ["Priority queue", "Sales team", "Add a lead", "How it works"]
)

with queue_tab:
    st.markdown("### Priority queue")
    st.caption(
        "Highest attention score first: commercial opportunity score plus a bounded waiting-time boost."
    )

    f1, f2, f3, f4 = st.columns([1.1, 1.1, 1.2, 1.4])
    with f1:
        route_filter = st.multiselect("Priority", ROUTE_ORDER, default=ROUTE_ORDER)
    with f2:
        status_filter = st.multiselect("Status", STATUS_ORDER, default=STATUS_ORDER)
    with f3:
        owner_filter = st.multiselect(
            "Owner", ["Unassigned"] + team["name"].tolist(),
            default=["Unassigned"] + team["name"].tolist(),
        )
    with f4:
        search = st.text_input("Search", placeholder="Account, person, or title")

    view = df[
        df["route"].isin(route_filter)
        & df["status"].isin(status_filter)
        & df["owner"].isin(owner_filter)
    ].copy()
    if search:
        query = search.lower()
        view = view[
            view["company"].str.lower().str.contains(query, na=False)
            | view["name"].str.lower().str.contains(query, na=False)
            | view["job_title"].str.lower().str.contains(query, na=False)
        ]

    st.dataframe(
        view[[
            "attention_score", "score", "company", "received_at", "waiting",
            "response_status", "route", "status", "owner"
        ]],
        hide_index=True,
        use_container_width=True,
        height=420,
        column_config={
            "attention_score": st.column_config.ProgressColumn(
                "Attention", min_value=0, max_value=100, width="small",
                help="Opportunity score plus up to 15 waiting-time points",
            ),
            "score": st.column_config.NumberColumn(
                "Opportunity", min_value=0, max_value=100, width="small"
            ),
            "company": st.column_config.TextColumn("Account", width="medium"),
            "received_at": st.column_config.DatetimeColumn(
                "Received", format="MMM D, h:mm a", width="medium"
            ),
            "waiting": st.column_config.TextColumn("Waiting", width="small"),
            "response_status": st.column_config.TextColumn("Response", width="small"),
            "route": st.column_config.TextColumn("Priority", width="medium"),
            "status": st.column_config.TextColumn("Status", width="medium"),
            "owner": st.column_config.TextColumn("Owner", width="medium"),
        },
    )

    if len(view):
        st.markdown("### Lead details & next action")
        lead_options = {
            f"{int(row.attention_score):02d} · {row.company} · {row.waiting} waiting": row.lead_id
            for _, row in view.iterrows()
        }
        selected_label = st.selectbox("Open lead", list(lead_options.keys()))
        lead_id = lead_options[selected_label]
        lead = df[df["lead_id"] == lead_id].iloc[0]

        st.markdown(
            f'<div class="priority-card"><h3>{lead["company"]} · attention {int(lead["attention_score"])}/100</h3>'
            f'<div class="muted">Opportunity {int(lead["score"])}/100 &nbsp;·&nbsp; '
            f'{lead["waiting"]} waiting (+{int(lead["time_points"])} points) &nbsp;·&nbsp; '
            f'{lead["response_status"]}</div></div>',
            unsafe_allow_html=True,
        )

        detail_col, action_col = st.columns([1.65, 1])
        with detail_col:
            st.markdown("**What came in**")
            st.info(lead["inbound_text"])
            s1, s2, s3 = st.columns(3)
            s1.metric("Potential", f"{int(lead['commercial_potential'])}/40")
            s2.metric("Intent", f"{int(lead['buying_intent'])}/35")
            s3.metric("Use case", f"{int(lead['use_case_score'])}/25")
            with st.expander("Why this priority?"):
                for reason in lead["why"].split(" | "):
                    st.markdown(f"- {reason}")
                st.markdown(
                    f"- Waiting time: +{int(lead['time_points'])}/15 "
                    f"({lead['waiting']} against a {int(lead['response_target_hours'])}h response target)"
                )
                st.markdown(f"- Persona: {lead['persona']}")
                st.markdown(f"- Use-case stage: {lead['use_case_maturity']}")
                st.markdown(f"- Data completeness: {lead['confidence']}")

        with action_col:
            st.markdown("**Update ownership and progress**")
            current_owner = lead["owner_id"] if lead["owner_id"] in owner_ids else "UNASSIGNED"
            selected_owner = st.selectbox(
                "Sales owner",
                owner_ids,
                index=owner_ids.index(current_owner),
                format_func=lambda owner_id: owner_names[owner_id],
                key=f"owner_{lead_id}",
            )
            current_status = lead["status"] if lead["status"] in STATUS_ORDER else "New"
            selected_status = st.selectbox(
                "Lead status", STATUS_ORDER, index=STATUS_ORDER.index(current_status), key=f"status_{lead_id}"
            )
            if st.button("Save lead", type="primary", use_container_width=True, key=f"save_{lead_id}"):
                row_index = st.session_state.leads.index[st.session_state.leads["lead_id"] == lead_id][0]
                st.session_state.leads.loc[row_index, "owner_id"] = selected_owner
                st.session_state.leads.loc[row_index, "status"] = selected_status
                st.success("Lead updated for this demo session.")
                st.rerun()
    else:
        st.info("No leads match these filters. Clear a filter to restore the queue.")

with team_tab:
    st.markdown("### Commercial team")
    st.caption(
        "Illustrative team data for the prototype — everyone handles the same inbound work. "
        "Initial calls are assigned only by remaining capacity."
    )
    active_statuses = ["New", "Reviewing", "Needs information", "Contacted", "Meeting booked"]
    cards = st.columns(len(team))
    workload_rows = []
    for card, (_, person) in zip(cards, team.iterrows()):
        person_leads = df[(df["owner_id"] == person["salesperson_id"]) & df["status"].isin(active_statuses)]
        active = len(person_leads)
        capacity = int(person["weekly_capacity"])
        workload_rows.append({
            "Salesperson": person["name"], "Role": person["role"], "Region": person["region"],
            "Active leads": active, "Weekly capacity": capacity,
            "Available slots": max(capacity - active, 0),
        })
        with card:
            st.markdown(
                f'<div class="team-card"><div class="initials">{person["avatar"]}</div>'
                f'<div><strong>{person["name"]}</strong></div><div class="muted">{person["role"]}</div>'
                f'<hr><div><strong>{active}/{capacity}</strong> active / capacity</div>'
                f'<div class="muted">{max(capacity - active, 0)} call slot(s) available</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("#### Workload overview")
    st.dataframe(pd.DataFrame(workload_rows), hide_index=True, use_container_width=True)
    if meeting_count:
        st.success(f"{meeting_count} meeting{'s' if meeting_count != 1 else ''} currently booked across the team.")

with add_tab:
    st.markdown("### Add and prioritize a lead")
    st.caption("Enter the information available at intake. The prototype will score, route, and suggest an owner.")
    with st.form("new_lead"):
        a, b = st.columns(2)
        with a:
            name = st.text_input("Name", "Alex Morgan")
            company = st.text_input("Company", "Example Corp")
            email = st.text_input("Email", "alex@example.com")
            title = st.text_input("Job title", "Senior Data Scientist")
            company_size = st.number_input("Company size", min_value=0, value=1200, step=100)
        with b:
            source = st.selectbox(
                "Source", ["SAP", "Direct website", "Free API", "Open-source / GitHub", "Other"]
            )
            activity = st.number_input("Recent account activities", min_value=1, value=2)
            same_users = st.number_input("Users from same company", min_value=1, value=1)
            inbound_text = st.text_area(
                "What they said or did",
                "We are evaluating TabPFN against LightGBM for a production demand forecasting use case.",
            )
        submitted = st.form_submit_button("Score this lead", type="primary")

    if submitted:
        numeric_ids = [int(str(value).replace("L", "")) for value in st.session_state.leads["lead_id"]]
        next_id = f"L{max(numeric_ids, default=0) + 1:03d}"
        new_row = {
            "lead_id": next_id, "name": name, "email": email, "company": company,
            "company_size": int(company_size), "job_title": title, "source": source,
            "inbound_time": pd.Timestamp.now(tz="UTC").isoformat(),
            "account_activity_count": int(activity), "same_company_users": int(same_users),
            "inbound_text": inbound_text,
        }
        result = score_lead(new_row)
        new_row["owner_id"] = suggested_owner(st.session_state.leads, team)
        new_row["status"] = "New"
        st.session_state.pending_lead = new_row

    if st.session_state.pending_lead is not None:
        pending = st.session_state.pending_lead
        result = score_lead(pending)
        pending_owner = owner_names[pending["owner_id"]]
        st.markdown(
            f'<div class="priority-card"><h3>{result["route"]} · {result["score"]}/100</h3>'
            f'<div class="muted">Suggested owner: {pending_owner} · Status: {pending["status"]}</div></div>',
            unsafe_allow_html=True,
        )
        p1, p2, p3 = st.columns(3)
        p1.metric("Potential", f"{result['commercial_potential']}/40")
        p2.metric("Intent", f"{result['buying_intent']}/35")
        p3.metric("Use case", f"{result['use_case_score']}/25")
        add_col, clear_col = st.columns([1, 1])
        if add_col.button("Add to priority queue", type="primary", use_container_width=True):
            st.session_state.leads = pd.concat(
                [st.session_state.leads, pd.DataFrame([pending])], ignore_index=True
            )
            st.session_state.pending_lead = None
            st.rerun()
        if clear_col.button("Discard", use_container_width=True):
            st.session_state.pending_lead = None
            st.rerun()

with method_tab:
    st.markdown("### How prioritization works")
    st.markdown(
        """
The system asks three simple questions:

1. **Commercial potential** — How valuable could the account be?
2. **Buying intent** — What evidence suggests active commercial evaluation?
3. **Use-case maturity** — How concrete and deployment-ready is the problem?

These create the **opportunity score**. A separate waiting-time signal adds 0–15 points to create the **attention score** used to order the queue. Waiting can move an older lead up the working list, but it does not make that lead more commercially valuable.

Response targets are intentionally simple for the prototype: P1 within 4 hours, P2 within 12 hours, P3 within 24 hours, and P4 within 48 hours. Clear high-value and low-value cases still use explicit rules; people make the final call when evidence is ambiguous.

**The current weights are hypotheses, not proven truth.** Historical meetings, opportunities, and revenue outcomes would be used to recalibrate them.
        """
    )
    st.info("Automate the obvious, escalate the valuable, and use human judgment for ambiguity.")

st.caption(
    "Prototype only · Team members are fictional · Changes are stored only for this browser session · "
    "Production integrations and score calibration are outside this demo."
)
