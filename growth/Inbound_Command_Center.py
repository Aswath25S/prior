
from pathlib import Path

import pandas as pd
import streamlit as st

from lead_engine import build_lead_view

st.set_page_config(
    page_title="Prior Labs — Inbound Triage Copilot",
    page_icon="⚡",
    layout="wide",
)

# -----------------------------
# Prototype configuration
# -----------------------------
STATUS_ORDER = ["In Progress", "Success", "Rejection"]
ENGAGEMENT_OPTIONS = ["Sales assigned", "Queued — unassigned", "Self-serve", "Completed"]
APP_DIR = Path(__file__).resolve().parent
DATA_VERSION = "2026-09-24-team-v2"
SELF_SERVE_QUALITY_THRESHOLD = 30
SELF_SERVE_WAIT_HOURS = 14 * 24

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
    return {
        "normalized_source": source,
        "account_type": account_type,
        "persona": persona,
        "use_case_maturity": uc["use_case_maturity"],
        "commercial_potential": potential,
        "buying_intent": intent,
        "use_case_score": maturity,
        "score": raw_score,
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
    result["qualification_score"] = result["score"]
    result["qualification_points"] = (result["qualification_score"] * 0.85).round().astype(int)
    result["score"] = (
        result["qualification_points"] + result["time_points"]
    ).clip(upper=100).astype(int)
    result["self_serve"] = (
        (result["qualification_score"] < SELF_SERVE_QUALITY_THRESHOLD)
        & (result["waiting_hours"] >= SELF_SERVE_WAIT_HOURS)
    )
    result["waiting"] = result["waiting_hours"].apply(format_wait)
    return result

@st.cache_data
def load_leads():
    return pd.read_csv(APP_DIR / "sample_leads.csv")


@st.cache_data
def load_sales_team():
    return pd.read_csv(APP_DIR / "sales_team.csv")


def assign_salespeople(scored_leads, sales_team):
    """Give limited sales capacity to eligible leads in descending score order."""
    result = scored_leads.sort_values(
        ["score", "qualification_score", "received_at"], ascending=[False, False, True]
    ).copy()
    valid_owner_ids = set(sales_team["salesperson_id"])
    result.loc[
        (result["status"] == "In Progress") | ~result["owner_id"].isin(valid_owner_ids),
        "owner_id",
    ] = "UNASSIGNED"
    capacity_left = {
        row["salesperson_id"]: int(row["weekly_capacity"])
        for _, row in sales_team.iterrows()
    }

    for index, lead in result.iterrows():
        if lead["status"] != "In Progress":
            continue
        if lead["self_serve"]:
            continue
        available = [person_id for person_id, capacity in capacity_left.items() if capacity > 0]
        if not available:
            continue
        owner_id = sorted(available, key=lambda person_id: (-capacity_left[person_id], person_id))[0]
        result.at[index, "owner_id"] = owner_id
        capacity_left[owner_id] -= 1

    result["engagement"] = "Queued — unassigned"
    result.loc[result["owner_id"] != "UNASSIGNED", "engagement"] = "Sales assigned"
    result.loc[result["self_serve"], "engagement"] = "Self-serve"
    result.loc[result["status"].isin(["Success", "Rejection"]), "engagement"] = "Completed"
    return result


if st.session_state.get("data_version") != DATA_VERSION:
    st.session_state.leads = load_leads()
    st.session_state.pending_lead = None
    st.session_state.data_version = DATA_VERSION
# Keep a running Streamlit session compatible when the underlying demo schema changes.
if "owner_id" not in st.session_state.leads.columns:
    st.session_state.leads["owner_id"] = "UNASSIGNED"
if "status" not in st.session_state.leads.columns:
    st.session_state.leads["status"] = "In Progress"
if "inbound_time" not in st.session_state.leads.columns:
    st.session_state.leads["inbound_time"] = pd.Timestamp.now(tz="UTC").isoformat()
if "pending_lead" not in st.session_state:
    st.session_state.pending_lead = None

team = load_sales_team()
owner_names = dict(zip(team["salesperson_id"], team["name"]))
owner_names["UNASSIGNED"] = "Unassigned"

df = build_lead_view(st.session_state.leads, team)
df = df.sort_values(
    ["score", "qualification_score", "received_at"], ascending=[False, False, True]
).reset_index(drop=True)

# A restrained visual layer keeps the hierarchy clear while retaining native Streamlit behavior.
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1400px;
        padding-top: 5rem;
        padding-bottom: 3rem;
    }
    @media (max-width: 768px) {
        .block-container {padding-top: 4.5rem;}
    }
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, .28);
        padding: 1rem;
        border-radius: 12px;
    }
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {color: var(--text-color) !important;}
    [data-testid="stMetricLabel"] {opacity: .75;}
    div[data-testid="stTabs"] button {font-weight: 600;}
    .eyebrow {
        display: inline-block;
        color: var(--text-color) !important;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, .35);
        border-radius: 999px;
        padding: .3rem .65rem;
        margin-bottom: .35rem;
        opacity: 1;
        font-size: .78rem;
        line-height: 1.5;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    .hero-title {color: var(--text-color); font-size: 2.05rem; font-weight: 750; letter-spacing: -.035em; margin: .2rem 0;}
    .hero-copy {color: var(--text-color); opacity: .68; font-size: 1rem; margin-bottom: 1.3rem;}
    .priority-card {
        color: var(--text-color);
        border: 1px solid rgba(128, 128, 128, .28);
        border-left: 5px solid var(--primary-color);
        border-radius: 12px;
        padding: 1rem 1.15rem;
        margin: .5rem 0 1rem;
        background: var(--secondary-background-color);
    }
    .priority-card h3 {margin: 0 0 .25rem; font-size: 1.15rem;}
    .muted {color: var(--text-color); opacity: .68; font-size: .9rem;}
    .team-card {
        color: var(--text-color);
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, .28);
        border-radius: 12px;
        padding: 1rem;
        min-height: 168px;
    }
    .team-card hr {border-color: rgba(128, 128, 128, .28);}
    .initials {
        display: inline-flex;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        align-items: center;
        justify-content: center;
        background: var(--primary-color);
        color: var(--background-color);
        font-weight: 700;
        margin-bottom: .5rem;
    }
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
st.page_link("pages/1_Sales_Workspace.py", label="Open salesperson workspace", icon="👤")

sales_assigned_count = int((df["engagement"] == "Sales assigned").sum())
self_serve_count = int((df["engagement"] == "Self-serve").sum())
unassigned_count = int((df["engagement"] == "Queued — unassigned").sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total inbound", len(df), help="All leads in the unified queue")
k2.metric("Sales assigned", sales_assigned_count, help="Highest-scoring leads within team capacity")
k3.metric("Queued — unassigned", unassigned_count, help="Eligible leads waiting for sales capacity")
k4.metric("Self-serve", self_serve_count, help="Low-quality leads waiting at least 14 days")

queue_tab, team_tab, add_tab, method_tab = st.tabs(
    ["Priority queue", "Sales team", "Add a lead", "How it works"]
)

with queue_tab:
    st.markdown("### Priority queue")
    st.caption("Search, filter, and sort the full inbound queue. Highest score is the default order.")

    f1, f2, f3, f4, f5 = st.columns([1.6, 1.25, 1.15, 1.05, 1.25])
    with f1:
        search = st.text_input(
            "Search", placeholder="Name, company, or title", key=f"search_{DATA_VERSION}"
        )
    with f2:
        sort_by = st.selectbox(
            "Sort by",
            [
                "Priority score — highest first",
                "Priority score — lowest first",
                "Waiting time — longest first",
                "Inbound time — newest first",
                "Name — A to Z",
                "Company — A to Z",
            ],
            key=f"sort_by_{DATA_VERSION}",
        )
    with f3:
        engagement_filter = st.multiselect(
            "Engagement", ENGAGEMENT_OPTIONS, default=[],
            key=f"engagement_filter_{DATA_VERSION}",
            help="Leave empty to show every engagement type.",
        )
    with f4:
        status_filter = st.multiselect(
            "Status", STATUS_ORDER, default=[], key=f"status_filter_{DATA_VERSION}",
            help="Leave empty to show every status.",
        )
    with f5:
        owner_filter = st.multiselect(
            "Owner", ["Unassigned"] + team["name"].tolist(),
            default=[],
            key=f"owner_filter_{DATA_VERSION}",
            help="Leave empty to show every owner.",
        )

    view = df.copy()
    if engagement_filter:
        view = view[view["engagement"].isin(engagement_filter)]
    if status_filter:
        view = view[view["status"].isin(status_filter)]
    if owner_filter:
        view = view[view["owner"].isin(owner_filter)]
    if search:
        query = search.lower()
        view = view[
            view["company"].str.lower().str.contains(query, na=False)
            | view["name"].str.lower().str.contains(query, na=False)
            | view["job_title"].str.lower().str.contains(query, na=False)
        ]

    if sort_by == "Priority score — highest first":
        view = view.sort_values(
            ["score", "qualification_score", "received_at"], ascending=[False, False, True]
        )
    elif sort_by == "Priority score — lowest first":
        view = view.sort_values(
            ["score", "qualification_score", "received_at"], ascending=[True, True, True]
        )
    elif sort_by == "Waiting time — longest first":
        view = view.sort_values(["waiting_hours", "score"], ascending=[False, False])
    elif sort_by == "Inbound time — newest first":
        view = view.sort_values(["received_at", "score"], ascending=[False, False])
    elif sort_by == "Name — A to Z":
        view = view.sort_values("name", key=lambda values: values.str.lower())
    else:
        view = view.sort_values("company", key=lambda values: values.str.lower())

    st.dataframe(
        view[[
            "score", "name", "company", "received_at", "waiting",
            "engagement", "status", "owner"
        ]],
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "score": st.column_config.ProgressColumn(
                "Priority score", min_value=0, max_value=100, width="small",
                help="Commercial potential + buying intent + use-case maturity + waiting time",
            ),
            "name": st.column_config.TextColumn("Name", width="medium"),
            "company": st.column_config.TextColumn("Company", width="medium"),
            "received_at": st.column_config.DatetimeColumn(
                "Received", format="MMM D, h:mm a", width="medium"
            ),
            "waiting": st.column_config.TextColumn("Waiting", width="small"),
            "engagement": st.column_config.TextColumn("Engagement", width="medium"),
            "status": st.column_config.TextColumn("Status", width="medium"),
            "owner": st.column_config.TextColumn("Owner", width="medium"),
        },
    )

    if len(view):
        st.markdown("### Lead details & next action")
        lead_options = {
            f"{row['name']} · {row['company']} · score {int(row['score']):02d} · {row['waiting']} waiting": row["lead_id"]
            for _, row in view.iterrows()
        }
        selected_label = st.selectbox("Open lead", list(lead_options.keys()))
        lead_id = lead_options[selected_label]
        lead = df[df["lead_id"] == lead_id].iloc[0]

        st.markdown(
                f'<div class="priority-card"><h3>{lead["name"]} · {lead["company"]} · priority score {int(lead["score"])}/100</h3>'
            f'<div class="muted">{lead["engagement"]} &nbsp;·&nbsp; '
            f'{lead["waiting"]} waiting &nbsp;·&nbsp; {lead["owner"]}</div></div>',
            unsafe_allow_html=True,
        )

        detail_col, action_col = st.columns([1.65, 1])
        with detail_col:
            received_label = lead["received_at"].strftime("%b %d, %Y · %I:%M %p UTC")
            st.markdown("**Lead profile**")
            profile_contact, profile_account, profile_activity = st.columns(3)
            with profile_contact:
                st.caption("CONTACT")
                st.write(f"**{lead['name']}**")
                st.write(lead["email"])
                st.write(lead["job_title"])
                st.caption(f"Lead ID · {lead['lead_id']}")
            with profile_account:
                st.caption("ACCOUNT")
                st.write(f"**{lead['company']}**")
                st.write(f"{int(lead['company_size']):,} employees")
                st.write(lead["account_type"])
                st.caption(f"Persona · {lead['persona']}")
            with profile_activity:
                st.caption("ACTIVITY")
                st.write(f"**{lead['normalized_source']}**")
                st.write(f"Received {received_label}")
                st.write(f"{int(lead['account_activity_count'])} recent activities")
                st.caption(f"{int(lead['same_company_users'])} user(s) from this account")

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
                    f"- Lead quality contribution: {int(lead['qualification_points'])}/85 "
                    f"(the three-dimension subtotal scaled to 85%)"
                )
                st.markdown(
                    f"- Waiting time: +{int(lead['time_points'])}/15 "
                    f"({lead['waiting']} in the queue)"
                )
                if lead["self_serve"]:
                    st.markdown(
                        f"- Self-serve rule: lead quality below {SELF_SERVE_QUALITY_THRESHOLD} "
                        "and waiting time of at least 14 days"
                    )
                st.markdown(f"- Persona: {lead['persona']}")
                st.markdown(f"- Use-case stage: {lead['use_case_maturity']}")
                st.markdown(f"- Data completeness: {lead['confidence']}")

        with action_col:
            st.markdown("**Engagement and progress**")
            if lead["owner_id"] != "UNASSIGNED":
                engagement_note = f"Assigned to {lead['owner']} by score and available capacity."
            elif lead["engagement"] == "Queued — unassigned":
                engagement_note = (
                    "All sales capacity is currently full. This lead remains ranked and will be "
                    "assigned automatically when a slot opens."
                )
            else:
                engagement_note = "No sales owner is needed because this lead is handled through self-service."
            st.info(f"{lead['engagement']}: {engagement_note}")
            current_status = lead["status"] if lead["status"] in STATUS_ORDER else "In Progress"
            selected_status = st.selectbox(
                "Lead status", STATUS_ORDER, index=STATUS_ORDER.index(current_status), key=f"status_{lead_id}"
            )
            st.caption(
                "In Progress = actively being worked · Success = positive commercial outcome · "
                "Rejection = closed without progressing"
            )
            if st.button("Save lead", type="primary", width="stretch", key=f"save_{lead_id}"):
                row_index = st.session_state.leads.index[st.session_state.leads["lead_id"] == lead_id][0]
                st.session_state.leads.loc[row_index, "owner_id"] = lead["owner_id"]
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
    active_statuses = ["In Progress"]
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
    st.dataframe(pd.DataFrame(workload_rows), hide_index=True, width="stretch")

with add_tab:
    st.markdown("### Add and prioritize a lead")
    st.caption("Enter the information available at intake. The prototype will score and assign sales capacity.")
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
        new_row["owner_id"] = "UNASSIGNED"
        new_row["status"] = "In Progress"
        st.session_state.pending_lead = new_row

    if st.session_state.pending_lead is not None:
        pending = st.session_state.pending_lead
        candidate_leads = pd.concat(
            [st.session_state.leads, pd.DataFrame([pending])], ignore_index=True
        )
        candidate_df = build_lead_view(candidate_leads, team)
        result = candidate_df[candidate_df["lead_id"] == pending["lead_id"]].iloc[0]
        pending_owner = owner_names[result["owner_id"]]
        st.markdown(
            f'<div class="priority-card"><h3>Priority score {int(result["score"])}/100</h3>'
            f'<div class="muted">{result["engagement"]} · Owner: {pending_owner}</div></div>',
            unsafe_allow_html=True,
        )
        p1, p2, p3 = st.columns(3)
        p1.metric("Potential", f"{result['commercial_potential']}/40")
        p2.metric("Intent", f"{result['buying_intent']}/35")
        p3.metric("Use case", f"{result['use_case_score']}/25")
        add_col, clear_col = st.columns([1, 1])
        if add_col.button("Add to priority queue", type="primary", width="stretch"):
            st.session_state.leads = pd.concat(
                [st.session_state.leads, pd.DataFrame([pending])], ignore_index=True
            )
            st.session_state.pending_lead = None
            st.rerun()
        if clear_col.button("Discard", width="stretch"):
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

The three lead-quality dimensions are scaled to contribute **up to 85 points**. Waiting time contributes **up to 15 points**. Together they produce one final **priority score out of 100**, which determines queue order.

Sales capacity is assigned from the top of the ranked queue. Because salespeople are interchangeable in this version, each eligible lead goes to the person with the most capacity left. Leads that fall outside total team capacity are marked **Queued — unassigned** and receive the next available slot automatically.

A lead moves to **self-serve only when both conditions are true**: its unscaled lead-quality score is below 30 and it has waited at least 14 days. This prevents fresh but incomplete leads from being discarded too early.

Lead outcomes use three statuses: **In Progress** while the lead is actively being worked, **Success** after a positive commercial outcome, and **Rejection** when the lead is closed without progressing. Successes and rejections no longer consume active sales capacity, but retain the salesperson who handled them.

**The current weights are hypotheses, not proven truth.** Historical meetings, opportunities, and revenue outcomes would be used to recalibrate them.
        """
    )
    st.info("Rank every lead, use scarce sales capacity from the top down, and age low-quality leads into self-service.")

st.caption(
    "Prototype only · Team members are fictional · Changes are stored only for this browser session · "
    "Production integrations and score calibration are outside this demo."
)
