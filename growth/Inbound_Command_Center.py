
import importlib

import pandas as pd
import streamlit as st

import lead_engine


# Streamlit can rerun this page without refreshing modules imported earlier in
# the session. Reload the shared engine so this page and its output schema agree.
lead_engine = importlib.reload(lead_engine)
DATA_VERSION = lead_engine.DATA_VERSION
SELF_SERVE_QUALITY_THRESHOLD = lead_engine.SELF_SERVE_QUALITY_THRESHOLD
STATUS_ORDER = lead_engine.STATUS_ORDER
build_account_view = lead_engine.build_account_view
build_lead_view = lead_engine.build_lead_view
format_wait = lead_engine.format_wait
load_leads = lead_engine.load_leads
load_sales_team = lead_engine.load_sales_team
next_lead_id = lead_engine.next_lead_id
persist_owner_assignments = lead_engine.persist_owner_assignments
record_calendly_invites = lead_engine.record_calendly_invites

st.set_page_config(
    page_title="Prior Labs — Inbound Triage Copilot",
    page_icon="⚡",
    layout="wide",
)

ENGAGEMENT_OPTIONS = [
    "Sales assigned", "Queued — unassigned", "Self-serve", "Completed"
]

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
persist_owner_assignments(st.session_state.leads, df)
new_invites = record_calendly_invites(st.session_state.leads, df, team)
if new_invites:
    st.session_state.setdefault("email_outbox", []).extend(new_invites)
df = build_lead_view(st.session_state.leads, team)
# Keep existing browser sessions usable across the waiting-time schema change.
if "contact_waiting" not in df.columns:
    if "contact_waiting_hours" in df.columns:
        df["contact_waiting"] = df["contact_waiting_hours"].apply(format_wait)
    else:
        df["contact_waiting"] = df.get("waiting", "Not available")
df = df.sort_values(
    ["score", "received_at"], ascending=[False, True]
).reset_index(drop=True)
account_df = build_account_view(df)

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
    '<div class="hero-copy">Balance opportunity quality with response time, know who owns each lead, and move every company to a clear next step.</div>',
    unsafe_allow_html=True,
)
st.page_link("pages/1_Sales_Workspace.py", label="Open salesperson workspace", icon="👤")

sales_assigned_count = int((account_df["engagement"] == "Sales assigned").sum())
self_serve_count = int((account_df["engagement"] == "Self-serve").sum())
unassigned_count = int((account_df["engagement"] == "Queued — unassigned").sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Accounts", len(account_df), help=f"Resolved from {len(df)} inbound contacts")
k2.metric("Sales assigned", sales_assigned_count, help="Highest-priority accounts within team capacity")
k3.metric("Queued — unassigned", unassigned_count, help="Eligible accounts waiting for sales capacity")
k4.metric("Self-serve", self_serve_count, help="Low-quality leads waiting at least 14 days")

queue_tab, outcomes_tab, team_tab, add_tab, method_tab = st.tabs(
    ["Priority queue", "Outcomes", "Sales team", "Add a lead", "How it works"]
)

with queue_tab:
    st.markdown("### Account priority queue")
    st.caption("Multiple contacts and signals resolve to one account, owner, and time-aware priority.")

    f1, f2, f3, f4, f5 = st.columns([1.6, 1.25, 1.15, 1.05, 1.25])
    with f1:
        search = st.text_input(
            "Search", placeholder="Company name", key=f"search_{DATA_VERSION}"
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

    view = account_df.copy()
    if engagement_filter:
        view = view[view["engagement"].isin(engagement_filter)]
    if status_filter:
        view = view[view["status"].isin(status_filter)]
    if owner_filter:
        view = view[view["owner"].isin(owner_filter)]
    if search:
        view = view[
            view["account_name"].str.contains(
                search.strip(), case=False, na=False, regex=False
            )
        ]

    if sort_by == "Priority score — highest first":
        view = view.sort_values(
            ["score", "received_at"], ascending=[False, True]
        )
    elif sort_by == "Priority score — lowest first":
        view = view.sort_values(
            ["score", "received_at"], ascending=[True, True]
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
            "score", "account_name", "account_contact_count", "waiting",
            "engagement", "owner"
        ]],
        hide_index=True,
        width="stretch",
        height=420,
        column_config={
            "score": st.column_config.ProgressColumn(
                "Priority score", min_value=0, max_value=100, width="small",
                help="Account-level commercial potential + buying intent + use-case maturity",
            ),
            "account_name": st.column_config.TextColumn("Account", width="medium"),
            "account_contact_count": st.column_config.NumberColumn("Contacts", width="small"),
            "waiting": st.column_config.TextColumn("Account waiting", width="small"),
            "engagement": st.column_config.TextColumn("Engagement", width="medium"),
            "status": st.column_config.TextColumn("Status", width="medium"),
            "owner": st.column_config.TextColumn("Owner", width="medium"),
        },
    )

    if len(view):
        st.markdown("### Account details & next action")
        lead_options = {
            f"{row['account_name']} · score {int(row['score']):02d} · {int(row['account_contact_count'])} contact(s)": row["account_id"]
            for _, row in view.iterrows()
        }
        selected_label = st.selectbox("Open account", list(lead_options.keys()))
        account_id = lead_options[selected_label]
        account_contacts = df[df["account_id"] == account_id].copy()
        lead = account_contacts.sort_values(
            ["contact_qualification_score", "received_at"], ascending=[False, True]
        ).iloc[0]

        st.subheader(f"{lead['account_name']} · Priority Score {int(lead['score'])}/100")
        st.caption(
            f"{lead['engagement']} · "
            f"Account waiting {lead['waiting']} since earliest inbound · {lead['owner']}"
        )

        detail_col, action_col = st.columns([1.65, 1])
        with detail_col:
            received_label = (
                lead["received_at"].strftime("%b %d, %Y · %I:%M %p UTC")
                if pd.notna(lead["received_at"])
                else "Unknown"
            )
            st.markdown("**Account overview**")
            profile_account, profile_signals = st.columns(2)
            with profile_account:
                st.caption("ACCOUNT")
                st.write(f"**{lead['account_name']}**")
                st.write(f"{int(lead['company_size']):,} employees")
                st.write(lead["account_type"])
                st.caption(f"Account ID · {lead['account_id']}")
            with profile_signals:
                st.caption("COMBINED SIGNALS")
                st.write(f"**{int(lead['account_contact_count'])} known contact(s)**")
                st.write(f"{int(lead['account_activity_signal'])} recent account activities")

            st.markdown("**Primary contact**")
            primary_contact, primary_source = st.columns(2)
            with primary_contact:
                st.write(f"**{lead['name']}**")
                st.write(lead["email"])
                st.write(lead["job_title"])
            with primary_source:
                st.write(f"**{lead['normalized_source']}**")
                st.write(f"Received {received_label}")
                st.write(f"Contact waiting {lead['contact_waiting']}")

            st.markdown("**Primary inbound signal**")
            st.info(lead["inbound_text"])

            contact_table = account_contacts.copy()
            contact_table["contact_role"] = contact_table["lead_id"].apply(
                lambda value: "Primary" if value == lead["lead_id"] else "Supporting"
            )
            contact_table = contact_table.sort_values(
                ["contact_qualification_score", "received_at"], ascending=[False, True]
            )
            st.markdown("**Contacts at this account**")
            st.dataframe(
                contact_table[[
                    "contact_role", "name", "email", "job_title", "normalized_source",
                    "contact_qualification_score", "received_at", "contact_waiting", "inbound_text",
                ]],
                hide_index=True,
                width="stretch",
                column_config={
                    "contact_role": st.column_config.TextColumn("Role", width="small"),
                    "name": st.column_config.TextColumn("Contact", width="medium"),
                    "email": st.column_config.TextColumn("Email", width="medium"),
                    "job_title": st.column_config.TextColumn("Title", width="medium"),
                    "normalized_source": st.column_config.TextColumn("Source", width="medium"),
                    "contact_qualification_score": st.column_config.NumberColumn(
                        "Individual signal", width="small"
                    ),
                    "received_at": st.column_config.DatetimeColumn(
                        "Received", format="MMM D, h:mm a", width="medium"
                    ),
                    "contact_waiting": st.column_config.TextColumn(
                        "Contact waiting", width="small"
                    ),
                    "inbound_text": st.column_config.TextColumn("Inbound signal", width="large"),
                },
            )
            s1, s2, s3 = st.columns(3)
            s1.metric("Potential", f"{int(lead['commercial_potential'])}/40")
            s2.metric("Intent", f"{int(lead['buying_intent'])}/35")
            s3.metric("Use case", f"{int(lead['use_case_score'])}/25")
            with st.expander("Why this priority?"):
                for reason in lead["why"].split(" | "):
                    st.markdown(f"- {reason}")
                st.markdown(
                    f"- Account quality contribution: {int(lead['qualification_points'])}/85 "
                    f"from an unscaled quality score of {int(lead['qualification_score'])}/100"
                )
                st.markdown(
                    f"- Waiting-time contribution: +{int(lead['time_points'])}/15 "
                    f"after {lead['waiting']}"
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
                    "All sales capacity is currently full. This account remains ranked and will be "
                    "assigned automatically when a slot opens."
                )
            else:
                engagement_note = "No sales owner is needed because this lead is handled through self-service."
            st.info(f"{lead['engagement']}: {engagement_note}")
            st.write(f"**Current status:** {lead['status']}")
            st.caption(
                "Salespeople record milestones, rejection reasons, opportunities, and next actions "
                "in the salesperson workspace."
            )
            st.page_link(
                "pages/1_Sales_Workspace.py",
                label="Open salesperson workspace to update",
                icon="👤",
                width="stretch",
            )
    else:
        st.info("No leads match these filters. Clear a filter to restore the queue.")

with outcomes_tab:
    st.markdown("### Decision history")
    st.caption(
        "Review completed account decisions and the evidence captured when each decision was made."
    )
    completed = account_df[account_df["status"].isin(["Success", "Rejected"])].copy()
    successes = completed[completed["status"] == "Success"].copy()
    rejections = completed[completed["status"] == "Rejected"].copy()

    success_metric, rejection_metric = st.columns(2)
    success_metric.metric("Successes", len(successes))
    rejection_metric.metric("Rejections", len(rejections))

    for decisions in (successes, rejections):
        decisions["decision_at"] = pd.to_datetime(
            decisions["outcome_recorded_at"], utc=True, errors="coerce"
        )
        decisions.sort_values(
            "decision_at", ascending=False, na_position="last", inplace=True
        )

    common_column_config = {
        "account_name": st.column_config.TextColumn("Account", width="medium"),
        "name": st.column_config.TextColumn("Primary contact", width="medium"),
        "owner": st.column_config.TextColumn("Sales owner", width="medium"),
        "normalized_source": st.column_config.TextColumn("Source", width="medium"),
        "qualification_score_at_outcome": st.column_config.NumberColumn(
            "Quality score at decision", format="%d"
        ),
        "priority_score_at_outcome": st.column_config.NumberColumn(
            "Priority score at decision", format="%d"
        ),
        "pipeline_stage": st.column_config.TextColumn("Final stage", width="medium"),
        "decision_at": st.column_config.DatetimeColumn(
            "Decision recorded", format="MMM D, YYYY, h:mm a"
        ),
    }

    st.markdown("#### Success")
    if successes.empty:
        st.caption("No successful account decisions have been recorded yet.")
    st.dataframe(
        successes[[
            "account_name", "name", "owner", "normalized_source",
            "qualification_score_at_outcome", "priority_score_at_outcome",
            "pipeline_stage", "estimated_value_band", "decision_at",
        ]],
        hide_index=True,
        width="stretch",
        column_config={
            **common_column_config,
            "estimated_value_band": st.column_config.TextColumn(
                "Customer value", width="medium"
            ),
        },
    )

    st.markdown("#### Rejected")
    if rejections.empty:
        st.caption("No rejected account decisions have been recorded yet.")
    st.dataframe(
        rejections[[
            "account_name", "name", "owner", "normalized_source",
            "qualification_score_at_outcome", "priority_score_at_outcome",
            "pipeline_stage", "disposition_reason", "disposition_notes", "decision_at",
        ]],
        hide_index=True,
        width="stretch",
        column_config={
            **common_column_config,
            "disposition_reason": st.column_config.TextColumn(
                "Rejection reason", width="medium"
            ),
            "disposition_notes": st.column_config.TextColumn(
                "Decision notes", width="large"
            ),
        },
    )

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
        active = int(person_leads["account_id"].nunique())
        capacity = int(person["weekly_capacity"])
        workload_rows.append({
            "Salesperson": person["name"], "Role": person["role"], "Region": person["region"],
            "Active accounts": active, "Weekly capacity": capacity,
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
            activity = st.number_input("Recent company activities", min_value=1, value=2)
            inbound_text = st.text_area(
                "What they said or did",
                "We are evaluating TabPFN against LightGBM for a production demand forecasting use case.",
            )
        submitted = st.form_submit_button("Score this lead", type="primary")

    if submitted:
        next_id = next_lead_id(st.session_state.leads)
        new_row = {
            "lead_id": next_id, "name": name, "email": email, "company": company,
            "company_size": int(company_size), "job_title": title, "source": source,
            "inbound_time": pd.Timestamp.now(tz="UTC").isoformat(),
            "account_activity_count": int(activity),
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

1. **Commercial potential** — How valuable could the company be?
2. **Buying intent** — What evidence suggests active commercial evaluation?
3. **Use-case maturity** — How concrete and deployment-ready is the problem?

The three evidence dimensions first produce a **lead-quality score out of 100**. Quality is scaled to contribute up to **85 points**, and waiting time contributes up to **15 points**, creating one final priority score.

Contacts sharing an account are resolved into one commercial opportunity, one owner, and one capacity slot. Existing ownership remains stable. Whenever capacity opens, the highest-scoring eligible unassigned account receives the slot.

The time component is deliberately bounded, so urgency can move an older account up the running queue without overwhelming commercial quality. Low-quality accounts move to self-serve only after 14 days.

Lead outcomes use three statuses: **In Progress** while the lead is actively being worked, **Success** after a positive commercial outcome, and **Rejected** when the lead is closed without progressing. Successes and rejected leads no longer consume active sales capacity, but retain the salesperson who handled them.

**The current weights are hypotheses, not proven truth.** Historical meetings, opportunities, and revenue outcomes would be used to recalibrate them.
        """
    )
    st.info("Resolve signals to accounts, rank one running queue, allocate available capacity, and learn from outcomes.")

st.caption(
    "Prototype only · Team members are fictional · Changes are stored only for this browser session · "
    "Production integrations and score calibration are outside this demo."
)
