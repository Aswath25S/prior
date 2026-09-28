import importlib

import pandas as pd
import streamlit as st

import lead_engine
from call_prep import call_prep_fingerprint, generate_call_prep


# Streamlit can retain imported modules while switching pages during development.
# Reload the shared engine so newly added workflow fields are available immediately.
lead_engine = importlib.reload(lead_engine)
DATA_VERSION = lead_engine.DATA_VERSION
MEETING_STATUS_ORDER = lead_engine.MEETING_STATUS_ORDER
REJECTION_REASONS = lead_engine.REJECTION_REASONS
STATUS_ORDER = lead_engine.STATUS_ORDER
VALUE_BANDS = lead_engine.VALUE_BANDS
build_lead_view = lead_engine.build_lead_view
build_account_view = lead_engine.build_account_view
format_wait = lead_engine.format_wait
load_leads = lead_engine.load_leads
load_sales_team = lead_engine.load_sales_team
record_calendly_invites = lead_engine.record_calendly_invites
persist_owner_assignments = lead_engine.persist_owner_assignments
sync_meeting_statuses = lead_engine.sync_meeting_statuses
update_lead_progress = lead_engine.update_lead_progress


st.set_page_config(
    page_title="Prior Labs — Sales Workspace",
    page_icon="👤",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1320px; padding-top: 5rem; padding-bottom: 3rem;}
    @media (max-width: 768px) {.block-container {padding-top: 4.5rem;}}
    [data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, .28);
        padding: 1rem;
        border-radius: 12px;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
        color: var(--text-color) !important;
    }
    [data-testid="stMetricLabel"] {opacity: .75;}
    .workspace-label {
        display: inline-block;
        color: var(--text-color) !important;
        background: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, .35);
        border-radius: 999px;
        padding: .3rem .65rem;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.get("data_version") != DATA_VERSION or "leads" not in st.session_state:
    st.session_state.leads = load_leads()
    st.session_state.pending_lead = None
    st.session_state.data_version = DATA_VERSION

team = load_sales_team()
lead_view = build_lead_view(st.session_state.leads, team)
persist_owner_assignments(st.session_state.leads, lead_view)
new_invites = record_calendly_invites(st.session_state.leads, lead_view, team)
if new_invites:
    st.session_state.setdefault("email_outbox", []).extend(new_invites)
lead_view = build_lead_view(st.session_state.leads, team)
if "contact_waiting" not in lead_view.columns:
    if "contact_waiting_hours" in lead_view.columns:
        lead_view["contact_waiting"] = lead_view["contact_waiting_hours"].apply(format_wait)
    else:
        lead_view["contact_waiting"] = lead_view.get("waiting", "Not available")

st.markdown('<div class="workspace-label">Prior Labs · Sales</div>', unsafe_allow_html=True)
st.title("My lead workspace")
st.caption("Work the highest-priority assigned accounts and record evidence-rich outcomes.")

salesperson_names = team["name"].tolist()
selected_name = st.selectbox("Salesperson", salesperson_names, key="sales_workspace_person")
person = team[team["name"] == selected_name].iloc[0]
person_id = person["salesperson_id"]
person_leads = lead_view[lead_view["owner_id"] == person_id].copy()
person_accounts = build_account_view(person_leads) if not person_leads.empty else person_leads.copy()

active_count = int((person_accounts["status"] == "In Progress").sum())
success_count = int((person_accounts["status"] == "Success").sum())
rejected_count = int((person_accounts["status"] == "Rejected").sum())
capacity = int(person["weekly_capacity"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Active accounts", active_count)
m2.metric("Available capacity", max(capacity - active_count, 0))
m3.metric("Successes", success_count)
m4.metric("Rejected", rejected_count)

st.markdown("### Lead portfolio")
search_col, status_col, sort_col = st.columns([1.6, 1, 1.25])
with search_col:
    search = st.text_input("Search", placeholder="Name, company, or title", key="sales_lead_search")
with status_col:
    status_filter = st.multiselect(
        "Status", STATUS_ORDER, default=[], help="Leave empty to show every status.",
        key="sales_status_filter",
    )
with sort_col:
    sort_by = st.selectbox(
        "Sort by",
        [
            "Priority score — highest first", "Waiting time — longest first",
            "Name — A to Z", "Company — A to Z",
        ],
        key="sales_sort",
    )

portfolio = person_accounts.copy()
if status_filter:
    portfolio = portfolio[portfolio["status"].isin(status_filter)]
if search:
    query = search.lower()
    portfolio = portfolio[
        portfolio["company"].str.lower().str.contains(query, na=False)
        | portfolio["name"].str.lower().str.contains(query, na=False)
        | portfolio["job_title"].str.lower().str.contains(query, na=False)
    ]
if sort_by == "Waiting time — longest first":
    portfolio = portfolio.sort_values(["waiting_hours", "score"], ascending=[False, False])
elif sort_by == "Name — A to Z":
    portfolio = portfolio.sort_values("name", key=lambda values: values.str.lower())
elif sort_by == "Company — A to Z":
    portfolio = portfolio.sort_values("company", key=lambda values: values.str.lower())
else:
    portfolio = portfolio.sort_values(["score", "received_at"], ascending=[False, True])

st.dataframe(
    portfolio[[
        "score", "name", "account_name", "pipeline_stage", "waiting",
        "next_action", "status",
    ]],
    hide_index=True,
    width="stretch",
    height=360,
    column_config={
        "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, width="small"),
        "name": st.column_config.TextColumn("Name", width="medium"),
        "account_name": st.column_config.TextColumn("Account", width="medium"),
        "pipeline_stage": st.column_config.TextColumn("Stage", width="medium"),
        "waiting": st.column_config.TextColumn("Account waiting", width="small"),
        "next_action": st.column_config.TextColumn("Next action", width="large"),
        "status": st.column_config.TextColumn("Progress", width="small"),
    },
)

if len(portfolio):
    st.markdown("### Update a lead")
    lead_options = {
        f"{row['name']} · {row['company']} · score {int(row['score']):02d}": row["lead_id"]
        for _, row in portfolio.iterrows()
    }
    selected_label = st.selectbox("Lead", list(lead_options), key="sales_selected_lead")
    lead_id = lead_options[selected_label]
    lead = lead_view[lead_view["lead_id"] == lead_id].iloc[0]
    received_label = (
        lead["received_at"].strftime("%b %d, %Y · %I:%M %p UTC")
        if pd.notna(lead["received_at"])
        else "Unknown"
    )
    current_meeting_status = (
        lead["meeting_status"]
        if lead["meeting_status"] in MEETING_STATUS_ORDER
        else "Invite sent"
    )

    context_col, update_col = st.columns([1.7, 1])
    with context_col:
        st.markdown(f"#### {lead['name']} · {lead['company']}")
        profile_contact, profile_account, profile_activity = st.columns(3)
        with profile_contact:
            st.caption("CONTACT")
            st.write(f"**{lead['name']}**")
            st.write(lead["email"])
            st.write(lead["job_title"])
        with profile_account:
            st.caption("COMPANY")
            st.write(f"**{int(lead['company_size']):,} employees**")
            st.write(lead["account_type"])
            st.write(f"Lead ID · {lead['lead_id']}")
        with profile_activity:
            st.caption("ACTIVITY")
            st.write(f"**{lead['normalized_source']}**")
            st.write(received_label)
            st.write(f"Contact waiting · {lead['contact_waiting']}")
            st.write(
                f"{int(lead['account_activity_signal'])} activities · "
                f"{int(lead['account_contact_count'])} known contact(s)"
            )
        st.info(lead["inbound_text"])

        if current_meeting_status in {"Scheduled", "In progress"}:
            st.markdown("### Call preparation")
            fingerprint = call_prep_fingerprint(lead)
            briefs = st.session_state.setdefault("call_prep_briefs", {})
            cached = briefs.get(lead_id)
            should_generate = not cached or cached.get("fingerprint") != fingerprint
            if should_generate:
                with st.spinner("Preparing the call brief…"):
                    brief, metadata = generate_call_prep(lead)
                cached = {
                    "fingerprint": fingerprint,
                    "brief": brief,
                    "metadata": metadata,
                }
                briefs[lead_id] = cached

            brief = cached["brief"]
            metadata = cached["metadata"]
            generated_at = pd.to_datetime(
                metadata["generated_at"], utc=True, errors="coerce"
            )
            generated_label = (
                generated_at.strftime("%b %d, %Y · %I:%M %p UTC")
                if pd.notna(generated_at)
                else "Unknown"
            )
            mode_label = "AI-generated" if metadata["mode"] == "ai" else "Fact-based preview"
            st.caption(f"{mode_label} · Generated {generated_label}")
            st.info(brief["summary"])

            st.markdown("**Meeting objective**")
            st.write(brief["meeting_objective"])
            st.markdown("**Likely use case**")
            st.write(brief["use_case_hypothesis"])

            signals_col, gaps_col = st.columns(2)
            with signals_col:
                st.markdown("**Buying signals**")
                for signal in brief["buying_signals"]:
                    st.markdown(f"- {signal}")
            with gaps_col:
                st.markdown("**Information to confirm**")
                for gap in brief["information_gaps"]:
                    st.markdown(f"- {gap}")

            with st.expander("Discovery questions", expanded=True):
                for index, question in enumerate(brief["discovery_questions"], start=1):
                    st.markdown(f"{index}. {question}")
            with st.expander("Suggested agenda"):
                for index, agenda_item in enumerate(brief["suggested_agenda"], start=1):
                    st.markdown(f"{index}. {agenda_item}")
            with st.expander("Risks to address"):
                for risk in brief["risks"]:
                    st.markdown(f"- {risk}")

            st.markdown("**Desired next step**")
            st.success(brief["desired_next_step"])
            st.caption(metadata["notice"])
            if st.button("Regenerate call brief", width="stretch", key=f"regenerate_{lead_id}"):
                with st.spinner("Regenerating the call brief…"):
                    brief, metadata = generate_call_prep(lead)
                briefs[lead_id] = {
                    "fingerprint": fingerprint,
                    "brief": brief,
                    "metadata": metadata,
                }
                st.rerun()
    with update_col:
        st.markdown("#### Progress")
        current_status = lead["status"] if lead["status"] in STATUS_ORDER else "In Progress"
        updated_status = st.selectbox(
            "Current progress",
            STATUS_ORDER,
            index=STATUS_ORDER.index(current_status),
            key=f"outcome_{lead_id}",
        )
        success_milestone = ""
        pipeline_stage = str(lead["pipeline_stage"] or "Initial follow-up")
        disposition_reason = ""
        disposition_notes = ""
        opportunity_created = False
        opportunity_stage = ""
        estimated_value_band = ""

        if updated_status == "In Progress":
            pipeline_stage = st.text_input(
                "Current stage",
                value=pipeline_stage,
                key=f"pipeline_stage_{lead_id}",
                placeholder="e.g. Security review, technical discovery, or POC planning",
            )
            if pipeline_stage.strip() and pipeline_stage.strip().lower() != "initial follow-up":
                current_value = lead["estimated_value_band"]
                value_index = VALUE_BANDS.index(current_value) if current_value in VALUE_BANDS else 0
                estimated_value_band = st.selectbox(
                    "Estimated opportunity size (optional)",
                    VALUE_BANDS,
                    index=value_index,
                    key=f"value_band_{lead_id}",
                )
        elif updated_status == "Success":
            success_milestone = "Commercial win"
            pipeline_stage = "Customer"
            opportunity_created = True
            opportunity_stage = "Closed won"
            st.success("Success means this account has become a customer.")
            current_value = lead["estimated_value_band"]
            value_index = VALUE_BANDS.index(current_value) if current_value in VALUE_BANDS else 0
            estimated_value_band = st.selectbox(
                "Customer value (optional)",
                VALUE_BANDS,
                index=value_index,
                key=f"value_band_{lead_id}",
            )
        elif updated_status == "Rejected":
            pipeline_stage = str(lead["pipeline_stage"] or "Initial follow-up")
            current_reason = lead["disposition_reason"]
            reason_index = (
                REJECTION_REASONS.index(current_reason)
                if current_reason in REJECTION_REASONS
                else 0
            )
            disposition_reason = st.selectbox(
                "Why was it closed?",
                REJECTION_REASONS,
                index=reason_index,
                key=f"reason_{lead_id}",
            )
            disposition_notes = st.text_area(
                "Outcome notes",
                value=str(lead["disposition_notes"] or ""),
                key=f"notes_{lead_id}",
                placeholder="What should the team learn from this outcome?",
            )

        if updated_status != "Rejected":
            next_action = st.text_input(
                "Next action (optional)",
                value=str(lead["next_action"] or ""),
                key=f"next_action_{lead_id}",
                placeholder="Send security pack and confirm evaluation owner",
            )
            existing_due = pd.to_datetime(lead["next_action_due"], errors="coerce")
            next_action_due = st.date_input(
                "Next-action due date",
                value=existing_due.date() if pd.notna(existing_due) else None,
                key=f"next_action_due_{lead_id}",
            )
        else:
            next_action = ""
            next_action_due = ""
        st.markdown("#### Meeting")
        sent_at = pd.to_datetime(lead["calendly_sent_at"], utc=True, errors="coerce")
        sent_label = sent_at.strftime("%b %d, %Y · %I:%M %p UTC") if pd.notna(sent_at) else "Not sent"
        meeting_start = pd.to_datetime(lead["meeting_start_at"], utc=True, errors="coerce")
        st.info(f"Meeting status: **{current_meeting_status}**")
        if pd.notna(sent_at):
            st.caption(f"Calendly invitation sent to {lead['email']} · {sent_label}")
        elif lead["email"] == "Not provided":
            st.caption("Calendly invitation not sent because no email address is available.")
        else:
            st.caption("Calendly invitation has not been sent.")
        if (
            current_meeting_status in {"Scheduled", "In progress"}
            and pd.notna(meeting_start)
            and lead["meeting_join_url"]
        ):
            meeting_label = meeting_start.strftime("%b %d, %Y · %I:%M %p UTC")
            st.success(f"Meeting scheduled for {meeting_label}")
            st.link_button(
                f"Join meeting · {meeting_label}", lead["meeting_join_url"], width="stretch"
            )
        elif lead["calendly_url"]:
            st.link_button("Open salesperson Calendly", lead["calendly_url"], width="stretch")

        with st.expander("Demo: simulate a Calendly booking"):
            default_start = (
                pd.Timestamp.now(tz="UTC").normalize()
                + pd.Timedelta(days=1, hours=10)
            )
            booking_date = st.date_input(
                "Meeting date", value=default_start.date(), key=f"booking_date_{lead_id}"
            )
            booking_time = st.time_input(
                "Meeting time (UTC)", value=default_start.time(), key=f"booking_time_{lead_id}"
            )
            join_url = st.text_input(
                "Join URL",
                value=f"https://meet.google.com/lookup/prior-{lead_id.lower()}",
                key=f"join_url_{lead_id}",
            )
            if st.button("Record demo booking", width="stretch", key=f"book_{lead_id}"):
                row_index = st.session_state.leads.index[
                    st.session_state.leads["lead_id"] == lead_id
                ][0]
                scheduled_start = pd.Timestamp.combine(booking_date, booking_time).tz_localize("UTC")
                st.session_state.leads.loc[
                    row_index, "meeting_start_at"
                ] = scheduled_start.isoformat()
                st.session_state.leads.loc[
                    row_index, "meeting_end_at"
                ] = (scheduled_start + pd.Timedelta(minutes=30)).isoformat()
                st.session_state.leads.loc[row_index, "meeting_join_url"] = join_url
                sync_meeting_statuses(st.session_state.leads)
                st.rerun()

        st.caption(
            "Meeting status is derived automatically from Calendly event times. "
            "The demo booking control stands in for a production webhook."
        )
        if st.button("Save progress", type="primary", width="stretch", key=f"save_{lead_id}"):
            update_lead_progress(
                st.session_state.leads,
                lead,
                updated_status,
                pipeline_stage=pipeline_stage,
                success_milestone=success_milestone,
                disposition_reason=disposition_reason,
                disposition_notes=disposition_notes,
                opportunity_created=opportunity_created,
                opportunity_stage=opportunity_stage,
                estimated_value_band=estimated_value_band,
                next_action=next_action,
                next_action_due=next_action_due,
            )
            st.success("Progress and structured outcome data saved. Capacity has been recalculated.")
            st.rerun()
else:
    st.info("No leads match these filters for this salesperson.")

st.caption("Prototype only · Changes are stored for this browser session.")
