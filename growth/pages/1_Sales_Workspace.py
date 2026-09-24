import streamlit as st

from lead_engine import DATA_VERSION, STATUS_ORDER, build_lead_view, load_leads, load_sales_team


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

st.markdown('<div class="workspace-label">Prior Labs · Sales</div>', unsafe_allow_html=True)
st.title("My lead workspace")
st.caption("Track assigned leads, focus on the highest score, and record the commercial outcome.")

salesperson_names = team["name"].tolist()
selected_name = st.selectbox("Salesperson", salesperson_names, key="sales_workspace_person")
person = team[team["name"] == selected_name].iloc[0]
person_id = person["salesperson_id"]
person_leads = lead_view[lead_view["owner_id"] == person_id].copy()

active_count = int((person_leads["status"] == "In Progress").sum())
success_count = int((person_leads["status"] == "Success").sum())
rejection_count = int((person_leads["status"] == "Rejection").sum())
capacity = int(person["weekly_capacity"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Active leads", active_count)
m2.metric("Available capacity", max(capacity - active_count, 0))
m3.metric("Successes", success_count)
m4.metric("Rejections", rejection_count)

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

portfolio = person_leads.copy()
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
    portfolio[["score", "name", "company", "waiting", "engagement", "status"]],
    hide_index=True,
    use_container_width=True,
    height=360,
    column_config={
        "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, width="small"),
        "name": st.column_config.TextColumn("Name", width="medium"),
        "company": st.column_config.TextColumn("Company", width="medium"),
        "waiting": st.column_config.TextColumn("Waiting", width="small"),
        "engagement": st.column_config.TextColumn("Engagement", width="medium"),
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
    received_label = lead["received_at"].strftime("%b %d, %Y · %I:%M %p UTC")

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
            st.caption("ACCOUNT")
            st.write(f"**{int(lead['company_size']):,} employees**")
            st.write(lead["account_type"])
            st.write(f"Lead ID · {lead['lead_id']}")
        with profile_activity:
            st.caption("ACTIVITY")
            st.write(f"**{lead['normalized_source']}**")
            st.write(received_label)
            st.write(
                f"{int(lead['account_activity_count'])} activities · "
                f"{int(lead['same_company_users'])} account user(s)"
            )
        st.info(lead["inbound_text"])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Priority", f"{int(lead['score'])}/100")
        c2.metric("Potential", f"{int(lead['commercial_potential'])}/40")
        c3.metric("Intent", f"{int(lead['buying_intent'])}/35")
        c4.metric("Use case", f"{int(lead['use_case_score'])}/25")
    with update_col:
        st.markdown("#### Progress")
        current_status = lead["status"] if lead["status"] in STATUS_ORDER else "In Progress"
        updated_status = st.selectbox(
            "Outcome", STATUS_ORDER, index=STATUS_ORDER.index(current_status), key=f"outcome_{lead_id}"
        )
        st.caption(
            "In Progress = active · Success = positive commercial outcome · "
            "Rejection = closed without progressing"
        )
        if st.button("Save progress", type="primary", use_container_width=True, key=f"save_{lead_id}"):
            row_index = st.session_state.leads.index[st.session_state.leads["lead_id"] == lead_id][0]
            st.session_state.leads.loc[row_index, "owner_id"] = lead["owner_id"]
            st.session_state.leads.loc[row_index, "status"] = updated_status
            st.success("Progress updated. Capacity and assignments have been recalculated.")
            st.rerun()
else:
    st.info("No leads match these filters for this salesperson.")

st.caption("Prototype only · Changes are stored for this browser session.")
