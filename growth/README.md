# Inbound Triage Copilot

An explainable Streamlit prototype for turning inbound contacts into an account-level priority queue, assigning limited sales capacity, and recording commercial outcomes.

The project is deliberately rules-based. Its scoring and routing logic is visible in Python, works without external services, and is intended to be inspected, challenged, and recalibrated with real outcome data. Optional OpenAI integration generates call-preparation briefs for booked meetings.

> **Unofficial prototype:** this project is not affiliated with Prior Labs and is not a replacement for a CRM, marketing-automation system, calendar provider, or email service.

## Features

- Resolve multiple contacts into a single company opportunity.
- Rank companies using transparent commercial, intent, use-case, and waiting-time rules.
- Select a primary contact separately from company qualification.
- Assign the highest-ranked eligible companies within salesperson capacity.
- Route old, low-quality opportunities to self-service and keep them at the bottom of work queues.
- Track `In Progress`, `Success`, and `Rejected` outcomes.
- Simulate Calendly invitations and meeting lifecycle changes without sending real email.
- Generate structured call preparation through OpenAI when configured, with a deterministic local fallback.
- Preserve decision-time scores for later outcome analysis.

All committed names, companies, contact details, activities, and meeting information are fictional.

## Quick start

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r growth/requirements.txt
streamlit run growth/Inbound_Command_Center.py
```

Streamlit exposes both pages through its page navigation:

- **Inbound Command Center**: account prioritization, team capacity, outcomes, and intake.
- **Sales Workspace**: salesperson portfolio, meeting state, progress updates, and call preparation.

## Optional OpenAI configuration

The core queue, scoring, assignment, and meeting simulation do not require an API key.

To enable AI-generated call preparation:

```bash
cp growth/.env.example growth/.env
```

Then edit the ignored `growth/.env` file:

```dotenv
OPENAI_API_KEY="your-key"
OPENAI_CALL_PREP_MODEL="gpt-6-astra"
```

Restart Streamlit after changing the file. Call preparation uses the Responses API with a strict JSON schema and `store=False`. It sends the selected fictional/demo lead context only when a scheduled or in-progress meeting needs a brief. API failures fall back to a local fact-based brief.

## How scoring works

The application keeps three concepts separate:

1. **Primary-contact score (0–100)** answers who the salesperson should approach. It uses authority (45), role relevance (25), source (20), and contactability (10). Company size, company activity, number of contacts, and inbound-message content are excluded.
2. **Qualification score (0–100)** estimates opportunity quality from commercial potential (40), buying intent (35), and use-case maturity (25). A company's qualification score is the highest qualification score among its contacts.
3. **Priority score (0–100)** controls queue order. Qualification contributes up to 85 points and waiting time contributes up to 15.

Waiting-time points are bounded at day thresholds: 0, 2, 4, 7, 11, or 15. A company becomes self-serve only when its qualification score is below 30 and it has waited at least seven days.

These weights are hypotheses. A production implementation should calibrate them against meetings, qualified opportunities, cycle time, conversion, and revenue—not treat them as universal truth.

## Data model

`sample_leads.csv` contains contact-level input records. Important fields include:

| Field | Meaning |
| --- | --- |
| `lead_id` | Stable contact identifier |
| `name`, `email`, `job_title` | Contact identity and role evidence |
| `company`, `company_size` | Company-resolution and commercial-potential inputs |
| `source` | Normalized acquisition source |
| `inbound_time` | Timezone-aware arrival timestamp |
| `account_activity_count` | Company activity signal captured with the contact |
| `inbound_text` | Optional free-text evidence; usually absent in the fixture |
| `owner_id` | Assigned salesperson or `UNASSIGNED` |
| `status` | `In Progress`, `Success`, or `Rejected` |

The prototype derives an account identifier primarily from business email domains. Generic email addresses and records labelled `Independent` or `Not provided` remain separate. Production systems should use CRM account IDs and reviewed identity-resolution rules.

Missing fields use explicit conservative fallbacks. Missing inbound text is classified as `Not observed` rather than negative intent. Invalid timestamps remain unknown and contribute no waiting-time points.

To keep the public demo useful over time, `load_leads()` rebases only the bundled fictional timestamps so the newest contact is 16 hours old while preserving every interval and ordering relationship in the CSV. Pass `rebase_demo_times=False` when inspecting the raw fixture. User-created records and timestamps supplied to the scoring engine are never shifted.

## Assignment behavior

- Only active, non-self-serve companies are eligible.
- Companies are ranked by priority score; earliest inbound time breaks a tie.
- Each company consumes one capacity slot regardless of contact count.
- Existing owners are retained when the company remains above the capacity cutoff and that salesperson still has capacity.
- Remaining selected companies go to the salesperson with the most capacity left.
- Closing an account releases its slot for the next eligible company.

All UI edits live in Streamlit session state. CSV files are demo fixtures, not a durable database.

## Project structure

```text
growth/
├── Inbound_Command_Center.py   # Main Streamlit application
├── pages/1_Sales_Workspace.py  # Salesperson workflow
├── lead_engine.py              # Data preparation, scoring, routing, and outcomes
├── call_prep.py                # Structured OpenAI call prep and local fallback
├── sample_leads.csv            # Fictional contact fixture
├── sales_team.csv              # Fictional capacity fixture
├── requirements.txt
└── tests/test_lead_engine.py
```

The UI should consume derived fields from `lead_engine.py`; avoid duplicating scoring or assignment logic inside Streamlit pages.

## Tests

From the repository root:

```bash
python -m unittest discover -s growth/tests -t .
```

The tests cover account grouping, capacity allocation, primary-contact selection, missing fields, waiting-time thresholds, self-service routing, meeting state, and outcome updates.

## A useful walkthrough

1. Open the priority queue and inspect a multi-contact company such as ACME or Northstar.
2. Compare the qualification-driving contact with the selected primary contact.
3. Review the score explanation and waiting-time contribution.
4. Inspect team capacity and the unassigned queue.
5. Open the Sales Workspace, simulate a booking, and update an outcome.
6. Return to Outcomes to inspect the captured decision record.

## Extension points

Good next contributions include:

- Extract scoring weights and thresholds into validated configuration.
- Replace CSV/session state with a repository and database layer.
- Add pluggable CRM identity resolution and deduplication.
- Add real email/calendar adapters behind explicit interfaces and test doubles.
- Introduce authentication, authorization, audit logs, and tenant isolation.
- Calibrate ranking rules with historical outcomes and fairness checks.
- Add accessibility and end-to-end browser tests.

See the repository [roadmap](../docs/ROADMAP.md) and [contribution guide](../CONTRIBUTING.md).

## Privacy and production limitations

Do not load real customer or prospect information without appropriate authorization and a reviewed data-processing design. The current application lacks durable storage, access control, encryption configuration, deletion workflows, observability, rate limiting, and compliance controls. Simulated email and calendar actions are clearly labelled and do not contact anyone.
