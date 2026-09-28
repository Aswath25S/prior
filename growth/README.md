# Prior Labs — Inbound Triage Copilot

A lightweight working prototype built voluntarily as additional material for a CEO interview. It demonstrates operating judgment and a testable commercial workflow; it is not presented as a production system or a replacement for Prior Labs' CRM.

## What it demonstrates

1. A unified account queue across SAP, website, API, GitHub, and other sources, with multiple contacts resolved into one commercial opportunity.
2. Transparent prioritization on three dimensions:
   - Commercial potential (0–40)
   - Buying intent (0–35)
   - Use-case maturity (0–25)
3. One account priority score combining lead quality (up to 85 points) with bounded waiting-time urgency (up to 15 points).
4. A score-sorted running queue with a recommended route, stable owner, and visible supporting contacts.
5. An explanation panel showing *why* each lead received its score.
6. Stable account-level allocation of limited sales capacity; contacts at the same company share one owner and consume one slot.
7. An illustrative commercial-team view showing workload and capacity.
8. Editable progress controls for the selected lead.
9. A form to simulate a new inbound lead live during the demo.
10. A self-serve rule for leads with quality below 30 that have waited at least 14 days.
11. Three clear lifecycle statuses: In Progress, Success, and Rejected. Qualified opportunities and evaluations/POCs remain In Progress; Success is reserved for accounts that become customers.
12. A separate salesperson workspace for reviewing an individual portfolio and updating outcomes.
13. Automatic demo Calendly emails on assignment, with event-driven meeting status and a future-meeting join button in the salesperson workspace.
14. Automatic call preparation for booked meetings, using structured OpenAI output when configured and a fact-based fallback otherwise.
15. Structured outcome capture for qualification milestones, rejection reasons, opportunity stage/value, next action, and score-at-decision.
16. An outcome-learning view comparing qualification and opportunity creation across inbound sources.

The names in `sales_team.csv` are fictional and exist only to demonstrate the operating workflow. All salespeople are interchangeable in this first version; initial calls are assigned to whoever has the most capacity left.
Sample arrival times are parsed from their recorded timezone offsets and converted to UTC for calculation; they are never shifted when the app starts. Active waiting time runs from the account's earliest valid inbound timestamp to the current time. Closed accounts stop accumulating waiting time at their recorded outcome time.
The fixture intentionally includes missing fields, inconsistent company names, shared inboxes, generic emails, approximate sizes, intermediaries, academics, and duplicate enterprise contacts. This makes data-quality assumptions visible during the demo instead of presenting unrealistically clean CRM data.
Missing fields receive explicit fallbacks such as `Unknown contact`, `Unknown role`, `Not provided`, and `No inbound message provided`; invalid timestamps remain visible as unknown rather than crashing the workflow. Leads without a usable email remain prioritizable but do not generate a simulated Calendly invitation.

## Run locally

```bash
cd prior
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r growth/requirements.txt
streamlit run growth/Inbound_Command_Center.py
```

To enable AI-generated call preparation, set an OpenAI API key before starting Streamlit:

```bash
export OPENAI_API_KEY="your_api_key_here"
# Optional: override the default call-preparation model.
export OPENAI_CALL_PREP_MODEL="gpt-6-astra"
```

Without an API key, the booked-meeting workflow remains fully usable and displays a clearly labelled fact-based brief.

## Suggested CEO demo flow (5 minutes)

1. Start on the unified queue and explain the objective:
   "Protect high-value prospects while removing low-value manual work."
2. Show that Northstar's SAP and deployment contacts resolve to one account, owner, and capacity slot.
3. Explain the three quality hypotheses and the bounded waiting-time component of the final score.
4. Open Atlas Insurance or Forge Industrial to show why product/open-source activity can reveal enterprise intent.
5. Open Independent to show an obvious self-serve case.
6. Show Independent or SmallLabs AI to demonstrate the low-quality + 14-day self-serve rule, then show team workload.
7. Open the salesperson workspace, choose a rep, and record a structured success or rejection outcome.
8. Return to the Outcomes tab to show how the decision feeds source conversion and calibration data.
9. Use "Add a lead" and type a new example live.
10. End with: "Every sales decision becomes training data for better rules later."

## Important prototype choices

- This is **not** a production ML model.
- AI is limited to optional call preparation in this implementation; routing remains auditable. The prototype does not send prospect data for AI processing unless an API key is deliberately configured. Unstructured intake extraction is a logical production extension, not a demonstrated feature here.
- The weights are hypotheses and should be calibrated with historical outcomes.
- Lead quality contributes up to 85 points and waiting time up to 15, allowing urgency to affect queue order without dominating commercial evidence.
- Whenever capacity opens, it is allocated to the highest-scoring eligible unassigned account. Existing ownership remains stable; completing all active work for an account releases capacity while retaining historical ownership.
- Capacity is counted by active account, not by individual contact.
- The salesperson workspace shows one row per account. Marking a bundled account Success or Rejected closes all of its contact records, releases exactly one capacity slot, and assigns that slot to the highest-scoring eligible unassigned account.
- Calendly delivery and bookings are simulated in-session for the prototype. Production would connect the same assignment event to an email provider and use Calendly webhooks to supply meeting times, cancellations, and conference URLs.
- Call preparation is generated only for scheduled or in-progress meetings, cached per lead input, and never treats inferred context as verified company information.
- Completed outcomes retain the original qualification and priority scores, allowing future analysis to compare the decision-time hypothesis with actual progression.
- Prototype account resolution uses business email domains and explicit IDs. `Independent` and `Not provided` records always remain individual leads and are never bundled. Production should use CRM account IDs plus reviewed identity-resolution rules.
- In production this should write decisions, tasks, and outcomes back to the existing CRM/SAP workflow rather than become a parallel system of record.
