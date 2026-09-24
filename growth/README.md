# Prior Labs — Inbound Triage Copilot

A lightweight working prototype for the inbound-interest case.

## What it demonstrates

1. A unified inbound queue across SAP, website, API, GitHub, and other sources.
2. Transparent prioritization on three dimensions:
   - Commercial potential (0–40)
   - Buying intent (0–35)
   - Use-case maturity (0–25)
3. One final priority score combining lead quality (85 points) and waiting time (15 points).
4. A score-sorted priority queue with a visible status and sales owner for every lead.
5. An explanation panel showing *why* each lead received its score.
6. Automatic allocation of limited sales capacity from the top of the ranked queue; overflow remains queued and unassigned until capacity opens.
7. An illustrative commercial-team view showing workload and capacity.
8. Editable progress controls for the selected lead.
9. A form to simulate a new inbound lead live during the demo.
10. A self-serve rule for leads with quality below 30 that have waited at least 14 days.
11. Three clear lifecycle statuses: In Progress, Success, and Rejection.
12. A separate salesperson workspace for reviewing an individual portfolio and updating outcomes.
13. Automatic demo Calendly emails on assignment, with event-driven meeting status and a future-meeting join button in the salesperson workspace.

The names in `sales_team.csv` are fictional and exist only to demonstrate the operating workflow. All salespeople are interchangeable in this first version; initial calls are assigned to whoever has the most capacity left.

## Run locally

```bash
cd growth
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run Inbound_Command_Center.py
```

## Suggested CEO demo flow (5 minutes)

1. Start on the unified queue and explain the objective:
   "Protect high-value prospects while removing low-value manual work."
2. Show that the queue is sorted by one score and that sales capacity is assigned from the top down.
3. Open L001 (SAP bank) and show its score breakdown plus its automatically assigned owner.
4. Open L003 or L004 to show why product/open-source activity can still reveal an enterprise opportunity.
5. Open L008 to show an obvious self-serve case.
6. Show L008 or L010 to demonstrate the low-quality + 14-day self-serve rule, then show team workload.
7. Open the salesperson workspace, choose a rep, and update one lead to Success or Rejection.
8. Use "Add a lead" and type a new example live.
9. End with: "Every sales decision becomes training data for better rules later."

## Important prototype choices

- This is **not** a production ML model.
- AI should initially be used to turn unstructured text into structured fields; scoring remains auditable.
- The weights are hypotheses and should be calibrated with historical outcomes.
- One final priority score combines lead quality (85 points) and waiting-time urgency (15 points).
- Sales capacity is allocated to eligible leads in descending score order. Low-quality leads move to self-serve only after 14 days.
- In Progress leads consume active capacity. Success and Rejection are completed outcomes that release capacity while retaining the salesperson who handled the lead.
- Calendly delivery and bookings are simulated in-session for the prototype. Production would connect the same assignment event to an email provider and use Calendly webhooks to supply meeting times, cancellations, and conference URLs.
- In production, individual events should be resolved to a company so multiple users from that company become one stronger commercial signal.
