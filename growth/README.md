# Prior Labs — Inbound Triage Copilot

A lightweight working prototype for the inbound-interest case.

## What it demonstrates

1. A unified inbound queue across SAP, website, API, GitHub, and other sources.
2. Transparent prioritization on three dimensions:
   - Commercial potential (0–40)
   - Buying intent (0–35)
   - Use-case maturity (0–25)
3. Hard rules for obvious high-value and obvious low-value cases.
4. Four routing outcomes:
   - P1 — Engage now
   - P2 — Human review
   - P3 — Nurture / monitor
   - P4 — Self-serve
5. An explanation panel showing *why* each lead was routed.
6. A score-sorted priority queue with a visible status and sales owner for every lead.
7. An illustrative commercial-team view showing workload and capacity.
8. Editable ownership and progress controls for the selected lead.
9. A form to simulate a new inbound lead live during the demo.
10. Waiting-time prioritization with inbound timestamps, response targets, and overdue flags.

The names in `sales_team.csv` are fictional and exist only to demonstrate the operating workflow. All salespeople are interchangeable in this first version; initial calls are assigned to whoever has the most capacity left.

## Run locally

```bash
cd growth
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Suggested CEO demo flow (5 minutes)

1. Start on the unified queue and explain the objective:
   "Protect high-value prospects while removing low-value manual work."
2. Show that the queue is sorted by score and every lead has a clear owner and status.
3. Open L001 (SAP bank) and show the hard-rule escalation plus its assigned owner.
4. Open L003 or L004 to show why product/open-source activity can still reveal an enterprise opportunity.
5. Open L008 to show an obvious self-serve case.
6. Move a lead to a new owner or status, then show the team workload view.
7. Use "Add a lead" and type a new example live.
8. End with: "Every sales decision becomes training data for better rules later."

## Important prototype choices

- This is **not** a production ML model.
- AI should initially be used to turn unstructured text into structured fields; routing remains auditable.
- The weights are hypotheses and should be calibrated with historical outcomes.
- Commercial attractiveness and waiting time remain separate: the opportunity score measures lead quality, while a bounded time boost determines operational queue order.
- In production, individual events should be resolved to an account so multiple users from one company become one stronger commercial signal.
