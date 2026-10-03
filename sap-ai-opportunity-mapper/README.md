# SAP Enterprise AI Opportunity Mapper

A working, independent Streamlit strategy prototype that turns fictional enterprise challenges into a provisional portfolio of structured prediction opportunities. It performs financial calculations, assesses evidence, applies hard gates, ranks opportunities, and creates Markdown executive briefs and POC proposals. No external APIs, SAP connections or TabPFN inference are used.

## Run locally

Use the existing `venv` inside `prior`. From the `prior` directory:

```bash
cd sap-ai-opportunity-mapper
../venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Open http://127.0.0.1:8501. Python 3.11+ is supported; the supplied environment was tested with Python 3.14. Dependencies already existed in that environment. If installing on another machine, use that environment's Python:

```bash
../venv/bin/python -m pip install -r requirements.txt
```

Run tests from the project directory:

```bash
../venv/bin/python -m pytest -q
```

## Workflow

1. Select Northstar Manufacturing, Meridian Retail Group, Harborview Financial or Custom enterprise. Save edited account details in Enterprise Profile.
2. In Business Landscape, enable challenges and edit severity, owner, implementation complexity, current-solution status, gap hypothesis, evidence and financial assumptions. All five functions are represented in the catalog.
3. Inspect Opportunity Portfolio for the impact/feasibility matrix, comparison and selected-candidate evidence. Use the selector below the chart to reveal details.
4. Adjust the four weights in Prioritization. Scores, eligibility and recommendations recompute from current session inputs.
5. Generate an illustrative phased roadmap and edit success-criteria placeholders in Adoption Roadmap. Download the selected POC proposal.
6. Download the full assessment from Executive Brief; select which opportunity's roadmap to include.

Account edits last for the current browser session. Switching accounts or choosing Reset reloads the fictional profile. Switching screens preserves the current account's saved edits. Profile fields use a Save button; challenge inputs update immediately.

## Five-minute interview demo

- **0:00–0:45:** Select fictional Northstar Manufacturing: $7bn revenue, S/4HANA and Ariba. Explain that products provide context, not proof of available data or integration.
- **0:45–1:30:** Open Business Landscape → Procurement → supplier delivery-delay risk. Review the defined decision, historical outcomes and point-in-time features. Existing demo Yes values are explicitly fictional assumptions.
- **1:30–2:15:** In Financial scenario, change cost per delayed order from $900 to $1,800. Open Opportunity Portfolio and inspect the updated exposure and scenarios. Baseline exposure rises from $12.96m to $25.92m; addressable ceiling doubles from $4.536m to $9.072m. Base benefit depends on adoption and assumed effectiveness, not measured accuracy.
- **2:15–3:00:** Return to the supplier challenge's Data & technical evidence tab. Change historical outcomes to No. The candidate becomes blocked and the recommendation selects another eligible opportunity or calls for discovery. High impact cannot override this gate.
- **3:00–3:45:** Restore the evidence assumption. Show priority weights and the comparison with excess-inventory and equipment-failure opportunities. Equipment has unknown outcome/access evidence and requires groundwork.
- **3:45–4:30:** Select supplier delay in Adoption Roadmap. Review conditional phase exits, incumbent baseline, alert burden, and success criteria to be agreed. Download the POC proposal.
- **4:30–5:00:** Download Executive Brief. Optionally select Meridian and show that demand forecasting remains gated until an incremental-value hypothesis against existing functionality is supplied.

## Decision method

Generation requires an enabled challenge. Industry determines the challenges offered for selection; user-enabled challenges remain available if the profile industry changes. No datasets or outcomes are inferred from installed SAP products. Catalog definitions supply suggested targets and prediction timing; users separately confirm whether this framing and historical evidence exist.

Default weights are impact 35%, data/technical feasibility 30%, adoption readiness 20%, incremental value 15%. Nonnegative weights normalize to 100%; all-zero inputs fall back to defaults. Weights and thresholds are illustrative, not empirically validated.

- **Impact:** 55% annual base benefit score, 30% severity, 15% direct priority alignment. Benefit score is capped logarithmic scaling ($10k → 0, $10m → 100). Unknown benefit contributes zero only to its component and is still shown as unknown. Company revenue is never an input to scoring or model revenue.
- **Feasibility:** average of the fraction of affirmative data-readiness evidence and affirmative technical-framing evidence. Unknown and No both contribute zero; they have distinct UI labels and reports.
- **Readiness:** average of organizational evidence (including named owner) and inverse complexity, from 1 (simple) to 5 (complex).
- **Incremental value:** 70 for a stated improvement hypothesis, 40 for no identified incumbent, zero otherwise. A hypothesis is not evidence of superiority. Text must be reviewed by stakeholders.
- **Hard gates:** missing outcome definition/labels, history, decision-time features, timing, horizon, structured framing, access, business action or business owner block technical evaluation. Existing/satisfactory/unknown solutions without a gap hypothesis trigger benchmark justification. Other unknown evidence yields Needs validation, never a planning-ready classification.

Ranking sorts first by gates, then planning readiness, then provisional weighted score. The executive recommendation chooses the highest-ranked planning candidate or explicitly says none is ready. Matrix splits at 60/100 guide discussion; they never override gates. All positions are provisional, and gated opportunities use open diamonds.

## Financial calculations

Event losses: annual volume × adverse-event rate × attributable unit cost. For stockouts, unit cost is lost **contribution margin**, not revenue. Excess-inventory inputs represent annual carrying/obsolescence cost, not inventory asset value. Cash-forecast scenarios must represent attributable operational or financing costs, not the cash balance itself.

Payment losses: invoices × late-payment rate × average invoice balance × days late / 365 × annual cost of capital. The addressable ceiling also scales by `min(days reduced / days late, 1)`. Total outstanding balances are never savings. If overriding payment exposure, enter annual carrying costs; the override already incorporates duration and capital assumptions.

Ceiling = exposure × addressable share. Base benefit = ceiling × adoption × intervention effectiveness. Conservative = half of base. Optimistic multiplies adoption by 1.2 and effectiveness by 1.3, each capped at 100%. Effectiveness is an assumption, not measured TabPFN performance. Sensitivity shows ±20% in attributable cost/exposure, holding other inputs fixed.

Annual operating costs and separate one-time implementation costs are optional. Net annual benefit is shown only with operating costs; payback is shown only when both costs are supplied and net benefit is positive. Missing required inputs produce “Not enough information to estimate”; zero remains a valid value.

## Architecture

`src/models.py` holds dataclasses; `data/` holds three fictional accounts and 12 structured catalog entries. Modules separate profile loading, matching, finances, feasibility, prioritization, landscape views, roadmaps and Markdown generation. `app.py` contains the six-screen interface. No final recommendation is tied to a company name.

Tests cover profile isolation, opportunity matching, labels and decision gates, unknowns, financial arithmetic and sensitivity, costs/payback, priority/weight/complexity changes, all-account report generation, all screens including custom accounts, and live UI edits. Actual verification results are recorded in `VALIDATION.md`.

## Known limitations

- All readiness evidence is asserted by the user or seeded fictionally; there is no schema inspection, benchmarking or validation of a typed gap hypothesis.
- Priority alignment uses transparent text matching. Scenario factors and score scales are discussion aids, not investment-grade estimates or causal effects.
- The roadmap is conditional. A missing label history may take longer than 90 days to establish. Month-length days are approximated by twelve-week phases.
- Session-only edits; no saved-account store, authentication, integrations, production ingestion or actual training/inference.
- Products listed in catalog entries may be absent from the account. Such entries remain hypotheses and require source/access confirmation; churn data in particular may live outside ERP.
- The prototype makes no acquisition, availability, endorsement, measured-accuracy or superiority claims. Current product facts should be verified separately before external commercial use.

## Three highest-value future improvements

1. Add an evidence register with source, owner, verification date and schema/label checks; distinguish assertions from verified readiness.
2. Add an authorized offline benchmark workflow with point-in-time splits, incumbent comparisons, calibration and intervention-budget evaluation.
3. Add portfolio economics with measured intervention capacity/costs, uncertainty ranges and versioned scenario snapshots for stakeholder decisions.
