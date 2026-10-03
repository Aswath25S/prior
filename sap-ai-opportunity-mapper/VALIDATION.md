# Verification record

Verified October 2, 2026 (America/New_York) with the existing `prior/venv` environment, Python 3.14 and Streamlit 1.64.0.

## Automated checks

Executed from `prior`:

```text
venv/bin/python -m pytest sap-ai-opportunity-mapper/tests -q
...................                                                      [100%]
19 passed in 1.20s
```

Coverage includes actual financial arithmetic, payment carrying costs, zero versus missing values, optional cost/payback behavior, profile isolation, matching, outcome/timing/access/action gates, incumbent justification, unknown sample sufficiency, sensitivity, complexity, normalized weights, recommendation changes, reports and conditional roadmaps.

Streamlit AppTest executed all six screens for Northstar, Meridian, Harborview and Custom enterprise without application exceptions. It changed supplier unit cost through the UI and verified recalculated exposure; it then set historical outcomes to No and verified hard gates.

## Scenario results with untouched fictional inputs

- Northstar: supplier delivery-delay risk is the initial planning candidate. Equipment failure remains gated because historical outcome and access assumptions are unknown.
- Meridian: stockout risk is the initial planning candidate. Demand prediction remains gated until an incremental-value hypothesis is supplied against the existing capability.
- Harborview: cash-collection forecasting is the initial planning candidate under the illustrative inputs. Churn and operational incident risk remain gated. This is a scoring result, not a claim of customer value.

## Local launch

Started using `../venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501` from the project folder. The server reported its localhost URL. An HTTP HEAD request returned `200 OK` with HTML content.

The sandbox initially blocked localhost binding/network access; the server and HTTP check succeeded with approved execution outside the sandbox.

## Visual verification limit

The Browser skill was initialized, but browser discovery returned no available browser. No screenshot or human visual inspection was performed. UI verification consists of actual Streamlit screen renders and interaction assertions in AppTest, plus the running server HTTP check. No model training, accuracy validation, real data access or realized-benefit validation was performed.
