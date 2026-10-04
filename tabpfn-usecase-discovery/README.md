# TabPFN Use-Case Discovery Engine

An unofficial prototype that turns a tabular dataset and business context into a confirmed TabPFN use case, automatic data checks, and an optional measured evaluation. It then directs interested users to official Prior Labs documentation, API access, examples, community support, and contact options. Discovery and profiling run locally. TabPFN cloud inference runs only after explicit user confirmation. The app does not calculate ROI or claim production readiness.

## Run locally

Python 3.11+ is supported (verified here with Python 3.14).

```bash
cd tabpfn-usecase-discovery
source ../venv/bin/activate
pip install -r requirements.txt
python scripts/generate_sample_data.py
streamlit run app.py
```

The app opens at `http://localhost:8501`. Uploaded data stays in the local Streamlit process and is not persisted by application code.

This repository uses the shared `prior/venv` environment. Do not create a second virtual environment inside this project.

To run the optional TabPFN benchmark, create an API key at `platform.priorlabs.ai/account/api-keys`. Save it in the project's `.env` file:

```dotenv
TABPFN_TOKEN="your-token"
```

Restart Streamlit after editing `.env`. The app loads this file automatically, regardless of the launch directory. The file is ignored by Git; `.env.example` is a blank template. An existing environment variable takes precedence over `.env`.

The prototype uses the configured key for all evaluations; there is no user-facing token input. You can also set the key before launch:

```bash
export TABPFN_TOKEN="your-token"
streamlit run app.py
```

The benchmark sends the selected training and holdout feature data to the TabPFN cloud service only after the user checks the explicit consent box. Do not use data you are not permitted to share.

## Test

### Optional OpenAI objective refinement

Add `OPENAI_API_KEY="your-key"` and optionally `OPENAI_MODEL="gpt-4.1-mini"` to the project's ignored `.env`, then restart Streamlit. Use **Refine objective with OpenAI**, review/edit the proposed brief, and click **Use this objective and capability**.

Refinement uses the Responses API with a strict typed response and `store=False`. It sends the four context inputs and computed schema/statistics; it does not send raw rows. User-entered descriptions and column names are included, so avoid inserting secrets into them. Errors leave local discovery available. Calls occur only on the refinement button, not on every rerun.

The confirmed primary capability directly routes the automatic assessment, evaluation, and recommended official documentation. The five steps are dataset selection, objective refinement, recommendation, evaluation, and possible next steps. There is no separate expert questionnaire or data-review confirmation. All supported categories have a Step 4 evaluation with measured results. Ambiguous or unsupported goals can route to clarification. Invalid column references are rejected locally; the API's valid format does not guarantee a semantically correct recommendation. Users can correct the brief before applying it.

The proposed feature list is intersected with conservative evaluation defaults; targets and suspected leakage fields are excluded by default. Changing source inputs invalidates the brief and all dependent results.

### Run tests

```bash
pytest -q
```

## Architecture

- `app.py`: Streamlit interaction and visualization only.
- `src/data_loader.py`: bounded CSV parsing and user-facing validation errors.
- `src/data_profiler.py`: type, missingness, duplicate, identifier, cardinality, and target-candidate detection.
- `src/use_case_discovery.py`: explicit, extendable industry/use-case catalog and evidence-based matching.
- `src/objective_refinement.py`: structured OpenAI refinement, local validation, and deterministic capability routing.
- `src/readiness.py`: target, feature-quality, leakage, temporal, and unanswered deployment checks.
- `src/prioritization.py`: explainable prioritization using user-provided importance and observed feasibility.
- `src/benchmark.py`: conservative feature defaults, leakage-safe splitting, preprocessing, baseline models, TabPFN inference, and measured holdout metrics.
- `src/analysis.py`: bounded conditional TabPFN workflows for imputation, anomaly ranking, and synthetic sampling.
- `src/result_views.py`: model comparison tables, individual SHAP waterfalls, and downloads.
- `src/shap_explanations.py`: exact or permutation-estimated empirical interventional Shapley values for the API model.
- `scripts/generate_sample_data.py`: reproducible synthetic equipment-failure and retail-demand examples.

## Built-in example datasets

Each example shows a data description and the objective behind its generation. Enter the objective you want to explore in Step 2, then refine it to choose the analysis.

| Example | Intended exploration |
| --- | --- |
| Manufacturing — equipment failure | Classify 30-day failure risk to prioritize maintenance |
| Retail — product demand | Forecast dated store/product demand for inventory planning |

The manufacturing example contains 1,500 machine observations; retail contains 4,800 daily store/product records. Both are generated locally with fixed random seeds; the generation process does not call OpenAI or TabPFN.

## Heuristic boundaries

Step 2 collects industry, a business objective, information about the data, and an optional target. Discovery uses explicit rules to match objectives and data descriptions to prediction, outlier detection, missing-data recovery, predictive factor analysis, or synthetic generation. Prioritization considers objective alignment, data feasibility, and implementation complexity; no business-value ratings are inferred.

All evaluations use the installed TabPFN API client; no GPU or second virtual environment is needed. The initial evaluation methods are explicit:

- Classification: one table compares traditional models and TabPFN across accuracy, balanced accuracy, precision, recall, F1, and applicable AUC scores. A TabPFN confusion matrix shows actual versus predicted outcome counts on unseen test records for binary and multiclass tasks. The equipment-failure example labels missed failures and false alarms.
- Regression: one comparison table shows MAE, RMSE, and R². Model prediction ranges remain available in the measured artifacts.
- Forecasting: whole-date chronological holdout with the same numeric comparison table. This tests historical supplied inputs; it does not implement multi-step future forecasting or generate future covariates.
- Interpretation: retains one “Inspect an individual result” SHAP waterfall and expandable guidance. Empirical interventional SHAP uses eight sampled learning records as the reference background and explains up to five unseen records with at most 12 fields. Up to six fields use the exact Shapley formula for that finite background; larger feature sets use 16 sampled permutations. Contributions add to the difference between the reference output and the record's output, including nonlinear interactions. Binary analyses explain the less common training outcome's probability; multiclass analyses explain each record's predicted class probability; numeric analyses explain the estimated value. These require additional API requests. Results depend on the sampled background, and mixed-field combinations may be unrealistic for correlated fields. SHAP values describe model behavior, not causal effects. The backend can also compute global permutation importance, but the simplified UI skips these extra calls.
- Missing-value recovery: holds out 20% of rows, masks known cells one field at a time, compares recovery with training-only median/mode, and fills real gaps in held-out records. Downloads preserve original missing flags. Artificial masking does not establish accuracy for actual missing cells.
- Unusual-record ranking: predicts each field from the others and aggregates numeric scaled residual/category probability surprise percentiles. Shows a comparison table with Isolation Forest and one record-level field surprise chart. Alert-list agreement is not accuracy; precision/recall require review labels.
- Synthetic generation: samples the first field empirically, then uses conditional TabPFN category probabilities or numeric quantiles for subsequent fields. Compares marginal distributions, pairwise encoded relationships, and exact reference matches against untouched holdout records and a resampling baseline. Numeric sampling uses the middle 90% of quantiles; missingness is not synthesized, and extreme values may be underrepresented. No privacy guarantee is made.

Targetless evaluations use at most 1,000 distinct records and 2–8 fields, with identifiers, constants, and date columns omitted. They are conditional client workflows, not the full experimental joint-density methods in [official TabPFN extensions](https://github.com/PriorLabs/tabpfn-extensions). Changing evaluation settings hides mismatched old results.

Result pages show comparison tables and an individual result view, plus a confusion matrix for classification. Missing-value recovery retains one individual recovery view; generation shows per-field fidelity comparisons. Detailed method notes are under “How to interpret this evaluation.”

Column names and lightweight statistics cannot establish business meaning, feature availability, causality, or regulatory acceptability. Every discovered opportunity is a candidate requiring confirmation. Leakage matches are warnings, not assertions. Evaluation metrics come from a single holdout and do not establish production performance or business impact. Changes to the dataset or business context clear prior results.

Step 5 does not generate a proposal. It links to the official documentation overview, quickstart, recommended capability guide, API-key management, benchmarking guide, client and extension repositories, TabPFN Discord, and the Prior Labs contact page.
