# TabPFN Use-Case Discovery Engine

An unofficial prototype that turns a tabular dataset and business context into transparent use-case candidates, readiness checks, prioritization, an optional real TabPFN benchmark, and an exportable POC proposal. Discovery and profiling run locally. TabPFN cloud inference runs only after explicit user confirmation. The app does not calculate ROI or claim production readiness.

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

You can also enter the key in the app's password field or set it before launch:

```bash
export TABPFN_TOKEN="your-token"
streamlit run app.py
```

The benchmark sends the selected training and holdout feature data to the TabPFN cloud service only after the user checks the explicit consent box. Do not use data you are not permitted to share.

## Test

### Optional OpenAI objective refinement

Add `OPENAI_API_KEY="your-key"` and optionally `OPENAI_MODEL="gpt-4.1-mini"` to the project's ignored `.env`, then restart Streamlit. Use **Refine objective with OpenAI**, review/edit the proposed brief, and click **Use this objective and capability**.

Refinement uses the Responses API with a strict typed response and `store=False`. It sends the four context inputs and computed schema/statistics; it does not send raw rows. User-entered descriptions and column names are included, so avoid inserting secrets into them. Errors leave local discovery available. Calls occur only on the refinement button, not on every rerun.

The confirmed primary capability directly routes assessment, evaluation, and the POC. Classification/regression/forecasting use the existing prediction benchmark. Anomaly detection, imputation, interpretability, and synthetic generation route to their extension-specific POC flows (execution not yet integrated). Ambiguous or unsupported goals can route to clarification. Invalid column references are rejected locally; the API's valid format does not guarantee a semantically correct recommendation. Users can correct the brief before applying it.

The proposed feature list is intersected with conservative benchmark defaults; targets and suspected leakage fields are excluded by default. Changing source inputs invalidates the brief and all dependent results. The original objective remains in the proposal alongside the confirmed brief, deliverable, decision, criteria, and questions.

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
- `src/poc_generator.py`: dataset-specific Markdown proposal generation.
- `scripts/generate_sample_data.py`: reproducible synthetic manufacturing, retail, and absent-target examples.

## Heuristic boundaries

Step 2 collects industry, a business objective, information about the data, and an optional target. Discovery uses explicit rules to match objectives and data descriptions to prediction, outlier detection, missing-data recovery, predictive factor analysis, or synthetic generation. Prioritization considers objective alignment, data feasibility, and implementation complexity; no business-value ratings are inferred.

Classification/regression benchmarks can run through the installed TabPFN client. Other capabilities have tailored readiness checks, evaluation designs, and POC proposals, but their experimental extension runners are not integrated. See [official TabPFN extensions](https://github.com/PriorLabs/tabpfn-extensions). An API key alone does not enable these runners.

Column names and lightweight statistics cannot establish business meaning, feature availability, causality, or regulatory acceptability. Every discovered opportunity is a candidate requiring confirmation. Leakage matches are warnings, not assertions. Benchmark metrics come from a single holdout and do not establish production performance or business impact. Changes to the dataset or business context clear prior proposals and benchmark results.
