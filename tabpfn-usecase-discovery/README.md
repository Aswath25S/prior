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

```bash
pytest -q
```

## Architecture

- `app.py`: Streamlit interaction and visualization only.
- `src/data_loader.py`: bounded CSV parsing and user-facing validation errors.
- `src/data_profiler.py`: type, missingness, duplicate, identifier, cardinality, and target-candidate detection.
- `src/use_case_discovery.py`: explicit, extendable industry/use-case catalog and evidence-based matching.
- `src/readiness.py`: target, feature-quality, leakage, temporal, and unanswered deployment checks.
- `src/prioritization.py`: explainable prioritization using user-provided importance and observed feasibility.
- `src/benchmark.py`: conservative feature defaults, leakage-safe splitting, preprocessing, baseline models, TabPFN inference, and measured holdout metrics.
- `src/poc_generator.py`: dataset-specific Markdown proposal generation.
- `scripts/generate_sample_data.py`: reproducible synthetic manufacturing, retail, and absent-target examples.

## Heuristic boundaries

Column names and lightweight statistics cannot establish business meaning, feature availability, causality, or regulatory acceptability. Every discovered opportunity is a candidate requiring confirmation. Leakage matches are warnings, not assertions. Priority and operational importance are explicitly user-provided. Benchmark metrics come from a single holdout and do not establish production performance or business impact.
