# Applied AI Workflow Prototypes

[![CI](https://github.com/Aswath25S/prior/actions/workflows/ci.yml/badge.svg)](https://github.com/Aswath25S/prior/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)

This repository contains two independent, runnable Streamlit projects for exploring applied AI workflows around commercial operations and tabular data. Both projects are designed as transparent starting points: the sample data is fictional or synthetic, the important decisions are visible in code, and contributors can run useful local versions without API credentials.

> **Project status:** experimental. These applications are suitable for learning, prototyping, and extension—not as production systems or systems of record.

## Projects

| Project | What it explores | External services required? |
| --- | --- | --- |
| [Inbound Triage Copilot](growth/README.md) | Account resolution, explainable lead prioritization, capacity-aware assignment, outcome tracking, and optional AI call preparation | No. OpenAI call preparation is optional. |
| [TabPFN Use-Case Discovery Engine](tabpfn-usecase-discovery/README.md) | Local CSV profiling, use-case discovery, readiness checks, optional objective refinement, and measured TabPFN evaluations | No for discovery. OpenAI refinement and TabPFN evaluation are optional. |

The projects share a repository but not application state. You can install and run either one independently.

## Quick start

Prerequisites:

- Git
- Python 3.11 or newer
- A shell capable of activating a Python virtual environment

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/Aswath25S/prior.git
cd prior
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Run the inbound triage project:

```bash
python -m pip install -r growth/requirements.txt
streamlit run growth/Inbound_Command_Center.py
```

Or install and run the TabPFN discovery project instead:

```bash
python -m pip install -r tabpfn-usecase-discovery/requirements.txt
python tabpfn-usecase-discovery/scripts/generate_sample_data.py
streamlit run tabpfn-usecase-discovery/app.py
```

Streamlit prints the local URL after launch. API keys are not required for the core local workflows.

## Repository layout

```text
.
├── growth/                       # Inbound Triage Copilot
│   ├── Inbound_Command_Center.py # Main Streamlit entry point
│   ├── pages/                    # Salesperson workspace
│   ├── lead_engine.py            # Scoring and assignment rules
│   ├── call_prep.py              # Optional OpenAI call preparation
│   └── tests/
├── tabpfn-usecase-discovery/     # TabPFN Use-Case Discovery Engine
│   ├── app.py                    # Streamlit entry point
│   ├── src/                      # Discovery, readiness, and evaluation logic
│   ├── data/samples/             # Reproducible synthetic datasets
│   ├── scripts/                  # Dataset generation
│   └── tests/
├── docs/ROADMAP.md               # Suggested directions for contributors
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE
```

## Configuration and privacy

Each project includes an `.env.example`. Copy only the template for the project you are running:

```bash
cp growth/.env.example growth/.env
# or
cp tabpfn-usecase-discovery/.env.example tabpfn-usecase-discovery/.env
```

Real `.env` files are ignored by Git. Never commit API keys, customer records, proprietary datasets, or exported CRM data.

Data boundaries differ by feature:

- The triage workflow is local unless OpenAI call preparation is configured and invoked.
- Discovery and profiling are local in the TabPFN app.
- Optional OpenAI objective refinement sends business context and computed schema/statistics, not raw rows.
- Optional TabPFN evaluation sends the selected evaluation data only after explicit in-app consent.

Read each project guide before using non-public data.

## Tests

After installing the relevant project dependencies:

```bash
# Inbound Triage Copilot
python -m unittest discover -s growth/tests -t .

# TabPFN Use-Case Discovery Engine
python -m pytest -q tabpfn-usecase-discovery/tests
```

The GitHub Actions workflow runs both suites on pull requests and pushes to `main`.

## Contributing

Contributions are welcome. Start with [CONTRIBUTING.md](CONTRIBUTING.md), follow the [Code of Conduct](CODE_OF_CONDUCT.md), then review the [roadmap](docs/ROADMAP.md) and the README for the project you want to change. Issues that improve documentation, tests, accessibility, data handling, explainability, or modularity are especially useful.

## Security

Please do not report vulnerabilities or exposed credentials in a public issue. Follow [SECURITY.md](SECURITY.md).

## Independence and trademarks

This is an unofficial community repository. It is not endorsed by or affiliated with Prior Labs. TabPFN and Prior Labs names and trademarks belong to their respective owners. Links to official services or documentation are provided for interoperability and learning.

## License

Code and repository-authored documentation are available under the [MIT License](LICENSE). Bundled datasets are synthetic and are distributed with this repository under the same license.
