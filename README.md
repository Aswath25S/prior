# Prior: small applied AI prototypes

This repository contains two independent Streamlit prototypes exploring practical workflows around AI and data. They were built as learning and interview projects. They are not production systems, and their sample records are fictional or synthetic.

## Projects

### Inbound Triage Copilot (`growth/`)

An illustrative commercial workflow for organizing inbound leads, explaining account priorities, assigning limited sales capacity, and recording outcomes. It runs locally with fictional sample leads and needs no API key for its core features.

[Project guide](growth/README.md)

### TabPFN Use-Case Discovery Engine (`tabpfn-usecase-discovery/`)

A local data exploration app that profiles tabular data, suggests possible prediction tasks, and can run optional evaluations. Its bundled example datasets are synthetic. Optional objective refinement uses OpenAI; optional TabPFN evaluation sends selected data to the TabPFN cloud service after in-app consent.

[Project guide](tabpfn-usecase-discovery/README.md)

## Getting started

Use Python 3.11 or newer. Each project has its own dependency list; install only the one for the app you want to run.

```bash
git clone <repository-url>
cd prior
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install -r growth/requirements.txt
streamlit run growth/Inbound_Command_Center.py
```

For the data discovery app, install `tabpfn-usecase-discovery/requirements.txt` instead and launch `streamlit run tabpfn-usecase-discovery/app.py` from the repository root. See that project's guide for optional API setup and data handling details.

## Data and privacy

Do not commit private data, API keys, or `.env` files. The committed demo datasets are fictional/synthetic. The triage app can send lead context to OpenAI for call preparation only when configured with an API key. The discovery app processes uploaded data locally for discovery and profiling; optional OpenAI refinement sends descriptions and computed statistics, while optional TabPFN benchmarking can send selected feature and target data to the cloud after explicit consent. Review each project's README before using non-public data.

## Repository map

```text
growth/                       Inbound Triage Copilot
tabpfn-usecase-discovery/     TabPFN Use-Case Discovery Engine
```

## License

No license has been added yet. Until one is chosen, the default copyright rules apply; please don't assume the code is available for reuse. Add a license file before representing this repository as open source.
