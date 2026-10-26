# Contributing

Thanks for helping improve these prototypes. Contributions should keep the projects understandable, locally runnable, and honest about what has and has not been validated.

Participation in this repository is governed by the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before you start

1. Read the root [README](README.md) and the README for the project you want to change.
2. Search existing issues and pull requests before opening a new one.
3. For a substantial behavioral change, open an issue describing the problem, proposed scope, and data/privacy implications before investing in implementation.

Small documentation corrections, focused tests, and clearly bounded bug fixes can go directly to a pull request.

## Local setup

```bash
git clone https://github.com/Aswath25S/prior.git
cd prior
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the dependencies for the project you are changing:

```bash
python -m pip install -r growth/requirements.txt
# or
python -m pip install -r tabpfn-usecase-discovery/requirements.txt
```

Never commit a populated `.env` file. Both applications work in a reduced local mode without credentials.

## Development principles

- Keep business rules deterministic and inspectable unless a feature is explicitly described as AI-assisted.
- Keep Streamlit pages focused on interaction and presentation; put reusable logic in engine or `src/` modules.
- Prefer small functions, explicit names, and typed or validated boundaries for external-service responses.
- Preserve useful behavior when optional APIs are unavailable.
- Do not add real customer, lead, payment, health, or proprietary business data.
- Label synthetic data, simulations, heuristics, and unvalidated assumptions clearly.
- Add or update tests for behavioral changes.
- Update the relevant README when setup, configuration, privacy boundaries, or user behavior changes.

## Running tests

```bash
# Inbound Triage Copilot
python -m unittest discover -s growth/tests -t .

# TabPFN Use-Case Discovery Engine
python -m pytest -q tabpfn-usecase-discovery/tests
```

Tests must not require live OpenAI or TabPFN credentials. Use mocks or local deterministic fallbacks for external-service behavior.

## Adding sample data

Committed sample data must be demonstrably synthetic or fictional and safe to distribute under the repository license.

When adding or changing a dataset:

- Include a reproducible generator where practical.
- Document the intended task, target, time semantics, and known limitations.
- Avoid realistic personal identifiers or values copied from real organizations.
- Add a validation test for schema, row count, and reproducibility.

## Pull requests

Keep each pull request focused. Include:

- The problem and why it matters.
- A concise description of the solution.
- Commands used to verify the change.
- Screenshots for visible UI changes.
- Any new data sent to external services.
- Known limitations or follow-up work.

By contributing, you agree that your contribution is licensed under the repository's [MIT License](LICENSE).
