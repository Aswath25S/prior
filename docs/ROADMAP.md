# Project Roadmap

This roadmap is a menu of useful contribution areas, not a commitment to dates or a fixed product direction. Open an issue before starting a large item so scope and interfaces can be agreed first.

## Repository-wide

- Add formatting and linting with a minimal, documented configuration.
- Add end-to-end browser checks for the critical Streamlit paths.
- Add dependency and secret scanning in CI.
- Improve keyboard navigation, contrast, and screen-reader labels.
- Add lightweight release notes and semantic versioning when the first reusable release is ready.

## Inbound Triage Copilot

### Good first contributions

- Move score thresholds and weights into a validated configuration file.
- Add boundary tests for every title/source classifier.
- Add CSV schema validation with actionable error messages.
- Add a reset-demo-data control that clearly explains session behavior.
- Add downloadable score explanations for debugging and review.

### Larger directions

- Define repository interfaces for leads, accounts, people, assignments, and outcomes.
- Add a local SQLite implementation before attempting a hosted database.
- Introduce CRM, calendar, and email adapters behind testable interfaces.
- Add identity-resolution confidence and a manual merge/unmerge workflow.
- Calibrate scoring against historical outcomes and monitor drift and fairness.
- Add authentication, role-based access, audit history, and retention controls.

## TabPFN Use-Case Discovery Engine

### Good first contributions

- Add small synthetic examples for additional industries and task types.
- Expand validation messages for unsupported CSV encodings and schemas.
- Add tests for ambiguous objectives and unusual target types.
- Add downloadable discovery/readiness reports that contain no raw data by default.
- Improve result explanations for non-specialist users.

### Larger directions

- Extract a stable, UI-independent discovery API.
- Add pluggable profilers and use-case catalogs.
- Add repeatable evaluation manifests and cached local artifacts.
- Support additional leakage-safe validation strategies and uncertainty summaries.
- Add privacy-preserving dataset summaries and organization-level policy controls.
- Compare API-backed evaluation with fully local TabPFN execution where appropriate.

## Out of scope without design review

- Sending external email or calendar invitations.
- Persisting uploaded customer data to a hosted service.
- Automatic purchasing, CRM mutation, or other irreversible actions.
- Claims of production readiness, ROI, regulatory compliance, or causal impact based on prototype output.
