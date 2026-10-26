# Security Policy

## Supported code

This repository contains experimental prototypes rather than a supported production release. Security fixes are applied to the latest code on the `main` branch.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability, exposed credential, or sensitive-data incident.

Use [GitHub private vulnerability reporting](https://github.com/Aswath25S/prior/security/advisories/new) when available. Include:

- The affected project and file.
- Reproduction steps or a minimal proof of concept.
- Expected impact.
- Any suggested mitigation.

If private reporting is unavailable, contact the repository owner through their GitHub profile without including exploit details in a public message.

## Secrets and data

- Never commit `.env` files, API tokens, credentials, or service-account material.
- Revoke and rotate a credential immediately if it is exposed, even if the commit is later removed.
- Treat uploaded datasets and lead records as sensitive unless they are explicitly synthetic or public.
- Review the data boundary in each project README before enabling OpenAI or TabPFN API features.

## Production use

The applications do not currently provide authentication, authorization, tenant isolation, durable audit logging, secret management, encrypted persistence, rate limiting, or formal compliance controls. A production deployment requires a separate threat model and security review.
