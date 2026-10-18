# Security policy

## Supported versions

Jev Runtime is currently a development preview. Security fixes are applied to the
latest commit on `main`; no older release line is supported yet.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private
vulnerability reporting flow:

<https://github.com/LLM-Perf/jev-runtime/security/advisories/new>

Include the affected version or commit, deployment mode, engine, reproduction
steps, impact, and any suggested mitigation. Do not include production credentials,
model artifacts, customer inputs, or other sensitive data.

We aim to acknowledge a complete report within five business days. Timelines for
validation, remediation, and disclosure depend on severity and affected upstream
engines. Please allow a coordinated fix before public disclosure.

## Security boundaries

Jev Runtime authenticates its own decision and management APIs, but operators are
responsible for network isolation, secret distribution, TLS termination, engine
endpoint security, model provenance, and host hardening. A probability distribution
is not a trust or safety decision by itself.
