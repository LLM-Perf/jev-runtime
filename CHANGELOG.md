# Changelog

All notable changes will be documented in this file. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and intends to use
[Semantic Versioning](https://semver.org/) after the development preview.

## [Unreleased]

### Added

- TokenSpeed source/hardware/configuration preflight, a pinned Qwen3-0.6B setup
  helper and expanded installation/operations documentation.
- Durable TokenSpeed completion receipts for response loss, cache eviction and
  confirmation of completed requests after a same-identity restart. Unknown and
  pending requests remain unconfirmed.
- TokenSpeed accounting in the shared four-type, strict-request and hot-switch
  validation harness. Real TokenSpeed GPU validation remains pending.
- Apache License 2.0 and public contribution and security policies.
- Configurable health-monitor startup grace.
- Safe automatic recovery for an unconfirmed health canary cancellation.

### Changed

- TokenSpeed source profile now accepts upstream
  `f4ac1affe11ad404720bcd150970487f75fbf59a`, retaining the legacy
  `7fa8acb1e885389825c077a6aec0326fbbbd7116` profile.
- TokenSpeed requests submit asynchronously to the Engine's owner loop rather
  than reserving one waiting executor thread per request.
- Periodic engine health monitoring now starts after bundle bootstrap completes.

### Fixed

- TokenSpeed local model and separate tokenizer paths must match their respective
  Jev configuration fields. Generated configs preserve that binding.
- The TokenSpeed launcher shuts down its owned Engine even after Runtime startup
  or drain fails, retaining unresolved evidence and the error.

TokenSpeed implements the complete typed-decision API and bundle lifecycle.
Its current scoring path uses K native requests for K labels.
The [validation report](docs/tokenspeed-validation.md) records 618 passing local
tests and 12 CPU source-ordering checks at `2d3ee4e`; GPU serving, joint readout
and performance certification remain open.

## 0.1.0a1 - Development preview

- Typed choice, Boolean, score, and rank decisions with explicit probability semantics.
- Native SGLang and vLLM plugins, standalone gateways, and a TokenSpeed adapter.
- Immutable bundle preparation, activation, rollback, request pinning, cancellation,
  recovery, admission control, health monitoring, managed LoRA, metrics, and SDKs.

No tagged release has been published yet.
