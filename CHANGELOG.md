# Changelog

All notable changes will be documented in this file. This project follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and intends to use
[Semantic Versioning](https://semver.org/) after the development preview.

## [Unreleased]

### Added

- Apache License 2.0 and public contribution and security policies.
- Configurable health-monitor startup grace.
- Safe automatic recovery for an unconfirmed health canary cancellation.

### Changed

- Periodic engine health monitoring now starts after bundle bootstrap completes.

## 0.1.0a1 - Development preview

- Typed choice, Boolean, score, and rank decisions with explicit probability semantics.
- Native SGLang and vLLM plugins, standalone gateways, and an experimental TokenSpeed adapter.
- Immutable bundle preparation, activation, rollback, request pinning, cancellation,
  recovery, admission control, health monitoring, managed LoRA, metrics, and SDKs.

No tagged release has been published yet.
