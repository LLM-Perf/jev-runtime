# DSW gateway upgrade and rollback through a stable proxy

The controller passed real two-worker-per-slot replacement and installed-wheel
upgrade/rollback on both native engines. All recorded continuous traffic requests
succeeded. This is a short, colocated functional qualification, not an availability
SLA, throughput result, engine failover certificate or complete release acceptance.

## Exact artifacts

- Previous source: `accef862f11bcc8ebb666225f328f8f088270713`.
- Candidate/controller/harness: `773914587b4fab287ca1e4b17f60ff66eb20d261`.
- Both snapshots produced three verified Python wheels. Two independently installed,
  offline, hash-locked gateway environments contain 44 distributions each, including
  Transformers 5.17.0/tokenizers 0.23.2. The imported directory is
  `site-packages/jev_runtime`; all 30 Python files per slot match the declared commit.
  Python `-I` prevents a checkout/PYTHONPATH from shadowing installed packages.
- Both packages retain development version `0.1.0a1`; source and manifest digests
  distinguish them. Wheel manifests are
  `79d14c67db9a1235ed2da1a34c1f0707202e4221b2f637535b9b36b1a4de225b` (previous) and
  `2a097d0e01bd6aae3ea5db98bbdd47eb11acac31027965b60ef92ee30f2c985a` (candidate).
- HAProxy 3.2.24, one event loop, private loopback listener and Runtime API socket.
  The audit retains source and binary digests. Proxy and native process identities
  remain unchanged throughout each installed-wheel run. The current controller
  includes Linux start ticks and boot ID when identifying the proxy.
- SmolLM2-1.7B-Instruct revision `31b70e2e869a7173562077fd711b654946d38674`,
  BF16 model/readout, TP1, eager, context 2048, GPU7 L20Z. Both serialized tokenizer
  profiles have implementation digest
  `sha256:08e96741a4f1c7a56136f9400ddf6d8f4f0e50fd09743f83579328e76e6cb6ef`.
- Native vLLM 0.30.0+cu129 uses Transformers 5.17.0/tokenizers 0.23.2; native
  SGLang 0.5.19 uses Transformers 5.12.1/tokenizers 0.22.2. Both use Torch
  2.13.0+cu129 and remain separate from the gateway environments. HTTP adapter
  capabilities retain `verified: false`; filesystem hashes do not attest GPU tensors.

## Observations

Each slot has two workers. Qualification observes all four workers rather than
treating a single readiness response as a replica barrier. Continuous traffic uses
two client loops and a fixed boolean decision. Its denominator excludes canaries,
deliberate negative checks and the separately held partial-body request.

| Exercise | Artifacts | Switches including reconciliation | Strict continuous requests | Errors |
|---|---|---:|---:|---:|
| CPU controlled-engine fixture | `accef86`, same code | 14 | 1,116 | 0 |
| Native vLLM, replica replacement | `accef86`, source imports | 13 | 446 | 0 |
| Native SGLang, replica replacement | `accef86`, source imports | 13 | 326 | 0 |
| Native vLLM, wheel upgrade/rollback | `accef86` ↔ `7739145` | 13 | 333 | 0 |
| Native SGLang, wheel upgrade/rollback | `accef86` ↔ `7739145` | 13 | 338 | 0 |

Every exercise verifies:

1. Complete HTTP headers with an incomplete JSON body retain the original backend
   across the switch. Drain refuses success while that stream remains, even before
   a registry lease exists. Completing its body returns a valid old-slot decision.
2. The next request on a persistent client connection selects the new slot.
3. The inactive slot reaches zero proxy streams/queues and zero tracked leases.
4. An incorrect requested release is rejected without changing traffic.
5. Interruption after live map mutation retains a pending intent. Reconciliation
   adopts the observed selection without another switch.
6. HAProxy remains alive with the same identity. The installed-wheel campaign also
   checks native PID/start-tick/boot identity before cleanup.

The CPU fixture additionally holds a gateway request, switches traffic, cancels
through the new slot, confirms HTTP 499 from the old request and observes drained
leases. This is **not** a claim of GPU cancellation during a code upgrade. Earlier
native cancellation tests remain separate; expanded GPU/fault/long-request rollout
coverage is still needed.

Local checks at `7739145`: **357 Python tests**, Ruff, dependency checks, CLI help
and both sets of three source-matched wheels pass. Fault cases include all journal
crash windows, lost acknowledgements, stale CAS, operation-ID reuse, missing or
unhealthy workers, unknown ownership, incompatible registry/model/engine/auth
policy, retained streams/dead-owner leases and proxy restart during a switch.
TypeScript code did not change and was not rerun at this checkpoint.

## Cleanup and reproduction

The audit verifies **129 historical owned process records**, including all gateway
and proxy groups: none remain live. GPU7 returned to 11,990 MiB free. Six model
files were rehashed against fixed Hub revision metadata; both tokenizer payloads
were verified. Native core/plugin imports point to the recorded source. Unrelated
services were not stopped or reconfigured.

Reports, hashes and process records are in
[`evidence/dsw/gateway-rollout-7739145`](../evidence/dsw/gateway-rollout-7739145).
Credentials, raw service logs, registry databases and model weights are excluded.
The verifier checks export hashes, harness/source bytes, installed core bytes,
traffic denominators and cleanup:

```sh
python evidence/harnesses/verify_gateway_rollout_7739145.py
```

See the [verified summary](../evidence/dsw/gateway-rollout-7739145/verified-summary.json),
[package checks](../evidence/package-check-7739145.json) and
[operator procedure](gateway-rollout.md). Earlier maintenance-window transitions
remain recorded in [the previous exercise](release-rollout-validation.md).

This establishes the tested same-host, compatible-schema path for these artifacts
and this engine/model profile. Proxy/host high availability, automatic failure
routing, multi-node coordination, schema migration/downgrade, engine/CUDA rollout,
performance certification, 24-hour soak, deployment images and handoff remain
separate unfinished gates. Model functional coverage remains **24/40 combinations**;
numerical and business-quality gates are separate. P18/P19/P24 remain partial.

The later [native cancellation follow-up](native-rollout-cancellation.md) exercises
real scoring RPCs across a different installed-wheel rollback on both engines.
It retains separate source identities, traffic denominators and initial failures.
