# API reference

The gateway serves one prefix (default: none) with two planes. The data plane
authenticates with `Authorization: Bearer <api key>`; tenant keys select the
tenant. The admin plane lives under `/admin` and requires a separate admin
key. Every error uses one envelope:

```json
{"error": {"code": "generation_conflict", "message": "Active generation changed"}}
```

Every response carries `X-Jev-Worker` with the serving worker's owner ID.
Sending `X-Jev-Timing: 1` adds a `Server-Timing` header with serial runtime
phases. Disconnecting the client cancels the server-side decision and is
reported as `client_disconnected`.

## Data plane

| Method | Path | Purpose |
|---|---|---|
| POST | `/v1/decisions` | Typed decision; see `examples/request.json` |
| POST | `/v1/systemone` | Restricted System One compatibility form |
| GET | `/v1/models` | Active aliases this worker has validated |
| GET | `/v1/capabilities` | Probed engine capabilities and limits |
| POST | `/v1/requests/{request_id}/cancel` | Durable, tenant-scoped cancellation |
| GET | `/ready` | Readiness: canaries, admission, control health |
| GET | `/metrics` | Prometheus exposition |

## Admin plane

| Method | Path | Purpose |
|---|---|---|
| GET/POST | `/admin/bundles` | List bundles and routes; upload an immutable bundle |
| POST | `/admin/bundles/prepare` | Validate/canary a bundle on this worker |
| POST | `/admin/bundles/activate` | Atomic route switch (`expected_generation` CAS) |
| POST | `/admin/bundles/disable` | Remove a route without touching in-flight requests |
| POST | `/admin/bundles/retire` | Retire a drained bundle |
| POST | `/admin/compile` | Export validated scoring inputs without dispatch |
| GET/POST | `/admin/adapters`, `/admin/adapters/{register,load,unload}` | Managed LoRA lifecycle |
| GET | `/admin/workers` | Worker states, owner liveness, prepared digests |
| GET/POST | `/admin/quiescence` | Drain status; begin backend quiescence |
| GET | `/admin/profile` | Worker identity, deployment, admission and health |
| GET | `/admin/requests/recovery` | Leases eligible for explicit recovery |
| GET | `/admin/requests/{request_id}/progress` | This worker's view of one active request |
| POST | `/admin/requests/{request_id}/recover` | Confirm abort and release a retained lease |
| GET | `/admin/raw-requests/recovery`, POST `/admin/raw-requests/{work_id}/recover` | Raw engine work recovery |
| GET | `/admin/recovery-operations` | In-flight recovery claims |

Mutating admin calls take `expected_generation` where a concurrent change
must not be lost; a `generation_conflict` means refresh and retry.

## Error codes

Grouped by HTTP status; a code appears once per status it can carry.

| Status | Code | Meaning / client action |
|---|---|---|
| 400 | `adapter_path` | Adapter source is not a usable local directory of files |
| 400 | `invalid_token_budget` | Token demand must be positive |
| 401 | `unauthorized` | Missing/invalid API key |
| 401 | `admin_unauthorized` | Admin plane requires the separate admin key |
| 403 | `adapter_path` | Adapter source escapes configured local roots |
| 404 | `adapter_not_found`, `bundle_not_found`, `route_not_found` | Unknown immutable object or alias |
| 409 | `generation_conflict` | Route/control generation changed; refresh and retry |
| 409 | `duplicate_request`, `duplicate_admission` | Request/lease identity already in flight |
| 409 | `not_ready`, `bundle_preparing`, `invalid_state` | Object state does not allow this operation |
| 409 | `model_mismatch`, `engine_mismatch`, `worker_mismatch`, `adapter_backend_mismatch` | Identity binding violation |
| 409 | `lease_not_owned`, `admission_lease_mismatch` | Lease ownership/tenancy mismatch |
| 409 | `backend_conflict` | Two providers registered the same backend name |
| 409 | `adapter_collision`, `adapter_corrupted`, `adapter_format`, `adapter_profile`, `adapter_tensor`, `adapter_config` | Adapter identity/content/profile rejected |
| 409 | `template_file_invalid`, `template_file_mismatch`, `tokenizer_contract`, `tokenizer_profile_mismatch` | Template/tokenizer contract violation |
| 409 | `lora_unsupported` | Managed LoRA not enabled or not allowed here |
| 409 | `quiescence_pending`, `recovery_conflict`, `recovery_not_confirmed` | Recovery preconditions not met |
| 409 | `tokenspeed_profile` | Unsupported TokenSpeed profile (experimental backend) |
| 413 | `branch_budget`, `context_budget`, `option_budget`, `label_budget` | Request exceeds a static expansion budget |
| 413 | `engine_token_budget`, `engine_branch_budget`, `engine_context_limit`, `engine_label_limit` | Request exceeds engine/admission budgets |
| 413 | `adapter_config`, `score_contract` | Oversized adapter config; label-per-request limit |
| 429 | `queue_full` | Shared admission queue is full; retry later |
| 499 | `request_cancelled` | Decision cancelled via the cancel endpoint |
| 499 | `client_disconnected` | Client disconnected; work was cancelled |
| 502 | `engine_error`, `engine_request_failed`, `engine_aborted` | Engine scoring failed |
| 502 | `invalid_response`, `invalid_scores` | Engine/plugin response violates the contract |
| 502 | `missing_branches` | Not all candidate branches completed |
| 502 | `score_contract`, `score_position` | Scores violate raw/position/shape invariants |
| 503 | `route_unavailable`, `no_active_bundle`, `bundle_not_ready` | No prepared bundle can serve the alias |
| 503 | `not_ready`, `not_started`, `engine_unavailable`, `control_unavailable` | Worker not (yet) able to serve; retry later |
| 503 | `backend_quiescing` | Backend is draining; new work is disabled |
| 503 | `backend_contract` | Backend plugin does not implement the contract |
| 503 | `unsupported_engine` | Engine lacks a required capability |
| 503 | `cancellation_unconfirmed` | Abort not confirmed; the lease is retained for recovery |
| 503 | `adapter_load`, `adapter_unload`, `adapter_pin`, `adapter_barrier`, `adapter_not_ready` | Engine did not confirm an adapter transition |
| 504 | `deadline_exceeded` | The request's `execution.timeout_ms` expired |

Engine-native plugins additionally expose `/plugins/jev-runtime/v1/scores`,
`/scores/cancel` and `/scoring-capabilities`; see
[multi-engine](multi-engine.md)（中文）for that contract.
