# Declared decision response model

`POST /v1/decisions` now declares its existing `DecisionResponse` return type.
FastAPI can use the model serializer and publish its response schema in OpenAPI,
rather than recursively converting an untyped response with `jsonable_encoder`.
This uses the framework's public response-model path; it does not bypass response
validation or change decision scoring, durable leases, cancellation or HTTP status.

The shared SDK fixtures cover ten valid response forms. HTTP tests compare parsed
JSON against the previous encoder, including explicit nulls, numeric values,
Unicode/escaped strings, partial/failed outcomes, timing and worker headers. The
existing auth, cancellation, deadline, partial-response and recovery-only guards
remain in the full regression suite.

A task-owned ASGI call profiler is used only in isolated DSW diagnostics. Its
instrumented timings are not normal throughput or GPU measurements. The first
per-request thread-clock experiment produced invalid negative SGLang timings and
an incomplete SIGTERM shutdown; retain those records and explicit recovery rather
than treating the experiment as a successful performance result. Corrected
continuous-window call profiles and separate unprofiled measurements are required
before reporting any measured benefit. No performance release gate follows from
the implementation alone.

See [the completed scoped validation](response-serialization-validation.md) for
corrected call profiles, matched unprofiled measurements and retained failure/recovery.
