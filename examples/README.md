# Examples

## Offline demo (no GPU)

```sh
python examples/demo_fixture.py
```

A self-contained serving loop against a deterministic in-process engine
double: it builds a registry, uploads two bundle versions through the admin
API, activates each with a generation check, and sends real
`/v1/decisions` requests through the Python SDK — including a zero-downtime
hot-switch from `support-intent@1` (generation 1) to `@2` (generation 2).

The demo exercises the serving contract (typed answers, probability
semantics, usage accounting, lifecycle). Its scores are fixed per label
position, so it is **not** model-quality, GPU-compatibility or performance
evidence. A CI test (`tests/test_demo_fixture.py`) keeps it runnable.

## Configuration samples

- `gateway.yaml` — gateway settings for a real engine deployment; replace
  `REPLACE_WITH_MODEL_COMMIT` with the pinned model revision.
- `request.json` — the decide request the demo sends (choice + boolean).
- `tokenspeed.yaml`, `tokenspeed-engine.json` — experimental TokenSpeed
  backend; paths under `/srv/...` are DSW-specific and must be adapted, see
  [multi-engine](../docs/multi-engine.md)（中文）.
