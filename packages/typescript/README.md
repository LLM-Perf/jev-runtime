# TypeScript client

Node 20+ and modern browsers with `fetch`. No runtime dependencies. This package
is private and is not published to npm. Build with `npm ci && npm run build`, then
install a locally packed archive or use this package in a private workspace.

```ts
import { JevClient } from "@llm-perf/jev-runtime";

const client = new JevClient({
  baseURL: "http://127.0.0.1:18795/plugins/jev-runtime",
  apiKey: process.env.JEV_API_KEY,
});
const result = await client.decide({
  model: "decision-model",
  input: { text: "I would like a refund." },
  questions: [{ id: "refund", type: "boolean", instruction: "Is a refund requested?" }],
});
console.log(result.answers.refund);
```

For a standalone gateway omit the plugin path from `baseURL`. Use a server-side
credential; do not put an administrative or shared engine key in browser bundles.
Pass `{signal: controller.signal}` to cancel the HTTP request. The server propagates
disconnect cancellation to engine scoring. No automatic retries are made.

Inspect `status`, `calibration_status`, and `probability_semantics`; a typed answer
does not imply calibrated confidence. Partial/failing responses retain their status.
Both SDKs reject malformed HTTP-200 decisions with `invalid_response` (502),
including inconsistent status/counts, non-normalized distributions, invalid selected
IDs, duplicate or unordered rankings, and numeric scores that disagree with the
probability-weighted explicit levels. Boolean values must be actual booleans.
Independent support must normalize to the returned probabilities; abstentions must
have a reason and no selected value. This validates the response contract, not the
model's real-world accuracy. `cancel()` accepts only an actual boolean response and
preserves the service's status/code on errors, including non-JSON gateway errors.

The opt-in `test/live.mjs` checks all four answer types, authentication errors and
unknown-request cancellation against a real service. Build first, then set
`JEV_BASE_URL`, `JEV_API_KEY` and a new `JEV_REPORT_FILE`; optional source-commit
environment fields label the artifact. Run `node test/live.mjs`. It is excluded
from the offline test glob and never prints the credential.
