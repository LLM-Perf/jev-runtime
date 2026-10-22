import assert from "node:assert/strict";
import { test } from "node:test";
import { JevClient, JevAPIError, parseDecision } from "../dist/index.js";

const answer = { type: "boolean", status: "answered", value: true,
  probabilities: {true: 0.8, false: 0.2}, probability_semantics: "conditional_label_distribution",
  support: null, label_mass: 0.9, calibration_status: "uncalibrated", abstained: false,
  reason: null, levels: null, error: null };
const valid = { request_id: "test", status: "completed", bundle: "task@1", bundle_digest: "sha256:" + "a".repeat(64),
  generation: 1, engine: {name: "fixture", version: "0"}, answers: {q: answer},
  usage: {questions: 1, successful_questions: 1, scoring_sequences: 1, logical_prompt_tokens: 10,
    engine_prompt_tokens: 10, engine_completion_tokens: 0, cached_prompt_tokens: null}, latency_ms: 1 };

test("uses native plugin prefix and bearer auth, preserving typed probabilities", async () => {
  const client = new JevClient({baseURL: "http://localhost/plugins/jev-runtime/", apiKey: "test-key",
    fetch: async (url, init) => {
      assert.equal(url, "http://localhost/plugins/jev-runtime/v1/decisions");
      assert.equal(init.headers.Authorization, "Bearer test-key");
      assert.equal(JSON.parse(init.body).model, "task");
      return Response.json(valid);
    }});
  assert.deepEqual(await client.decide({model: "task", input: {text: "x"}}), valid);
});

test("surfaces structured errors without automatic mutation retries", async () => {
  let calls = 0;
  const client = new JevClient({baseURL: "http://localhost", fetch: async () => {
    calls++; return Response.json({error: {code: "generation_conflict", message: "Changed"}}, {status: 409});
  }});
  await assert.rejects(client.decide({model: "x", input: {text: "x"}}),
    (e) => e instanceof JevAPIError && e.status === 409 && e.code === "generation_conflict");
  assert.equal(calls, 1);
});

test("rejects HTTP 200 with false success counts or missing probabilities", () => {
  assert.throws(() => parseDecision({...valid, usage: {...valid.usage, successful_questions: 0}}), JevAPIError);
  assert.throws(() => parseDecision({...valid, answers: {q: {...answer, probabilities: null}}}), JevAPIError);
  assert.throws(() => parseDecision({...valid, answers: {q: {...answer, probabilities: {true: 0.2, false: 0.2}}}}), JevAPIError);
});

test("timeout aborts the HTTP fetch", async () => {
  const client = new JevClient({baseURL: "http://localhost", timeoutMs: 10,
    fetch: async (_, init) => new Promise((resolve, reject) => {
      init.signal.addEventListener("abort", () => reject(init.signal.reason), {once: true});
    })});
  await assert.rejects(client.decide({model: "x", input: {text: "x"}}), (e) => e.name === "TimeoutError");
});

// Headers resolve immediately; the body stream errors when the request aborts,
// mirroring how fetch tears down an in-flight response body.
const abortedBodyFetch = (status, onCall) => async (_, init) => {
  onCall?.();
  return new Response(new ReadableStream({start(c) {
    // Fetch body readers can reject with AbortError even for a custom signal
    // reason or timeout; the client must retain the reason from its signal.
    init.signal.addEventListener("abort", () => c.error(new DOMException("Body aborted", "AbortError")), {once: true});
  }}), {status, headers: {"Content-Type": "application/json"}});
};

test("caller cancellation during body consumption preserves the custom reason", async () => {
  for (const status of [200, 503]) {
    let calls = 0;
    const client = new JevClient({baseURL: "http://localhost", fetch: abortedBodyFetch(status, () => calls++)});
    const caller = new AbortController();
    const reason = { message: "caller gave up" };
    const pending = client.decide({model: "x", input: {text: "x"}}, {signal: caller.signal});
    await new Promise((resolve) => setImmediate(resolve)); // headers arrived, body in flight
    caller.abort(reason);
    await assert.rejects(pending, (e) => e === reason);
    assert.equal(calls, 1);
  }
});

test("timeout during body consumption surfaces the TimeoutError", async () => {
  for (const status of [200, 503]) {
    let calls = 0;
    const client = new JevClient({baseURL: "http://localhost", timeoutMs: 10,
      fetch: abortedBodyFetch(status, () => calls++)});
    await assert.rejects(client.decide({model: "x", input: {text: "x"}}),
      (e) => e instanceof DOMException && e.name === "TimeoutError" && e.message === "Jev request timed out");
    assert.equal(calls, 1);
  }
});

test("body stream failures without an abort keep the response-contract errors", async () => {
  for (const [status, expected] of [[200, [502, "invalid_response"]], [503, [503, "http_error"]]]) {
    const client = new JevClient({baseURL: "http://localhost", fetch: async () =>
      new Response(new ReadableStream({start(c) { c.error(new Error("socket reset")); }}),
        {status, headers: {"Content-Type": "application/json"}})});
    await assert.rejects(client.decide({model: "x", input: {text: "x"}}),
      (e) => e instanceof JevAPIError && e.status === expected[0] && e.code === expected[1]);
  }
});
