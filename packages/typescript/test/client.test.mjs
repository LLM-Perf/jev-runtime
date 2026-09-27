import assert from "node:assert/strict";
import { test } from "node:test";
import { JevClient, JevAPIError, parseDecision } from "../dist/index.js";

const answer = { type: "boolean", status: "answered", value: true,
  probabilities: {true: 0.8, false: 0.2}, probability_semantics: "conditional_label_distribution",
  support: null, label_mass: 0.9, calibration_status: "uncalibrated", abstained: false,
  reason: null, levels: null, error: null };
const valid = { request_id: "test", status: "completed", bundle: "task@1", bundle_digest: "sha256:abc",
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
});

test("timeout aborts the HTTP fetch", async () => {
  const client = new JevClient({baseURL: "http://localhost", timeoutMs: 10,
    fetch: async (_, init) => new Promise((resolve, reject) => {
      init.signal.addEventListener("abort", () => reject(init.signal.reason), {once: true});
    })});
  await assert.rejects(client.decide({model: "x", input: {text: "x"}}), (e) => e.name === "TimeoutError");
});
