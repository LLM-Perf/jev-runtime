// Explicit opt-in integration check; never part of the offline npm test glob.
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { JevClient, JevAPIError } from "../dist/index.js";

const baseURL = process.env.JEV_BASE_URL;
const apiKey = process.env.JEV_API_KEY;
const output = process.env.JEV_REPORT_FILE;
if (!baseURL || !apiKey || !output) throw new Error("Set JEV_BASE_URL, JEV_API_KEY and JEV_REPORT_FILE");
const report = {
  source_commit: process.env.JEV_SOURCE_COMMIT,
  runtime_source_commit: process.env.JEV_RUNTIME_SOURCE_COMMIT,
  node_version: process.version,
  qualification: "real HTTP SDK contract check; not model accuracy or performance certification",
  checks: {},
};
try {
  const client = new JevClient({ baseURL, apiKey });
  assert.equal(await client.ready(), true);
  const request = {
    model: "decision-model", input: { text: "I would like a refund for this broken item." },
    questions: [
      { id: "refund", type: "boolean", instruction: "Is a refund requested?" },
      { id: "category", type: "choice", instruction: "Classify the customer's intent.",
        options: [{ id: "refund", description: "Request a refund" }, { id: "hello", description: "Say hello" }] },
      { id: "urgency", type: "score", instruction: "Rate urgency.",
        options: [{ id: "low", description: "Low urgency", value: 0 }, { id: "high", description: "High urgency", value: 1 }] },
      { id: "relevance", type: "rank", instruction: "Rank these intents by relevance.",
        options: [{ id: "refund", description: "Request a refund" }, { id: "hello", description: "Say hello" }] },
    ],
  };
  const result = await client.decide(request);
  assert.equal(result.status, "completed");
  assert.equal(result.usage.successful_questions, 4);
  assert.deepEqual(Object.keys(result.answers).sort(), ["category", "refund", "relevance", "urgency"]);
  report.checks.four_typed_answers = result;
  const unauthorized = new JevClient({ baseURL, apiKey: "intentionally-invalid" });
  await assert.rejects(unauthorized.decide(request), (error) => error instanceof JevAPIError && error.status === 401);
  report.checks.unauthorized_rejected = true;
  assert.equal(await client.cancel("missing-sdk-" + crypto.randomUUID()), false);
  report.checks.missing_request_not_acknowledged = true;
  report.passed = true;
} catch (error) {
  report.passed = false;
  report.failure = { type: error.constructor.name, message: String(error.message).slice(0, 1000) };
  throw error;
} finally {
  writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx" });
  console.log(JSON.stringify({ passed: report.passed, checks: Object.keys(report.checks) }));
}
