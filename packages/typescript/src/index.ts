export interface Option {
  id: string;
  description: string;
  value?: number;
}

export type Question = {
  id: string;
  instruction: string;
} & (
  | { type: "boolean"; options?: never }
  | { type: "choice" | "score" | "rank"; options: readonly Option[] }
);

export interface DecisionRequest {
  request_id?: string;
  model: string;
  bundle?: string;
  input: { text: string };
  questions?: readonly Question[];
  execution?: { timeout_ms?: number; allow_partial?: boolean };
}

export interface Answer {
  type: Question["type"];
  status: "answered" | "abstained" | "failed";
  value: string | boolean | number | readonly string[] | null;
  probabilities: Record<string, number> | null;
  probability_semantics: "conditional_label_distribution" | "normalized_support" | null;
  support: Record<string, number> | null;
  label_mass: number | null;
  calibration_status: "uncalibrated" | "calibrated";
  abstained: boolean;
  reason: string | null;
  levels: Record<string, number> | null;
  error: { code: string; message: string } | null;
}

export interface DecisionResponse {
  request_id: string;
  status: "completed" | "partial" | "failed";
  bundle: string;
  bundle_digest: string;
  generation: number;
  engine: { name: string; version: string };
  answers: Record<string, Answer>;
  usage: {
    questions: number;
    successful_questions: number;
    scoring_sequences: number;
    logical_prompt_tokens: number;
    engine_prompt_tokens: number | null;
    engine_completion_tokens: number | null;
    cached_prompt_tokens: number | null;
  };
  latency_ms: number;
}

export interface ClientOptions {
  /** Include /plugins/jev-runtime when attaching directly to an engine plugin. */
  baseURL: string;
  apiKey?: string;
  timeoutMs?: number;
  fetch?: typeof globalThis.fetch;
}

export class JevAPIError extends Error {
  constructor(public readonly status: number, public readonly code: string, message: string) {
    super(message);
    this.name = "JevAPIError";
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function invalid(message: string): never {
  throw new JevAPIError(502, "invalid_response", message);
}

function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function numericRecord(value: unknown): value is Record<string, number> {
  return isRecord(value) && Object.values(value).every(finite);
}

function sameKeys(left: Record<string, unknown>, right: Record<string, unknown>): boolean {
  return Object.keys(left).length === Object.keys(right).length &&
    Object.keys(left).every((key) => Object.hasOwn(right, key));
}

function close(left: number, right: number): boolean {
  return Math.abs(left - right) <= Math.max(1e-6, 1e-6 * Math.max(Math.abs(left), Math.abs(right)));
}

function validateAnswer(answer: unknown): boolean {
  if (!isRecord(answer) || !["choice", "boolean", "score", "rank"].includes(String(answer.type)) ||
      !["answered", "abstained", "failed"].includes(String(answer.status)) ||
      !["calibrated", "uncalibrated"].includes(String(answer.calibration_status)) ||
      typeof answer.abstained !== "boolean") {
    invalid("Invalid typed answer");
  }
  if (answer.status === "failed") {
    if (!isRecord(answer.error) || typeof answer.error.code !== "string" || !answer.error.code ||
        typeof answer.error.message !== "string" || !answer.error.message || answer.abstained ||
        [answer.value, answer.probabilities, answer.probability_semantics, answer.support,
          answer.label_mass, answer.levels, answer.reason].some((item) => item !== null)) {
      invalid("Failed answer requires an error and cannot contain scoring values");
    }
    return false;
  }
  const probs = answer.probabilities;
  if (answer.error !== null || !numericRecord(probs) || Object.keys(probs).length < 2 ||
      Object.values(probs).some((p) => p < 0 || p > 1) ||
      Math.abs(Object.values(probs).reduce((sum, p) => sum + p, 0) - 1) > 1e-4) {
    invalid("Answer probabilities are missing or invalid");
  }
  if (answer.label_mass !== null &&
      (!finite(answer.label_mass) || answer.label_mass < 0 || answer.label_mass > 1.0001)) {
    invalid("Label mass is invalid");
  }
  if (answer.type === "boolean" && !sameKeys(probs, {true: 0, false: 0})) {
    invalid("Boolean answer requires the true/false distribution");
  }
  if (answer.probability_semantics === "normalized_support") {
    const support = answer.support;
    if (!numericRecord(support) || !sameKeys(probs, support) || answer.label_mass !== null ||
        Object.values(support).some((p) => p < 0 || p > 1)) {
      invalid("Independent support must match all candidates and has no label mass");
    }
    const total = Object.values(support).reduce((sum, p) => sum + p, 0);
    if (total <= 0 || Object.entries(probs).some(([key, p]) => !close(p, support[key]! / total))) {
      invalid("Probabilities disagree with independent support");
    }
  } else if (answer.probability_semantics !== "conditional_label_distribution" || answer.support !== null) {
    invalid("Conditional label distributions cannot include independent support");
  }
  const levels = answer.levels;
  if (levels !== null && (answer.type !== "score" || !numericRecord(levels) || !sameKeys(probs, levels))) {
    invalid("Numeric score levels must match all candidates");
  }
  if (answer.status === "abstained") {
    if (!answer.abstained || typeof answer.reason !== "string" || !answer.reason || answer.value !== null) {
      invalid("Abstained answers require a reason and no selected value");
    }
    return true;
  }
  if (answer.abstained || answer.reason !== null) invalid("Answered values cannot be marked as abstained");
  let selected: string;
  if (answer.type === "boolean") {
    if (typeof answer.value !== "boolean") invalid("Boolean value has the wrong type");
    selected = answer.value ? "true" : "false";
  } else if (answer.type === "rank") {
    const order = answer.value;
    if (!Array.isArray(order) || order.length !== Object.keys(probs).length ||
        new Set(order).size !== order.length ||
        order.some((key) => typeof key !== "string" || !Object.hasOwn(probs, key))) {
      invalid("Rank must include every candidate exactly once");
    }
    if (order.some((key, index) => index > 0 && probs[order[index - 1]]! < probs[key]! - 1e-12)) {
      invalid("Rank must follow descending probability");
    }
    return true;
  } else if (answer.type === "score" && levels !== null) {
    // The check above establishes a finite numeric record with matching keys.
    const expected = Object.entries(probs).reduce((sum, [key, p]) => sum + p * (levels as Record<string, number>)[key]!, 0);
    if (!finite(answer.value) || !finite(expected) || !close(answer.value, expected)) {
      invalid("Numeric score must equal the explicit level expectation");
    }
    return true;
  } else {
    if (typeof answer.value !== "string" || !Object.hasOwn(probs, answer.value)) {
      invalid("Choice and categorical scores must name a candidate");
    }
    selected = answer.value;
  }
  if (probs[selected]! < Math.max(...Object.values(probs)) - 1e-12) {
    invalid("Selected answer must have maximal probability");
  }
  return true;
}

/** Reject malformed successful responses instead of fabricating typed defaults. */
export function parseDecision(value: unknown): DecisionResponse {
  if (!isRecord(value) || typeof value.request_id !== "string" || !value.request_id ||
      !["completed", "partial", "failed"].includes(String(value.status)) ||
      typeof value.bundle !== "string" || !value.bundle || typeof value.bundle_digest !== "string" ||
      !/^sha256:[a-f0-9]{64}$/.test(value.bundle_digest) ||
      !Number.isInteger(value.generation) || (value.generation as number) < 1 || !isRecord(value.answers) ||
      !isRecord(value.usage) || !isRecord(value.engine) ||
      typeof value.engine.name !== "string" || !value.engine.name ||
      typeof value.engine.version !== "string" || !value.engine.version ||
      !finite(value.latency_ms) || value.latency_ms < 0) {
    invalid("Server returned an invalid decision contract");
  }
  for (const key of ["questions", "successful_questions", "scoring_sequences", "logical_prompt_tokens",
    "engine_prompt_tokens", "engine_completion_tokens", "cached_prompt_tokens"]) {
    const count = value.usage[key];
    if (count === null && ["engine_prompt_tokens", "engine_completion_tokens", "cached_prompt_tokens"].includes(key)) continue;
    if (!Number.isInteger(count) || (count as number) < (key === "questions" ? 1 : 0)) {
      invalid("Usage counts must be nonnegative integers or explicitly unknown");
    }
  }
  const answers = Object.values(value.answers);
  const successful = answers.filter(validateAnswer).length;
  const expected = successful === answers.length ? "completed" : successful === 0 ? "failed" : "partial";
  if (answers.length !== value.usage.questions || successful !== value.usage.successful_questions || value.status !== expected) {
    invalid("Decision status and success counts disagree");
  }
  return value as unknown as DecisionResponse;
}

export class JevClient {
  private readonly baseURL: string;
  private readonly fetcher: typeof globalThis.fetch;
  constructor(private readonly options: ClientOptions) {
    this.baseURL = options.baseURL.replace(/\/+$/, "");
    const url = new URL(this.baseURL);
    if (!["http:", "https:"].includes(url.protocol)) throw new TypeError("Use an HTTP(S) base URL");
    if (options.timeoutMs !== undefined && (!(options.timeoutMs > 0) || !Number.isFinite(options.timeoutMs))) {
      throw new TypeError("timeoutMs must be finite and positive");
    }
    this.fetcher = options.fetch ?? globalThis.fetch;
  }

  private async call(path: string, body?: unknown, signal?: AbortSignal, timeoutMs = 40000): Promise<unknown> {
    const controller = new AbortController();
    const forward = () => controller.abort(signal?.reason);
    if (signal?.aborted) forward();
    else signal?.addEventListener("abort", forward, { once: true });
    const timer = setTimeout(() => controller.abort(new DOMException("Jev request timed out", "TimeoutError")),
      this.options.timeoutMs ?? timeoutMs);
    const headers: Record<string, string> = { Accept: "application/json" };
    if (this.options.apiKey) headers.Authorization = `Bearer ${this.options.apiKey}`;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    try {
      const response = await this.fetcher(this.baseURL + path, {
        method: body === undefined ? "GET" : "POST", headers,
        ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: controller.signal,
      });
      let payload: unknown;
      try { payload = await response.json(); }
      catch {
        if (!response.ok) throw new JevAPIError(response.status, "http_error", `Jev request failed (${response.status})`);
        invalid("Server response is not JSON");
      }
      if (!response.ok) {
        const error = isRecord(payload) && isRecord(payload.error) ? payload.error : {};
        throw new JevAPIError(response.status,
          typeof error.code === "string" && error.code ? error.code : "http_error",
          typeof error.message === "string" && error.message ? error.message : `Jev request failed (${response.status})`);
      }
      return payload;
    } finally {
      clearTimeout(timer);
      signal?.removeEventListener("abort", forward);
    }
  }

  async decide(request: DecisionRequest, options: { signal?: AbortSignal } = {}): Promise<DecisionResponse> {
    return parseDecision(await this.call("/v1/decisions", request, options.signal,
      (request.execution?.timeout_ms ?? 30000) + 10000));
  }

  async cancel(requestId: string, options: { signal?: AbortSignal } = {}): Promise<boolean> {
    const value = await this.call(`/v1/requests/${encodeURIComponent(requestId)}/cancel`, {}, options.signal);
    if (!isRecord(value) || typeof value.cancelled !== "boolean") {
      throw new JevAPIError(502, "invalid_response", "Invalid cancellation response");
    }
    return value.cancelled;
  }

  async ready(options: { signal?: AbortSignal } = {}): Promise<boolean> {
    const value = await this.call("/ready", undefined, options.signal);
    return isRecord(value) && value.ready === true;
  }
}
