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
  probability_semantics: string | null;
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

function probabilities(value: unknown, normalized = true): boolean {
  return isRecord(value) && Object.keys(value).length > 0 && Object.values(value).every(
    (p) => typeof p === "number" && Number.isFinite(p) && p >= 0 && p <= 1,
  ) && (!normalized || Math.abs(Object.values(value).reduce<number>((sum, p) => sum + (p as number), 0) - 1) < 1e-4);
}

/** Reject malformed successful responses instead of fabricating typed defaults. */
export function parseDecision(value: unknown): DecisionResponse {
  if (!isRecord(value) || typeof value.request_id !== "string" || !value.request_id ||
      !["completed", "partial", "failed"].includes(String(value.status)) ||
      typeof value.bundle !== "string" || typeof value.bundle_digest !== "string" ||
      !Number.isInteger(value.generation) || !isRecord(value.answers) ||
      !isRecord(value.usage) || !isRecord(value.engine) ||
      typeof value.engine.name !== "string" || typeof value.engine.version !== "string" ||
      typeof value.latency_ms !== "number" || !Number.isFinite(value.latency_ms)) {
    throw new JevAPIError(502, "invalid_response", "Server returned an invalid decision contract");
  }
  const answers = Object.values(value.answers);
  let successful = 0;
  for (const answer of answers) {
    if (!isRecord(answer) || !["choice", "boolean", "score", "rank"].includes(String(answer.type)) ||
        !["answered", "abstained", "failed"].includes(String(answer.status)) ||
        !["calibrated", "uncalibrated"].includes(String(answer.calibration_status))) {
      throw new JevAPIError(502, "invalid_response", "Invalid typed answer");
    }
    if (answer.status === "failed") {
      if (!isRecord(answer.error) || typeof answer.error.code !== "string") {
        throw new JevAPIError(502, "invalid_response", "Failed answer has no error");
      }
      continue;
    }
    successful += 1;
    if (!probabilities(answer.probabilities)) {
      throw new JevAPIError(502, "invalid_response", "Answer probabilities are missing or invalid");
    }
    if (answer.support !== null && !probabilities(answer.support, false)) {
      throw new JevAPIError(502, "invalid_response", "Independent support values are invalid");
    }
    if (answer.status === "answered") {
      const v = answer.value;
      const valid = answer.type === "boolean" ? typeof v === "boolean" :
        answer.type === "rank" ? Array.isArray(v) && v.every((item) => typeof item === "string") :
        answer.type === "score" ? typeof v === "string" || (typeof v === "number" && Number.isFinite(v)) :
        typeof v === "string";
      if (!valid) throw new JevAPIError(502, "invalid_response", "Answer value has the wrong type");
    }
  }
  if (answers.length !== value.usage.questions || successful !== value.usage.successful_questions ||
      (value.status === "completed" && successful !== answers.length) ||
      (value.status === "failed" && successful !== 0) ||
      (value.status === "partial" && (successful === 0 || successful === answers.length))) {
    throw new JevAPIError(502, "invalid_response", "Decision status and success counts disagree");
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
      catch { throw new JevAPIError(response.status, "invalid_response", "Server response is not JSON"); }
      if (!response.ok) {
        const error = isRecord(payload) && isRecord(payload.error) ? payload.error : {};
        throw new JevAPIError(response.status,
          typeof error.code === "string" ? error.code : "http_error",
          typeof error.message === "string" ? error.message : `Jev request failed (${response.status})`);
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
