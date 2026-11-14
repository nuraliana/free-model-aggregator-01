// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// WHAT A TRIAL GIVES YOU
//   A trial gives this server a brief and an env, and runs it as `node /bot/main.ts` from your
//   repository's root, with Node 22's standard library alone (nothing is installed; TypeScript runs
//   directly, and a relative import needs its .ts extension). The brief is at the path in
//   $MAKEFUN_PROMPT_FILE. It carries three JSON documents:
//     - the provider directory: ten simulated free-tier providers, each with its id, wire (openai,
//       anthropic or gemini), auth scheme, key environment variable, the key the service takes, and the
//       base path you join to $MAKEFUN_ENV_URL;
//     - the quality table: every model's public rating and what it can do (tools, vision, JSON mode,
//       context length), and whether it is free;
//     - the workload manifest: the twenty requests the client will send, in order, with each one's api,
//       whether it streams, its weight and the requirements a model must meet to serve it.
//   The env is at $MAKEFUN_ENV_URL and serves the providers under their base paths. This server reads
//   the brief, probes each provider's catalog route in the provider's own wire, builds one catalog, then:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//        x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//        /v1/messages, each with a JSON body or with server-sent events;
//     4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
//   body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// HOW THIS ENTRY ROUTES (quality 60%, served 15%, wire 10%, failover 10%, catalog 5%)
//   - quality: every free model that meets the request's requirements is ranked by the quality table's
//     rating_milli, best first; ties go to the provider called least so far, then by provider and model
//     id. The best model is always tried first: a refusal costs one extra upstream call, but a fallback
//     to a worse model costs quality on every request that takes it.
//   - wires: all six client x provider combinations are translated in lib/wire.ts, for JSON and for
//     SSE, with usage and stop reasons translated so the client's checks see the provider's numbers.
//     An anthropic-wire provider can never serve a JSON-mode request (its wire cannot carry JSON mode,
//     so its echo would always say json_mode false against a client that asked for it), so those are
//     excluded from JSON-mode candidates.
//   - failover: a 402, 429 or 5xx (or an unreachable upstream) moves the request to the next candidate:
//     each provider's best model first, then the other models of providers already tried, then the
//     providers that have already refused twice. Exactly one successful provider call per request.
//   - served: spreading happens through the rating ties and the failover order; a provider that keeps
//     refusing is tried last, not first, so one spent budget cannot take the rest of the workload down.
//   - catalog: the probe reports what each listing said, with the quality table's free flag and
//     capabilities where it has a row; a listing that hides its prices leaves the row short.
//   The trial's rules are in the kit's RULES.txt; read them for the score's columns and the tie-break.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the auth headers, the catalog probe.
//   lib/wire.ts        Anthropic <-> Chat Completions <-> Gemini translation and the SSE writers.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToGeminiRequest,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  openAIToAnthropicRequest,
  openAIToGeminiRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one upstream provider call may take, in milliseconds. */
const UPSTREAM_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Upstream calls made per provider id, this trial: the tie-break between equal ratings. */
const providerCalls = new Map<string, number>();
/** Upstream refusals (402, 429, 5xx, unreachable) per provider id, this trial. */
const providerFails = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that meets the request's requirements, best rating first; rating ties go to the
 * provider called least so far, then by provider and model id. A JSON-mode request never goes to an
 * anthropic-wire provider: that wire cannot carry JSON mode, so its echo would always say json_mode
 * false against a client that asked for it.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort(
    (a, b) =>
      b.rating - a.rating ||
      (providerCalls.get(a.probe.provider.id) ?? 0) - (providerCalls.get(b.probe.provider.id) ?? 0) ||
      (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) ||
      (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0),
  );
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for a
  // brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * The upstream URL and body for one candidate: the client's request translated into the provider's
 * own wire. OpenAI streams ask for the usage chunk too (the client never sets stream_options, so the
 * provider's stream would carry no usage without it); Gemini takes the model in the path, and the
 * stream flag picks generateContent or streamGenerateContent.
 */
function upstreamRequest(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json): { url: string; body: Json } {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const wire = candidate.probe.provider.wire;
  if (wire === 'openai') {
    const upstream: Json = clientWire === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (upstream.stream === true) upstream.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, body: upstream };
  }
  if (wire === 'anthropic') {
    const upstream: Json = clientWire === 'openai' ? openAIToAnthropicRequest(body, candidate.model.id) : { ...body, model: candidate.model.id };
    return { url: `${base}/v1/messages`, body: upstream };
  }
  const upstream: Json = clientWire === 'openai' ? openAIToGeminiRequest(body) : anthropicToGeminiRequest(body);
  const action = body.stream === true ? 'streamGenerateContent' : 'generateContent';
  return { url: `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`, body: upstream };
}

/** POST one request to the candidate provider; a refusal counts against that provider. */
async function callCandidate(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const { url, body: upstreamBody } = upstreamRequest(candidate, clientWire, body);
  const id = candidate.probe.provider.id;
  providerCalls.set(id, (providerCalls.get(id) ?? 0) + 1);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(candidate.probe.provider.auth, candidate.probe.provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  const res = await fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(upstreamBody),
    signal: AbortSignal.timeout(UPSTREAM_TIMEOUT_MS),
  });
  if (!res.ok) providerFails.set(id, (providerFails.get(id) ?? 0) + 1);
  return res;
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** A provider's JSON answer translated back into the client's wire, naming the model that served it. */
async function answerJson(clientWire: 'openai' | 'anthropic', candidate: Candidate, upstream: Response): Promise<Json> {
  const answer: unknown = await upstream.json();
  const json = isRecord(answer) ? answer : {};
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  if (clientWire === 'openai') {
    if (wire === 'openai') return json;
    if (wire === 'anthropic') return anthropicToOpenAIResponse(json, model);
    return geminiToOpenAI(json, model);
  }
  if (wire === 'openai') return openAIToAnthropic(json, model);
  if (wire === 'anthropic') return json;
  return geminiToAnthropic(json, model);
}

/** A provider's event stream translated back into the client's wire, naming the model that served it. */
async function answerStream(clientWire: 'openai' | 'anthropic', candidate: Candidate, upstream: Response, res: ServerResponse): Promise<void> {
  const wire = candidate.probe.provider.wire;
  const model = candidate.model.id;
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (upstream.body === null) {
    res.end();
    return;
  }
  if (clientWire === 'openai') {
    if (wire === 'openai') {
      await pipeStream(upstream.body, res);
      return;
    }
    const events = wire === 'anthropic' ? openAIEventsFromAnthropic(upstream.body, model) : openAIEventsFromGemini(upstream.body, model);
    for await (const event of events) res.write(event);
    res.end();
    return;
  }
  if (wire === 'openai') {
    for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
    res.end();
    return;
  }
  if (wire === 'anthropic') {
    await pipeStream(upstream.body, res);
    return;
  }
  for await (const event of anthropicEventsFromGemini(upstream.body, model)) res.write(event);
  res.end();
}

/**
 * One client request, served by the best free model that meets it, with failover: each provider's best
 * model first, then the other models of providers already tried, then the providers that have already
 * refused twice. Stops at the first successful provider call: exactly one success per request.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, requires.json === true);
  const fresh = candidates.filter((c) => (providerFails.get(c.probe.provider.id) ?? 0) < 2);
  const tired = candidates.filter((c) => (providerFails.get(c.probe.provider.id) ?? 0) >= 2);
  const seen = new Set<string>();
  const firstOfProvider = fresh.filter((c) => {
    if (seen.has(c.probe.provider.id)) return false;
    seen.add(c.probe.provider.id);
    return true;
  });
  const restOfFresh = fresh.filter((c) => !firstOfProvider.includes(c));
  const ordered = [...firstOfProvider, ...restOfFresh, ...tired];
  for (const candidate of ordered) {
    let upstream: Response;
    try {
      upstream = await callCandidate(candidate, clientWire, body);
    } catch (err) {
      providerFails.set(candidate.probe.provider.id, (providerFails.get(candidate.probe.provider.id) ?? 0) + 1);
      log(`upstream ${candidate.probe.provider.id}/${candidate.model.id} unreachable: ${(err as Error).message}`);
      continue;
    }
    if (!upstream.ok) {
      log(`upstream ${candidate.probe.provider.id}/${candidate.model.id} answered HTTP ${upstream.status}`);
      await upstream.text().catch(() => '');
      continue;
    }
    if (stream) {
      await answerStream(clientWire, candidate, upstream, res);
      return;
    }
    sendJson(res, 200, await answerJson(clientWire, candidate, upstream));
    return;
  }
  sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model that meets it. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
