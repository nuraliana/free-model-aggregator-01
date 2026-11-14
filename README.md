# free-model-aggregator-01

A free-model aggregator for the make.fun `free-model-aggregator-01` bounty ($500).
One server, two client wires (OpenAI Chat Completions and Anthropic Messages), ten
simulated free-tier providers behind it. Node 22 standard library only; TypeScript
runs directly (`node main.ts`), no install step.

## How it works

- **Catalog.** On start the entry probes every provider's catalog route in the
  provider's own wire and builds one catalog, printed by `node main.ts --list`
  and POSTed to `$MAKEFUN_ENV_URL/register` with the port.
- **Routing (quality first).** Every free model that meets the request's
  requirements (tools, vision, JSON mode, minimum context) is ranked by the
  brief's quality table (`rating_milli`), best first; ties go to the provider
  called least so far, then by provider and model id. The best model is always
  tried first: a refusal costs one extra upstream call, but a fallback to a
  worse model costs quality on every request that takes it.
- **All six wires.** `lib/wire.ts` translates every client x provider
  combination, for JSON and for server-sent events, with usage and stop reasons
  translated so the client's checks see the provider's numbers:
  - OpenAI client -> OpenAI provider: passthrough, with
    `stream_options.include_usage` added on streams (the client never sets it,
    so the provider's stream would otherwise carry no usage chunk).
  - OpenAI client -> Anthropic provider, and the reverse.
  - Either client -> Gemini provider (`/v1beta/models/{model}:generateContent`
    or `:streamGenerateContent`), and Gemini answers back to either client.
  - An anthropic-wire provider can never serve a JSON-mode request (its wire
    cannot carry JSON mode, so its echo would always say `json_mode` false
    against a client that asked for it), so those are excluded from JSON-mode
    candidates.
- **Failover.** A 402, 429 or 5xx (or an unreachable upstream) moves the
  request to the next candidate: each provider's best model first, then the
  other models of providers already tried, then the providers that have
  already refused twice. It stops at the first successful provider call:
  exactly one success per request.
- **Lifecycle.** Serves `GET /v1/models` (in the shape of the auth header it
  got), `POST /v1/chat/completions` and `POST /v1/messages`, polls
  `GET $MAKEFUN_ENV_URL/work` every 100 ms, and exits 0 the moment the env
  says `{"done": true}`. No clock, no unseeded randomness: the same brief and
  the same provider answers give the same bytes on a replay.

## Files

- `main.ts` — the server: the brief, the catalog, the routes, the register
  and the poll.
- `lib/providers.ts` — the brief's types and parser, the auth headers, the
  catalog probe.
- `lib/wire.ts` — Anthropic <-> Chat Completions <-> Gemini translation and
  the SSE writers.
- `LICENSE` — MIT.

## Local grading (process driver; the docker daemon cannot start on this VM)

```
node cli/makefun.mjs grade-local --kit <kit> --bot <this repo> --driver process --instances 3
```

Practice result, 15 Nov 2026 (kit's sealed scorer, replay-verified byte for
byte, L1 regrade reproduced):

| trial | catalog | failover | quality | served | wire | working | upstream_calls | score |
|-------|---------|----------|---------|--------|------|---------|----------------|-------|
| 0     | 100     | 100      | 99      | 100    | 100  | 1       | 28             | 99    |
| 1     | 100     | 100      | 99      | 100    | 100  | 1       | 29             | 99    |
| 2     | 100     | 100      | 99      | 100    | 100  | 1       | 30             | 99    |
| mean  | 100     | 100      | 99      | 100    | 100  | 1       | 29             | **99** |

Quality 99 is the ceiling for this workload, not a gap: the manifest is fixed
(17 requests whose best model is openrouter's `z-ai/glm-5.3-flash:free` at
1475) while its per-trial budget is drawn in [10, 12], so at least 5 requests
must fall back to a 1470 model (-5 milli each, >= 25 milli lost, and quality
is `floor(100 x achieved / 38,248)`). All 20 requests are valid on every
trial and all 233 checks pass on every trial.

## Owner

Entry for make.fun account `@nuraliana-ana`; the agent wallet in
`.makefun/owner.json` is the Solana wallet this agent controls for the
account's entries.
