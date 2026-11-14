// The client wires and the one upstream wire this baseline speaks.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. This baseline forwards both to a
// provider on the openai wire and translates back:
//
//   - the OpenAI path passes the client's body (with the routed model's id) to the provider and streams
//     the provider's JSON or SSE straight back, the way an OpenAI-compatible gateway does;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and rewrites the provider's answer into a
//     Messages object, or into the message_start / content_block_start / content_block_delta /
//     content_block_stop / message_delta / message_stop event order, with the stop reason and the usage
//     translated. Tool calls become tool_use blocks and input_json_delta events.
//
// The gemini and anthropic upstream wires are not implemented in this baseline (main.ts answers 501 for a
// provider on either): a real entry needs their request and event shapes too, which is the biggest single
// gap in this starter.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and the
// same provider answer give the same bytes on a replay. The two model listings use fixed timestamps for the
// same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/* ---- Added: the other upstream wires (Anthropic, Gemini) and their translators. ---- */

/** OpenAI finish_reason from an Anthropic stop_reason. */
export function openAIFinishOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Anthropic stop_reason from an OpenAI finish_reason. */
export function anthropicStopOf(finish: unknown): string {
  return stopReasonOf(finish);
}

/** Gemini finishReason as an OpenAI finish_reason. */
export function openAIFinishOfGemini(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'length';
  if (finish === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini finishReason as an Anthropic stop_reason. */
export function anthropicStopOfGemini(finish: unknown): string {
  if (finish === 'MAX_TOKENS') return 'max_tokens';
  if (finish === 'SAFETY') return 'refusal';
  return 'end_turn';
}

function stopListOf(v: unknown): string[] | undefined {
  if (typeof v === 'string' && v !== '') return [v];
  if (Array.isArray(v)) {
    const out = v.filter((x): x is string => typeof x === 'string');
    return out.length > 0 ? out : undefined;
  }
  return undefined;
}

/** One OpenAI message's text (system or user), for building another wire's fields. */
function openAIText(content: unknown): string {
  return textOf(content);
}

/** An OpenAI image_url part as an Anthropic image block. */
function anthropicImageBlock(part: Json): Json | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
  const url = part.image_url.url;
  const m = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (m !== null) return { type: 'image', source: { type: 'base64', media_type: m[1] !== '' ? m[1] : 'image/png', data: m[2] } };
  return { type: 'image', source: { type: 'url', url } };
}

/** An OpenAI image_url part as a Gemini part. */
function geminiImagePart(part: Json): Json | null {
  if (part.type !== 'image_url' || !isRecord(part.image_url) || typeof part.image_url.url !== 'string') return null;
  const url = part.image_url.url;
  const m = /^data:([^;,]+)?;base64,(.*)$/s.exec(url);
  if (m !== null) return { inlineData: { mimeType: m[1] !== '' ? m[1] : 'image/png', data: m[2] } };
  return { fileData: { fileUri: url } };
}

/** An Anthropic image block as a Gemini part. */
function geminiImagePartFromAnthropic(block: Json): Json | null {
  if (block.type !== 'image' || !isRecord(block.source)) return null;
  const source = block.source;
  if (source.type === 'base64' && typeof source.data === 'string') {
    return { inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } };
  }
  if (source.type === 'url' && typeof source.url === 'string') return { fileData: { fileUri: source.url } };
  return null;
}

/** An OpenAI Chat Completions request as an Anthropic Messages request naming `model`. */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const messages: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = openAIText(raw.content);
      if (t !== '') system = system === '' ? t : `${system}\n${t}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const content: Json[] = [];
      const text = openAIText(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content),
          },
        ],
      });
      continue;
    }
    // user (or anything else treated as user)
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
        else {
          const img = anthropicImageBlock(block);
          if (img !== null) parts.push(img);
        }
      }
    }
    messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = stopListOf(body.stop);
  if (stops !== undefined) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
    else if (choice === 'none') out.tool_choice = { type: 'none' };
  } else if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice.type === 'any' || choice.type === 'required') out.tool_choice = { type: 'any' };
    else if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** OpenAI messages as Gemini contents; returns {systemInstruction, contents}. */
function openAIContents(body: Json): { systemInstruction: Json | null; contents: Json[] } {
  const contents: Json[] = [];
  let system = '';
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const t = openAIText(raw.content);
      if (t !== '') system = system === '' ? t : `${system}\n${t}`;
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      const text = openAIText(raw.content);
      if (text !== '') parts.push({ text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      contents.push({
        role: 'user',
        parts: [{ functionResponse: { name: '', response: { text: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) } } }],
      });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else {
          const img = geminiImagePart(block);
          if (img !== null) parts.push(img);
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  return { systemInstruction: system === '' ? null : { parts: [{ text: system }] }, contents };
}

/** Anthropic messages as Gemini contents; returns {systemInstruction, contents}. */
function anthropicContents(body: Json): { systemInstruction: Json | null; contents: Json[] } {
  const contents: Json[] = [];
  const system = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
          else if (block.type === 'tool_use') parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
        }
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    const parts: Json[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
        else if (block.type === 'tool_result') {
          parts.push({ functionResponse: { name: '', response: { text: toolResultText(block.content) } } });
        } else {
          const img = geminiImagePartFromAnthropic(block);
          if (img !== null) parts.push(img);
        }
      }
    }
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  return { systemInstruction: system === '' ? null : { parts: [{ text: system }] }, contents };
}

function geminiToolsFromOpenAI(body: Json): Json[] | undefined {
  if (!Array.isArray(body.tools)) return undefined;
  const decls = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
    return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
  });
  return decls.length > 0 ? [{ functionDeclarations: decls }] : undefined;
}

function geminiToolsFromAnthropic(body: Json): Json[] | undefined {
  if (!Array.isArray(body.tools)) return undefined;
  const decls = body.tools.flatMap((raw) => {
    if (!isRecord(raw) || typeof raw.name !== 'string') return [];
    return [{ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
  });
  return decls.length > 0 ? [{ functionDeclarations: decls }] : undefined;
}

function geminiToolConfig(choice: unknown): Json | undefined {
  if (typeof choice === 'string') {
    if (choice === 'auto') return { functionCallingConfig: { mode: 'AUTO' } };
    if (choice === 'required' || choice === 'any') return { functionCallingConfig: { mode: 'ANY' } };
    if (choice === 'none') return { functionCallingConfig: { mode: 'NONE' } };
    return undefined;
  }
  if (!isRecord(choice)) return undefined;
  if (choice.type === 'auto') return { functionCallingConfig: { mode: 'AUTO' } };
  if (choice.type === 'any' || choice.type === 'required') return { functionCallingConfig: { mode: 'ANY' } };
  if (choice.type === 'none') return { functionCallingConfig: { mode: 'NONE' } };
  if (choice.type === 'tool' && typeof choice.name === 'string') return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.name] } };
  if (choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') return { functionCallingConfig: { mode: 'ANY', allowedFunctionNames: [choice.function.name] } };
  return undefined;
}

/** An OpenAI Chat Completions request as a Gemini generateContent request. */
export function openAIToGeminiRequest(body: Json): Json {
  const { systemInstruction, contents } = openAIContents(body);
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  const stops = stopListOf(body.stop);
  if (stops !== undefined) config.stopSequences = stops;
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) config.responseMimeType = 'application/json';
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const out: Json = { contents };
  if (systemInstruction !== null) out.systemInstruction = systemInstruction;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const tools = geminiToolsFromOpenAI(body);
  if (tools !== undefined) out.tools = tools;
  const toolConfig = geminiToolConfig(body.tool_choice);
  if (toolConfig !== undefined) out.toolConfig = toolConfig;
  return out;
}

/** An Anthropic Messages request as a Gemini generateContent request. */
export function anthropicToGeminiRequest(body: Json): Json {
  const { systemInstruction, contents } = anthropicContents(body);
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  const stops = stopListOf(body.stop_sequences);
  if (stops !== undefined) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  const out: Json = { contents };
  if (systemInstruction !== null) out.systemInstruction = systemInstruction;
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const tools = geminiToolsFromAnthropic(body);
  if (tools !== undefined) out.tools = tools;
  const toolConfig = geminiToolConfig(body.tool_choice);
  if (toolConfig !== undefined) out.toolConfig = toolConfig;
  return out;
}

/** A Gemini generateContent answer as an OpenAI Chat Completions object. */
export function geminiToOpenAI(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0;
  const output = typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: text }, finish_reason: openAIFinishOfGemini(candidate.finishReason), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. */
export function geminiToAnthropic(body: Json, model: string): Json {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = typeof usage.promptTokenCount === 'number' ? usage.promptTokenCount : 0;
  const output = typeof usage.candidatesTokenCount === 'number' ? usage.candidatesTokenCount : 0;
  return {
    id: 'msg_gemini',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text }],
    stop_reason: anthropicStopOfGemini(candidate.finishReason),
    stop_sequence: null,
    usage: { input_tokens: input, output_tokens: output },
  };
}

/** An Anthropic Messages answer as an OpenAI Chat Completions object. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = textOf(content);
  const calls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    calls.push({
      id: typeof block.id === 'string' ? block.id : 'toolu_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const input = numberOrZero(usage.input_tokens);
  const output = numberOrZero(usage.output_tokens);
  const message: Json = { role: 'assistant', content: text };
  if (calls.length > 0) message.tool_calls = calls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinishOf(body.stop_reason), logprobs: null }],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

/** Parse SSE text into frames: [{event, data}]. */
function parseFrames(text: string): { event: string | null; data: string }[] {
  const out: { event: string | null; data: string }[] = [];
  for (const frame of text.replace(/\r\n/g, '\n').split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (data.length > 0) out.push({ event, data: data.join('\n') });
  }
  return out;
}

async function readStreamText(stream: ReadableStream<Uint8Array>): Promise<string> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let out = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      out += decoder.decode(value, { stream: true });
    }
    out += decoder.decode();
  } finally {
    reader.releaseLock();
  }
  return out;
}

function openAIChunk(model: string, delta: Json, finish: string | null): string {
  return `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
}

function openAIUsageChunk(model: string, input: number, output: number): string {
  return `data: ${JSON.stringify({ id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [], usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output } })}\n\n`;
}

/**
 * An Anthropic SSE stream as OpenAI chat-completion chunks: role chunk, content and
 * tool-call deltas, a finish chunk, a usage chunk, then [DONE].
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const text = await readStreamText(upstream);
  const frames = parseFrames(text);
  let started = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;
  const toolIndexes = new Map<number, number>();
  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [openAIChunk(model, { role: 'assistant', content: '' }, null)];
  };
  for (const frame of frames) {
    if (frame.data.trim() === '') continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
      continue;
    }
    if (frame.event === 'content_block_delta') {
      const delta = isRecord(data.delta) ? data.delta : {};
      const index = typeof data.index === 'number' ? data.index : 0;
      if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
        yield* start();
        yield openAIChunk(model, { content: delta.text }, null);
      } else if (delta.type === 'input_json_delta') {
        let toolIndex = toolIndexes.get(index);
        if (toolIndex === undefined) {
          toolIndex = toolIndexes.size;
          toolIndexes.set(index, toolIndex);
        }
        yield* start();
        yield openAIChunk(model, { tool_calls: [{ index: toolIndex, function: { arguments: typeof delta.partial_json === 'string' ? delta.partial_json : '' } }] }, null);
      }
      continue;
    }
    if (frame.event === 'content_block_start') {
      const block = isRecord(data.content_block) ? data.content_block : {};
      const index = typeof data.index === 'number' ? data.index : 0;
      if (block.type === 'tool_use') {
        let toolIndex = toolIndexes.get(index);
        if (toolIndex === undefined) {
          toolIndex = toolIndexes.size;
          toolIndexes.set(index, toolIndex);
        }
        yield* start();
        yield openAIChunk(model, { tool_calls: [{ index: toolIndex, id: typeof block.id === 'string' ? block.id : '', type: 'function', function: { name: typeof block.name === 'string' ? block.name : '', arguments: '' } }] }, null);
      }
      continue;
    }
    if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') pendingStop = data.delta.stop_reason;
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
      continue;
    }
  }
  yield* start();
  yield openAIChunk(model, {}, openAIFinishOf(pendingStop ?? 'end_turn'));
  yield openAIUsageChunk(model, inputTokens, outputTokens);
  yield 'data: [DONE]\n\n';
}

/**
 * A Gemini SSE stream as OpenAI chat-completion chunks: role chunk, content deltas,
 * a finish chunk, a usage chunk from the final frame, then [DONE].
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const text = await readStreamText(upstream);
  const frames = parseFrames(text);
  let started = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let finish: string | null = null;
  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [openAIChunk(model, { role: 'assistant', content: '' }, null)];
  };
  for (const frame of frames) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string' && part.text !== '') {
        yield* start();
        yield openAIChunk(model, { content: part.text }, null);
      }
    }
    if (typeof candidate.finishReason === 'string') finish = candidate.finishReason;
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  yield* start();
  yield openAIChunk(model, {}, openAIFinishOfGemini(finish ?? 'STOP'));
  yield openAIUsageChunk(model, inputTokens, outputTokens);
  yield 'data: [DONE]\n\n';
}

/**
 * A Gemini SSE stream as the Anthropic event sequence: message_start, one text
 * block's deltas, message_delta with the stop reason and output tokens, message_stop.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const text = await readStreamText(upstream);
  const frames = parseFrames(text);
  let started = false;
  let textOpen = false;
  let outputTokens = 0;
  let finish: string | null = null;
  const start = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', { type: 'message_start', message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } } }),
      sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    ];
  };
  for (const frame of frames) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const candidates = Array.isArray(data.candidates) ? data.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (isRecord(part) && typeof part.text === 'string' && part.text !== '') {
        yield* start();
        textOpen = true;
        yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: part.text } });
      }
    }
    if (typeof candidate.finishReason === 'string') finish = candidate.finishReason;
    if (isRecord(data.usageMetadata)) outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
  }
  yield* start();
  if (textOpen) yield sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  yield sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicStopOfGemini(finish ?? 'STOP'), stop_sequence: null }, usage: { output_tokens: outputTokens } });
  yield sseEvent('message_stop', { type: 'message_stop' });
}
