/**
 * AgentKit runtime — load agent manifests, run the think–act loop, persist
 * sessions. The loop is brain-agnostic: local nanoGPT brains and scripted
 * mock brains (and any future brain type) share it.
 */

import fs from 'node:fs';
import path from 'node:path';

import { brainRespond } from './brains.js';
import { executeCall, listTools, parseActLines } from './tools.js';

export const AGENTS_DIR = 'agents';
export const SESSIONS_DIR = '.agentkit';

// ---------------------------------------------------------------------------
// manifests

function validateManifest(manifest) {
  for (const key of ['name', 'description', 'brain']) {
    if (!(key in manifest)) throw new Error(`agent manifest missing "${key}"`);
  }
  if (typeof manifest.brain !== 'object' || manifest.brain.type == null) {
    throw new Error('agent manifest field "brain" must be an object with a "type"');
  }
  manifest.tools ??= [];
  manifest.system ??= `You are ${manifest.name}. ${manifest.description}`;
  manifest.max_turns ??= 4;
  return manifest;
}

export function agentPath(name) {
  return path.join(AGENTS_DIR, `${name}.agent.json`);
}

export async function loadAgent(name) {
  const p = agentPath(name);
  if (!fs.existsSync(p)) {
    throw new Error(`agent not found: ${p} (create one with: agentkit create ${name})`);
  }
  return validateManifest(JSON.parse(fs.readFileSync(p, 'utf8')));
}

export function saveAgent(manifest) {
  fs.mkdirSync(AGENTS_DIR, { recursive: true });
  fs.writeFileSync(agentPath(manifest.name), JSON.stringify(manifest, null, 2) + '\n');
}

export function listAgents() {
  if (!fs.existsSync(AGENTS_DIR)) return [];
  return fs
    .readdirSync(AGENTS_DIR)
    .filter((f) => f.endsWith('.agent.json'))
    .map((f) => JSON.parse(fs.readFileSync(path.join(AGENTS_DIR, f), 'utf8')));
}

// ---------------------------------------------------------------------------
// sessions

export function sessionFile(agentName, sessionId) {
  return path.join(SESSIONS_DIR, agentName, `${sessionId}.jsonl`);
}

export function appendMessages(agentName, sessionId, messages) {
  const file = sessionFile(agentName, sessionId);
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.appendFileSync(file, messages.map((m) => JSON.stringify(m)).join('\n') + '\n');
}

export function readSession(agentName, sessionId) {
  const file = sessionFile(agentName, sessionId);
  if (!fs.existsSync(file)) return [];
  return fs
    .readFileSync(file, 'utf8')
    .split('\n')
    .filter(Boolean)
    .map((l) => JSON.parse(l));
}

export function newSessionId() {
  return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19) + '-' +
    Math.random().toString(36).slice(2, 6);
}

// ---------------------------------------------------------------------------
// the agent loop

/**
 * Run an agent until it produces a final answer (or turns out).
 * Yields progress records so CLIs/UIs can stream what happens:
 *   { type: 'turn', n } | { type: 'assistant', text } | { type: 'tool', name, args }
 *   { type: 'observation', name, text } | { type: 'final', text }
 */
export async function* runAgent(manifest, { task, sessionId = null, onTranscript = false } = {}) {
  const messages = [];
  if (task != null) messages.push({ role: 'user', content: task });

  let observation = null;
  const transcript = [];
  const record = (m) => {
    transcript.push(m);
    if (onTranscript && sessionId) appendMessages(manifest.name, sessionId, [m]);
  };

  for (let turn = 0; turn < manifest.max_turns; turn++) {
    yield { type: 'turn', n: turn + 1 };
    const text = await brainRespond(manifest.brain, {
      system: manifest.system,
      messages: task == null ? messages.slice() : messages,
      observation,
    });
    yield { type: 'assistant', text };
    record({ role: 'assistant', content: text });

    const { calls, remainder } = parseActLines(text);
    if (calls.length === 0 || turn === manifest.max_turns - 1) {
      const final = remainder.trim() || text.trim();
      yield { type: 'final', text: final };
      record({ role: 'final', content: final });
      if (sessionId && !onTranscript) appendMessages(manifest.name, sessionId, transcript);
      return;
    }

    // execute tool calls, then let the brain react to the observation
    for (const call of calls) {
      yield { type: 'tool', name: call.tool, args: call.args };
      const result = executeCall(call);
      yield { type: 'observation', name: call.tool, text: result };
      observation = `${call.tool}(${call.args.join(', ')}) -> ${result.replace(/\n+/g, ' ').slice(0, 400)}`;
    }
    messages.push({ role: 'assistant', content: text });
  }
  yield { type: 'final', text: transcript.at(-1)?.content ?? '' };
  if (sessionId) appendMessages(manifest.name, sessionId, transcript);
}

/** convenience: run and return only the final text */
export async function runAgentFinal(manifest, opts = {}) {
  let final = '';
  for await (const evt of runAgent(manifest, opts)) {
    if (evt.type === 'final') final = evt.text;
  }
  return final;
}

/** list of tool descriptors restricted to the agent's tools list */
export function agentToolset(manifest) {
  return listTools(manifest.tools);
}
