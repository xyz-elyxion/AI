/**
 * AgentKit tests using the scripted mock brain — verifies the full
 * think–act loop end to end without any network or trained checkpoint.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs'; 
import path from 'node:path';
import os from 'node:os';

import { parseActLines, executeCall } from '../agentkit/tools.js';
import { runAgentFinal, runAgent } from '../agentkit/agent.js';

test('act protocol parsing', () => {
  const text = `I should do math first.
act: calc(2+2*10)
and maybe read something
act: list_dir("data")`;
  const { calls, remainder } = parseActLines(text);
  assert.equal(calls.length, 2);
  assert.deepEqual(calls[0], { tool: 'calc', args: ['2+2*10'] });
  assert.deepEqual(calls[1], { tool: 'list_dir', args: ['data'] });
  assert.ok(remainder.includes('I should do math first.'));
  assert.ok(!remainder.includes('act:'));
});

test('calc tool', () => {
  assert.equal(executeCall({ tool: 'calc', args: ['5^2'] }), '25');
  assert.equal(executeCall({ tool: 'calc', args: ['5'] }), '5');
  assert.equal(executeCall({ tool: 'calc', args: ['343'] }), '343');
});

test('unknown tool fails soft', () => {
  const out = executeCall({ tool: 'does_not_exist', args: [] });
  assert.ok(out.startsWith('unknown tool'));
});

test('full agent loop with tool round-trip (mock brain)', async () => {
  const manifest = {
    name: 'mockcalc',
    description: 'test agent',
    brain: {
      type: 'mock',
      script: [
        'Let me compute that.\nact: calc(343)',
        'The answer is 343.',
      ],
    },
    tools: ['calc'],
    system: 'demo',
    max_turns: 4,
  };
  const events = [];
  for await (const evt of runAgent(manifest, { task: 'What is 343?' })) {
    events.push(evt);
  }
  const types = events.map((e) => e.type);
  assert.deepEqual(types, ['turn', 'assistant', 'tool', 'observation', 'turn', 'assistant', 'final']);
  const final = events.findLast((e) => e.type === 'final');
  assert.equal(final.text, 'The answer is 343.');
  const toolEvt = events.find((e) => e.type === 'tool');
  assert.equal(toolEvt.name, 'calc');
});

test('unknown brain type errors clearly', async () => {
  const manifest = {
    name: 'x',
    description: 'x',
    brain: { type: 'nope' },
    tools: [],
    max_turns: 1,
  };
  await assert.rejects(runAgentFinal(manifest, { task: 'hi' }), /unknown brain type/);
});

test('agent accel helper: exec tool safe truncation is fine', () => {
  const out = executeCall({ tool: 'exec', args: ['echo hello'] });
  assert.ok(out.includes('hello'));
});

test('write_file tool', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentkit-'));
  const p = path.join(dir, 'sub', 'out.txt');
  const out = executeCall({ tool: 'write_file', args: [p, 'written by agentkit'] });
  assert.equal(out, `wrote ${p}`);
  assert.equal(fs.readFileSync(p, 'utf8'), 'written by agentkit');
  fs.rmSync(dir, { recursive: true, force: true });
});
