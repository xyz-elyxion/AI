/**
 * Generates `corpus.txt` for the AgentKit tutorial brain: a small set of
 * synthetic agent transcripts that demonstrate the act protocol (think →
 * `act:` tool call → `[observed]` result → final answer).
 *
 * Re-run with: node data/tutorial/make-corpus.mjs   (overwrites corpus.txt)
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

// mulberry32-ish seeded PRNG for reproducible output
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rnd = mulberry32(20261023);
const pick = (arr) => arr[Math.floor(rnd() * arr.length)];

const NAMES = ['Nova', 'Atlas', 'Quill', 'Pixie'];
const GREETING_TASKS = ['Hello, who are you?', 'Introduce yourself.', 'Say hi to the team.'];
const GREETING_FINALS = [
  (n) => `Hello! I am ${n}, your on-device agent.`,
  (n) => `Greetings. ${n} here, running locally on this machine.`,
  (n) => `I am ${n}, built to run tools right here on this computer.`,
];

const CALC_TASKS = [];
for (let i = 0; i < 24; i++) {
  const a = 2 + Math.floor(rnd() * 97);
  const b = 2 + Math.floor(rnd() * 97);
  CALC_TASKS.push({ a, b });
}

function calcTranscripts(name) {
  const out = CALC_TASKS.map(({ a, b }) => {
    const product = a * b;
    const sum = a + b;
    const ops = [
      { task: `What is ${a}*${b}?`, expr: `${a}*${b}`, answer: product },
      { task: `Compute ${a}+${b} for me.`, expr: `${a}+${b}`, answer: sum },
      { task: `Multiply ${a} by ${b}.`, expr: `${a}*${b}`, answer: product },
      { task: `Add ${a} and ${b}.`, expr: `${a}+${b}`, answer: sum },
    ];
    const { task, expr, answer } = pick(ops);
    return [
      `[user] ${task}`,
      `[${name}] I will use the calc tool.`,
      `act: calc(${expr})`,
      `[observed] ${answer}`,
      `[${name}] The result is ${answer}.`,
    ].join('\n');
  });
  return out;
}

function readTranscripts(name) {
  const files = ['README.md', 'package.json', 'LICENSE'];
  const out = [];
  for (const f of files) {
    out.push([
      `[user] Show me the file ${f}.`,
      `[${name}] Reading it with the read_file tool.`,
      `act: read_file("${f}")`,
      `[observed] ...file contents shown...`,
      `[${name}] I read ${f}; its contents are shown above by the observed line.`,
    ].join('\n'));
  }
  return out;
}

function listTranscripts(name) {
  const dirs = ['data', 'agentkit', 'config', '.'];
  const out = dirs.map((d) => [
    `[user] List the directory ${d === '.' ? 'this repo' : d}.`,
    `[${name}] I will list it.`,
    `act: list_dir("${d === '.' ? '.' : d + '/'}")`,
    `[observed] ...directory listing...`,
    `[${name}] Done: the listing is in the observed line.`,
  ].join('\n'));
  return out;
}

function execTranscripts(name) {
  const cmds = [
    { cmd: 'echo hello', obs: 'hello', final: 'The command printed: hello.' },
    { cmd: 'pwd', obs: '/this/project/folder', final: 'We are inside the project folder.' },
    { cmd: 'date', obs: '2026-10-23', final: 'It is 2026-10-23 by the system clock.' },
  ];
  return cmds.map(({ cmd, obs, final }) => [
    `[user] Run the command: ${cmd}`,
    `[${name}] Running it with exec.`,
    `act: exec("${cmd}")`,
    `[observed] ${obs}`,
    `[${name}] ${final}`,
  ].join('\n'));
}

function plainTranscripts(name) {
  const states = [];
  for (let i = 0; i < 30; i++) {
    states.push([
      `[user] ${pick(GREETING_TASKS)}`,
      `[${name}] ${pick(GREETING_FINALS)(name)}`,
    ].join('\n'));
  }
  return states;
}

const all = [];
for (const name of NAMES) {
  all.push(
    ...calcTranscripts(name),
    ...readTranscripts(name),
    ...listTranscripts(name),
    ...execTranscripts(name),
    ...plainTranscripts(name)
  );
}

// interleave deterministically
for (let i = all.length - 1; i > 0; i--) {
  const j = Math.floor(rnd() * (i + 1));
  [all[i], all[j]] = [all[j], all[i]];
}

const corpus = all.join('\n\n') + '\n';
fs.writeFileSync(path.join(here, 'corpus.txt'), corpus);
console.log(`wrote corpus.txt with ${all.length} transcripts, ${corpus.length.toLocaleString()} chars`);
