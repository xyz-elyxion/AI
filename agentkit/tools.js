/**
 * AgentKit built-in tools.
 *
 * Local (nanoGPT) brains steer tools with the `act` protocol: a generated line
 * of the form
 *
 *     act: toolName(arg, arg, ...)
 *
 * Ask your agent to follow it via its system prompt — and for real reliability,
 * finetune the brain on transcripts that demonstrate it (see README).
 */

import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const registry = new Map();

export function registerTool(name, { description, parameters, run }) {
  registry.set(name, { name, description, parameters, run });
}

export function getTool(name) {
  return registry.get(name);
}

export function listTools(names = null) {
  const all = [...registry.values()];
  if (!names) return all;
  return all.filter((t) => names.includes(t.name));
}

/** human-friendly listing for the CLI */
export function describeTool(t) {
  return `${t.name} — ${t.description}\n    args: ${t.parameters || '(none)'}`;
}

/**
 * Parse `act:` lines out of generated text. Returns { calls, remainder } where
 * calls is [{tool, args}] and remainder is the text without the act lines.
 */
export function parseActLines(text) {
  const calls = [];
  const lines = String(text).split('\n').filter((line) => {
    const m = line.match(/^\s*act:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(?\s*(.*?)\s*\)?\s*$/);
    if (!m) return true;
    calls.push({
      tool: m[1],
      args: parseArgsString(m[2]),
    });
    return false;
  });
  return { calls, remainder: lines.join('\n') };
}

function parseArgsString(s) {
  // lightweight arg splitter: supports quoted strings, numbers, bare words
  const args = [];
  const re = /"([^"]*)"|'([^']*)'|([^\s,]+)|,/g;
  let m;
  let lastEnd = 0;
  while ((m = re.exec(s)) !== null) {
    if (m[0] === ',') continue;
    const raw = m[1] ?? m[2] ?? m[3];
    if (raw === undefined || raw === '') continue;
    args.push(coerce(raw));
    lastEnd = re.lastIndex;
  }
  lastEnd; // no-op
  return args;
}

function coerce(s) {
  if (/^-?\d+(\.\d+)?$/.test(s)) return Number(s);
  return s;
}

// ---- built-in tools -------------------------------------------------------

const MAX_OUTPUT = 8000; // keep observations small enough for tiny context sizes

registerTool('calc', {
  description: 'evaluate an arithmetic expression, e.g. calc(2+2*10)',
  parameters: ['expr'],
  run([expr]) {
    expr = String(expr);
    if (!/^[\d\s+\-*/().,%^eE]*$/.test(expr)) {
      throw new Error('calc: expression contains disallowed characters');
    }
    // translate ^ to ** for exponentiation before evaluating
    const js = expr.replace(/\^/g, '**');
    const value = Function(`"use strict"; return (${js});`)();
    return String(value);
  },
});

registerTool('read_file', {
  description: 'read a text file, e.g. read_file("README.md")',
  parameters: ['path'],
  run([p]) {
    const text = fs.readFileSync(String(p), 'utf8');
    return text.length > MAX_OUTPUT ? text.slice(0, MAX_OUTPUT) + '…[truncated]' : text;
  },
});

registerTool('write_file', {
  description: 'write content to a text file, e.g. write_file("notes.txt", "hello")',
  parameters: ['path', 'content'],
  run([p, content]) {
    fs.mkdirSync(path.dirname(String(p)), { recursive: true });
    fs.writeFileSync(String(p), content == null ? '' : String(content));
    return `wrote ${String(p)}`;
  },
});

registerTool('list_dir', {
  description: 'list files in a directory, e.g. list_dir("data")',
  parameters: ['path'],
  run([p]) {
    const entries = fs.readdirSync(p ? String(p) : '.', { withFileTypes: true });
    const listing = entries
      .map((e) => (e.isDirectory() ? `${e.name}/` : e.name))
      .join('\n');
    return listing.length > MAX_OUTPUT ? listing.slice(0, MAX_OUTPUT) + '…[truncated]' : listing;
  },
});

registerTool('exec', {
  description: 'run a shell command and return stdout, e.g. exec("ls -la")',
  parameters: ['command'],
  run([command]) {
    const out = execFileSync('sh', ['-c', String(command)], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
      timeout: 30_000,
      maxBuffer: 1024 * 1024,
    });
    return (out || '(no output)').length > MAX_OUTPUT
      ? out.slice(0, MAX_OUTPUT) + '…[truncated]'
      : out || '(no output)';
  },
});

registerTool('http_get', {
  description: 'fetch a URL and return the first ~20KB of text, e.g. http_get("https://example.com")',
  parameters: ['url'],
  run([url]) {
    const res = await_fetch(String(url));
    return res;
  },
});

// small sync-style wrapper using child spawn of a tiny inline node script so a
// tool "run" stays synchronous for the agent loop
function await_fetch(url) {
  const script = `fetch(process.argv[1]).then(async r => { const t = await r.text(); console.log(HTTP_STATUS + r.status); console.log(t.slice(0, ${MAX_OUTPUT})); }).catch(e => { console.error(e.message); process.exit(1); });`;
  try {
    return execFileSync(
      'node',
      ['-e', script, url],
      { encoding: 'utf8', timeout: 30_000, maxBuffer: 1024 * 1024, env: { ...process.env, HTTP_STATUS: '' } }
    ).replace(/^\n?/, '');
  } catch (e) {
    return `http_get failed: ${e.message.split('\n')[0]}`;
  }
}

/** execute one parsed call; returns readable text (never throws) */
export function executeCall(call) {
  const tool = getTool(call.tool);
  if (!tool) return `unknown tool: ${call.tool} (available: ${[...registry.keys()].join(', ')})`;
  try {
    return String(tool.run(call.args));
  } catch (e) {
    return `tool ${call.tool} failed: ${e.message.split('\n')[0]}`;
  }
}
