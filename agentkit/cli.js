#!/usr/bin/env node
/**
 * AgentKit — an AI agent builder for locally-trained brains.
 *
 * Agents are JSON manifests in agents/<name>.agent.json. Each manifest points
 * at a "brain" (by default a locally-trained nanoGPT checkpoint from this
 * repo's train.js pipeline — no external LLM API) plus a list of tools.
 *
 * Usage:
 *   agentkit create <name> [--template starter|coder|researcher]
 *   agentkit list
 *   agentkit run <name> --task "do a thing" [--max-turns N]
 *   agentkit chat <name>                   (interactive)
 *   agentkit discord <name>                (run as a Discord bot outside)
 *   agentkit tools
 *   agentkit show <name>
 */

import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';

import {
  AGENTS_DIR,
  loadAgent,
  listAgents,
  newSessionId,
  readSession,
  saveAgent,
  runAgent,
} from './agent.js';
import {
  describeTool,
  listTools,
  parseActLines,
} from './tools.js';

function argValue(flag) {
  const i = process.argv.indexOf(flag);
  return i === -1 ? null : process.argv[i + 1];
}

function hasFlag(flag) {
  return process.argv.includes(flag);
}

const [command, ...rest] = process.argv.slice(2);

async function main() {
  switch (command) {
    case 'create':
      return cmdCreate();
    case 'list':
      return cmdList();
    case 'chat':
      return cmdChat(rest[0]);
    case 'discord':
      return cmdDiscord(rest[0]);
    case 'run':
      return cmdRun();
    case 'show':
      return cmdShow(rest[0]);
    case 'tools':
      return cmdTools();
    default:
      console.log(USAGE);
      return command == null ? 0 : 1;
  }
}

const USAGE = `AgentKit — build AI agents on locally-trained brains (no LLM API needed)

Commands:
  agentkit create <name> [--template starter|coder|researcher]
  agentkit list
  agentkit show <name>
  agentkit run <name> --task "..." [--max-turns N]
  agentkit chat <name>
  agentkit discord <name>                (run as a Discord bot)
  agentkit tools

Agents live in ${AGENTS_DIR}/<name>.agent.json; sessions in .agentkit/.`;

const TEMPLATES = {
  starter: {
    description: 'a simple conversational agent',
    brain: { type: 'nano', checkpoint: 'out-demo', dataset: 'tutorial', temperature: 0.3, top_k: 5, max_new_tokens: 48 },
    tools: [],
  },
  coder: {
    description: 'writes code and runs shell commands',
    brain: { type: 'nano', checkpoint: 'out-demo', dataset: 'tutorial', temperature: 0.3, top_k: 5, max_new_tokens: 48 },
    tools: ['calc', 'read_file', 'write_file', 'exec', 'list_dir'],
  },
  researcher: {
    description: 'reads files and fetches web pages to answer questions',
    brain: { type: 'nano', checkpoint: 'out-demo', dataset: 'tutorial', temperature: 0.3, top_k: 5, max_new_tokens: 48 },
    tools: ['calc', 'read_file', 'list_dir', 'http_get'],
  },
};

async function cmdCreate() {
  const name = rest[0];
  if (!name || !/^[A-Za-z0-9_-]+$/.test(name)) {
    throw new Error('usage: agentkit create <name>  (letters, digits, -, _)');
  }
  let template = argValue('--template');
  if (!template) {
    const sel = readline.createInterface({ input, output });
    const answer = await sel.question(
      `Choose a template [${Object.keys(TEMPLATES).join('/')}] (starter): `
    );
    sel.close();
    template = (answer.trim() || 'starter').toLowerCase();
  }
  if (!TEMPLATES[template]) throw new Error(`unknown template "${template}"`);
  const t = TEMPLATES[template];
  const manifest = {
    name,
    description: t.description,
    brain: t.brain,
    tools: t.tools,
    system: `You are ${name}, ${t.description}. When useful, take actions by emitting a single line like "act: toolName(arg)" using only your available tools, then wait for an [observed] line before continuing.`,
    max_turns: 4,
  };
  saveAgent(manifest);
  console.log(`created ${AGENTS_DIR}/${name}.agent.json (template: ${template})`);
  console.log(`brain: locally-trained nanoGPT checkpoint "${manifest.brain.checkpoint}"`);
  console.log('\nEdit the manifest to tweak the prompt/tools, then run:');
  console.log(`  agentkit run ${name} --task "..."`);
}

function cmdList() {
  const agents = listAgents();
  if (agents.length === 0) {
    console.log(`no agents yet — create one: agentkit create my-agent`);
    return;
  }
  for (const a of agents) {
    console.log(
      `${a.name.padEnd(16)} brain=${a.brain.type} tools=[${a.tools.join(', ')}] turns=${a.max_turns}\n${' '.repeat(16)}${a.description}`
    );
  }
}

async function cmdShow(name) {
  const manifest = await loadAgent(name);
  console.log(JSON.stringify(manifest, null, 2));
}

function cmdTools() {
  for (const t of listTools()) console.log(describeTool(t));
}

async function cmdDiscord(name) {
  const manifest = await requireAgent(name);
  if (!manifest) return;
  const token = process.env.DISCORD_BOT_TOKEN;
  if (!token) {
    console.error('DISCORD_BOT_TOKEN is not set (Settings → Environment). Create an app at');
    console.error('https://discord.com/developers/applications, enable the MESSAGE CONTENT intent,');
    console.error('make a bot token and set it, then: agentkit discord ' + name);
    return 1;
  }
  console.log(`starting ${name} as a Discord bot (brain: ${manifest.brain.type}, tools: [${manifest.tools.join(', ')}])`);
  const { startDiscordAgent } = await import('./discord.js');
  await startDiscordAgent(manifest);
}

async function requireAgent(name) {
  try {
    return await loadAgent(name);
  } catch (e) {
    console.error(e.message);
    process.exit(1);
  }
}

async function cmdRun() {
  const name = rest[0];
  const task = argValue('--task');
  if (!name || !task) {
    throw new Error('usage: agentkit run <name> --task "..."');
  }
  const manifest = await requireAgent(name);
  const maxTurnsOverride = argValue('--max-turns');
  if (maxTurnsOverride) manifest.max_turns = parseInt(maxTurnsOverride, 10);
  const sessionId = hasFlag('--no-save') ? null : newSessionId();
  if (sessionId) console.log(`session: ${sessionId}`);

  for await (const evt of runAgent(manifest, { task, sessionId })) {
    renderEvent(evt);
  }
}

function renderEvent(evt) {
  switch (evt.type) {
    case 'turn':
      console.log(`\n── turn ${evt.n}`);
      break;
    case 'assistant':
      console.log(`[assistant] ${evt.text.trim()}`);
      break;
    case 'tool':
      console.log(`[tool] ${evt.name}(${evt.args.join(', ')})`);
      break;
    case 'observation': {
      const short = evt.text.replace(/\s+/g, ' ').slice(0, 300);
      console.log(`[observed] ${short}${evt.text.length > 300 ? '…' : ''}`);
      break;
    }
    case 'final':
      console.log(`\n=== final answer ===\n${evt.text.trim()}\n`);
      break;
  }
}

async function cmdChat(name) {
  const manifest = await requireAgent(name);
  if (!manifest) return;
  const sessionId = newSessionId();
  fs.mkdirSync(path.dirname(sessionPath(manifest.name, sessionId)), { recursive: true });
  console.log(`chatting with "${manifest.name}" (${manifest.brain.type} brain) — blank line to exit`);
  const rl = readline.createInterface({ input, output });
  while (true) {
    const task = await rl.question('\nyou> ');
    if (!task.trim()) break;
    console.log(`\n${manifest.name}>`);
    for await (const evt of runAgent(manifest, { task, sessionId })) {
      renderEvent(evt);
    }
  }
  rl.close();
  console.log(`session saved: ${sessionPath(manifest.name, sessionId)}`);
}

function sessionPath(agentName, sessionId) {
  return `.agentkit/${agentName}/${sessionId}.jsonl`;
}

main()
  .then((code) => process.exit(code ?? 0))
  .catch((e) => {
    console.error(e.message);
    process.exit(1);
  });
