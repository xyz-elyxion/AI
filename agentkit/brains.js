/**
 * AgentKit brains — the "model" behind an agent.
 *
 * - `nano`   : a locally-trained nanoGPT checkpoint (this repo's model.js /
 *              train.js pipeline, running on @tensorflow/tfjs-node). No
 *              external LLM API involved: the model genuinely runs here.
 * - `mock`   : a scripted brain used for testing the full agent loop without
 *              any network.
 *
 * Both brains share a simple interface:
 *   respond({ messages, manifest }) -> string (assistant text, may contain
 *   `act:` tool-call lines for the local brain).
 */

import fs from 'node:fs';
import path from 'node:path';
import { encodingForModel } from 'js-tiktoken';
import '../node-compat.js';

let lazyTf = null;
async function getTf() {
  if (!lazyTf) lazyTf = await import('@tensorflow/tfjs-node');
  return lazyTf;
}

// ---- nano brain (this repo's own trained GPT) -------------------------------

let cachedBrain = null;

/**
 * Load a nanoGPT checkpoint once and reuse it across agent turns.
 * checkpoint: path to a training out_dir containing ckpt.json + ckpt.bin
 */
export function loadNanoBrain(checkpointDir) {
  const key = path.resolve(checkpointDir);
  if (cachedBrain && cachedBrain.key === key) return cachedBrain;
  const ckptJson = path.join(key, 'ckpt.json');
  if (!fs.existsSync(ckptJson)) {
    throw new Error(
      `no checkpoint at ${key} (missing ckpt.json); train one with train.js first, ` +
      `e.g. node train.js config/agent_brain.js --out_dir=${checkpointDir}`
    );
  }
  return (async () => {
    const tf = await getTf();
    const { GPTConfig, GPT, loadCheckpoint } = await import('../model.js');
    const meta = JSON.parse(fs.readFileSync(ckptJson, 'utf8'));
    const model = new GPT(new GPTConfig(meta.model_args));
    loadCheckpoint(key, model);
    cachedBrain = { key: checkpointDir, model, config: meta.model_args };
    return cachedBrain;
  })();
}

/** how the brain was trained: char-level (meta.json in dataset dir) or BPE */
function resolveTokenizerFor(checkpointDir, modelArgs) {
  // nanoGPT train.js records vocab_size; char-level models additionally have a
  // data/<dataset>/meta.json referenced through the checkpoint config if saved
  // there at training time. For simplicity, the manifest can pass dataset.
  return null;
}

/**
 * Compose the raw prompt for the (tiny) local brain, staying close to its
 * training distribution: plain text, no invented role markers (a char-level
 * model trained on play scripts never saw "[assistant]" and gets lost). We
 * append the observation if any and clip to the model's block size.
 */
function composeLocalPrompt(brain, { system, messages, observation }) {
  // NOTE: we deliberately ignore `system` for tiny char-level brains — it is
  // out-of-distribution text the model never saw, and it eats the tiny context.
  const dialog = messages.map((m) => m.content).join('\n');
  const observationText = observation != null ? `${observation}\n` : '';
  const clip = Math.max(24, brain.config.block_size - 4);
  const tail = dialog.length > clip ? dialog.slice(-clip) : dialog;
  return `${tail}\n${observationText}`;
}

export async function nanoRespond({ checkpoint, dataset, system, messages, observation, temperature = 1.0, top_k = null, max_new_tokens = 120 }) {
  const brain = await loadNanoBrain(checkpoint);
  const tf = await getTf();
  // keep generation inside the context the model was trained with: longer
  // sequences are out-of-distribution and devolve quickly
  max_new_tokens = Math.max(8, Math.min(max_new_tokens, brain.config.block_size - 8));
  const prompt = composeLocalPrompt(brain, { system, messages, observation });
  // encode: use the dataset tokenizer when available (char-level), else GPT-2 BPE
  const ids = encodeFor(prompt, checkpoint, dataset);
  const out = brain.model.generate([ids], max_new_tokens, { temperature, top_k });
  return decodeFor(out[0].slice(ids.length), checkpoint, dataset);
}

function metaPathFor(checkpoint, dataset) {
  // dataset prepare.js scripts write data/<dataset>/meta.json
  const candidates = dataset
    ? [path.join('data', dataset, 'meta.json')]
    : [];
  // also try sibling-to-checkpoint locations
  candidates.push(
    path.join(path.dirname(checkpoint), 'meta.json'),
    path.join(path.dirname(checkpoint), '..', 'meta.json')
  );
  for (const p of candidates) {
    if (p && fs.existsSync(p)) return p;
  }
  return null;
}

function encodeFor(text, checkpoint, dataset) {
  const metaFile = metaPathFor(checkpoint, dataset);
  if (metaFile) {
    const meta = JSON.parse(fs.readFileSync(metaFile, 'utf8'));
    return [...text].map((c) => meta.stoi[c]).filter((v) => v !== undefined);
  }
  const enc = encodingForModel('gpt2');
  return enc.encode(text);
}

function decodeFor(ids, checkpoint, dataset) {
  const metaFile = metaPathFor(checkpoint, dataset);
  if (metaFile) {
    const meta = JSON.parse(fs.readFileSync(metaFile, 'utf8'));
    return ids.map((i) => meta.itos[String(i)] ?? '').join('');
  }
  const enc = encodingForModel('gpt2');
  return enc.decode(ids);
}

// ---- mock brain -------------------------------------------------------------

export function mockRespond({ script, indexRef }) {
  indexRef.i ??= 0;
  if (indexRef.i >= script.length) return '(mock brain exhausted its script)';
  return script[indexRef.i++];
}

// ---- dispatcher -------------------------------------------------------------

export async function brainRespond(brainConfig, { system, messages, observation }) {
  const brainType = brainConfig.type;
  if (brainType === 'nano') {
    return nanoRespond({ system, messages, observation, ...brainConfig });
  }
  if (brainType === 'mock') {
    return mockRespond({ script: brainConfig.script, indexRef: brainConfig });
  }
  throw new Error(`unknown brain type: ${brainType} (known: nano, mock)`);
}
