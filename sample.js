/**
 * Sample from a trained model
 */

import fs from 'node:fs';
import path from 'node:path';
import { encodingForModel } from 'js-tiktoken';

import { GPTConfig, GPT, loadCheckpoint } from './model.js';
import { configureConfig } from './configurator.js';

// -----------------------------------------------------------------------------
const DEFAULTS = {
  init_from: 'resume', // either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
  out_dir: 'out', // ignored if init_from is not 'resume'
  start: "\n", // or "​" or etc. Can also specify a file, use as: "FILE:prompt.txt"
  num_samples: 10, // number of samples to draw
  max_new_tokens: 500, // number of tokens generated in each sample
  temperature: 0.8, // 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
  top_k: 200, // retain only the top_k most likely tokens, clamp others to have 0 probability
  seed: 1337,
  device: 'cpu', // 'cpu'; gpu requires @tensorflow/tfjs-node-gpu
  dtype: 'float32',
  compile: false, // torch.compile is a PyTorch feature; no equivalent here
};
const config = await configureConfig(DEFAULTS);
// -----------------------------------------------------------------------------

if (config.dtype !== 'float32') {
  console.warn(`WARNING: the node backend only supports float32, ignoring --dtype=${config.dtype}`);
}
if (config.compile) {
  console.warn('WARNING: torch.compile has no Node.js equivalent, ignoring --compile=true');
}

// model
let checkpoint = null; // metadata of a resumed checkpoint, if any
let model;
if (config.init_from === 'resume') {
  // init from a model saved in a specific directory
  const ckptDir = config.out_dir;
  const ckptJson = path.join(ckptDir, 'ckpt.json');
  if (!fs.existsSync(ckptJson)) {
    throw new Error(`file not found: ${ckptJson}`);
  }
  checkpoint = JSON.parse(fs.readFileSync(ckptJson, 'utf8'));
  const gptconf = new GPTConfig(checkpoint.model_args);
  model = new GPT(gptconf);
  loadCheckpoint(ckptDir, model);
} else if (config.init_from.startsWith('gpt2')) {
  // init from a given GPT-2 model
  model = await GPT.fromPretrained(config.init_from, { dropout: 0.0 });
}
// note: the node backend has no dropout in eval mode (training=false) and no
// torch.compile/autocast equivalents, so model.eval() and `ctx` are implicit

// look for the meta file in case it is available in the dataset folder
let loadMeta = false;
let metaPath = null;
if (
  config.init_from === 'resume' &&
  checkpoint &&
  checkpoint.config &&
  'dataset' in checkpoint.config // older checkpoints might not have these...
) {
  metaPath = path.join('data', checkpoint.config.dataset, 'meta.json');
  loadMeta = fs.existsSync(metaPath);
}
let encode, decode;
if (loadMeta) {
  console.log(`Loading meta from ${metaPath}...`);
  const meta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
  // TODO want to make this more general to arbitrary encoder/decoder schemes
  const itos = meta.itos; // { "0": "\n", ... } (int -> char)
  const stoi = meta.stoi;
  encode = (s) => [...s].map((c) => stoi[c]);
  decode = (l) => l.map((i) => itos[String(i)]).join('');
} else {
  // ok let's assume gpt-2 encodings by default
  console.log('No meta.json found, assuming GPT-2 encodings...');
  const enc = encodingForModel('gpt2');
  encode = (s) => enc.encode(s);
  decode = (l) => enc.decode(l);
}

// encode the beginning of the prompt
let start = config.start;
if (start.startsWith('FILE:')) {
  start = fs.readFileSync(start.slice(5), 'utf8');
}
const start_ids = encode(start);
const x = [start_ids];

// run generation
for (let k = 0; k < config.num_samples; k++) {
  const y = model.generate(x, config.max_new_tokens, {
    temperature: config.temperature,
    top_k: config.top_k,
  });
  console.log(decode(y[0]));
  console.log('---------------');
}
