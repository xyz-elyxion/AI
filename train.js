/**
 * This training script can be run both on a single cpu/gpu in debug mode, and
 * also for larger training runs. (DDP/torchrun from the Python version has no
 * Node.js equivalent; this port always trains in a single process.)
 *
 * To run on a single device, example:
 * $ node train.js --batch_size=32
 */

import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import * as tf from '@tensorflow/tfjs-node';

import { GPTConfig, GPT, saveCheckpoint, loadCheckpoint } from './model.js';
import { configureConfig } from './configurator.js';
import { makeGetBatch } from './dataloader.js';

// -----------------------------------------------------------------------------
// default config values designed to train a gpt2 (124M) on OpenWebText
// I/O
const DEFAULTS = {
  out_dir: 'out',
  eval_interval: 2000,
  log_interval: 1,
  eval_iters: 200,
  eval_only: false, // if True, script exits right after the first eval
  always_save_checkpoint: true, // if True, always save a checkpoint after each eval
  init_from: 'scratch', // 'scratch' or 'resume' or 'gpt2*'
  // wandb logging
  wandb_log: false, // disabled by default (no winner WandB adapter in node; kept for parity)
  wandb_project: 'owt',
  wandb_run_name: 'gpt2', // 'run' + str(time.time())
  // data
  dataset: 'openwebtext',
  gradient_accumulation_steps: 5 * 8, // used to simulate larger batch sizes
  batch_size: 12, // if gradient_accumulation_steps > 1, this is the micro-batch size
  block_size: 1024,
  // model
  n_layer: 12,
  n_head: 12,
  n_embd: 768,
  dropout: 0.0, // for pretraining 0 is good, for finetuning try 0.1+
  bias: false, // do we use bias inside LayerNorm and Linear layers?
  // adamw optimizer
  learning_rate: 6e-4, // max learning rate
  max_iters: 600000, // total number of training iterations
  weight_decay: 1e-1,
  beta1: 0.9,
  beta2: 0.95,
  grad_clip: 1.0, // clip gradients at this value, or disable if == 0.0
  // learning rate decay settings
  decay_lr: true, // whether to decay the learning rate
  warmup_iters: 2000, // how many steps to warm up for
  lr_decay_iters: 600000, // should be ~= max_iters per Chinchilla
  min_lr: 6e-5, // minimum learning rate, should be ~= learning_rate/10 per Chinchilla
  // DDP settings
  backend: 'nccl', // not applicable in the Node.js port
  // system
  device: 'cpu', // 'cpu'; gpu requires @tensorflow/tfjs-node-gpu
  dtype: 'float32', // 'float32' (bfloat16/float16 autocast not supported here)
  compile: false, // torch.compile is a PyTorch feature; no equivalent
};
// -----------------------------------------------------------------------------
const config = await configureConfig(DEFAULTS);

// -----------------------------------------------------------------------------
// various inits, derived attributes, I/O setup
if (process.env.RANK != null) {
  console.warn('WARNING: DDP/torchrun are not supported in Node; training runs in a single process.');
}
const master_process = true;
const world_size = 1;
const tokens_per_iter = config.gradient_accumulation_steps * world_size * config.batch_size * config.block_size;
console.log(`tokens per iteration will be: ${tokens_per_iter.toLocaleString()}`);

if (master_process) {
  fs.mkdirSync(config.out_dir, { recursive: true });
}

// torch.manual_seed(1337): our weight init uses a fixed seed already; batch
// sampling uses Math.random (the seed only matters for reproducibility of the
// train.py reference; tfjs dropouts consume backend defaults)

// note: float16 support in the torch version used a GradScaler; the node
// backend is float32-only, so no GradScaler equivalent is needed
if (config.dtype !== 'float32') {
  console.warn(`WARNING: the node backend only supports float32, ignoring --dtype=${config.dtype}`);
}
if (config.compile) {
  console.warn('WARNING: torch.compile has no Node.js equivalent, ignoring --compile=true');
}
if (config.device.startsWith('gpu')) {
  console.log('note: --device=gpu requires @tensorflow/tfjs-node-gpu; currently running on the node backend');
}

let data_dir = fs.existsSync(path.join('data', config.dataset))
  ? path.join('data', config.dataset)
  : config.dataset;
if (!fs.existsSync(path.join(data_dir, 'train.bin'))) {
  throw new Error(
    `cannot find ${path.join(data_dir, 'train.bin')}; run e.g. "node data/${config.dataset}/prepare.js" first`
  );
}

// poor man's data loader, mirroring np.memmap reads in train.py
function getBatch(split) {
  const get = makeGetBatch(data_dir, {
    split,
    batch_size: config.batch_size,
    block_size: config.block_size,
  });
  return get();
}

// init these up here, can override if init_from='resume' (i.e. from a checkpoint)
let iter_num = 0;
let best_val_loss = 1e9;

// attempt to derive vocab_size from the dataset
const meta_path = path.join(data_dir, 'meta.json');
let meta_vocab_size = null;
if (fs.existsSync(meta_path)) {
  const meta = JSON.parse(fs.readFileSync(meta_path, 'utf8'));
  meta_vocab_size = meta.vocab_size;
  console.log(`found vocab_size = ${meta_vocab_size} (inside ${meta_path})`);
}

// model init
const model_args = {
  n_layer: config.n_layer,
  n_head: config.n_head,
  n_embd: config.n_embd,
  block_size: config.block_size,
  bias: config.bias,
  vocab_size: null, // filled in below
  dropout: config.dropout,
}; // start with model_args from command line
let model; // the model (in Python, possibly torch.compile-wrapped or DDP-wrapped)
if (config.init_from === 'scratch') {
  // init a new model from scratch
  console.log('Initializing a new model from scratch');
  // determine the vocab size we'll use for from-scratch training
  if (meta_vocab_size == null) {
    console.log('defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)');
  }
  model_args.vocab_size = meta_vocab_size != null ? meta_vocab_size : 50304;
  const gptconf = new GPTConfig(model_args);
  model = new GPT(gptconf);
} else if (config.init_from === 'resume') {
  // resume training from a checkpoint.
  console.log(`Resuming training from ${config.out_dir}`);
  const checkpoint = loadCheckpointMetaOnly(config.out_dir);
  const checkpoint_model_args = checkpoint.model_args;
  // force these config attributes to be equal otherwise we can't even resume
  // training; the rest of the attributes (e.g. dropout) can stay as desired
  for (const k of ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']) {
    model_args[k] = checkpoint_model_args[k];
  }
  const gptconf = new GPTConfig(model_args);
  model = new GPT(gptconf);
  iter_num = checkpoint.iter_num;
  best_val_loss = checkpoint.best_val_loss;
} else if (config.init_from.startsWith('gpt2')) {
  // initialize from OpenAI GPT-2 weights
  console.log(`Initializing from OpenAI GPT-2 weights: ${config.init_from}`);
  const override_args = { dropout: config.dropout };
  model = await GPT.fromPretrained(config.init_from, override_args);
  // read off the created config params, so we can store them into checkpoint correctly
  for (const k of ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']) {
    model_args[k] = model.config[k];
  }
}
// crop down the model block size if desired, using model surgery
if (config.block_size < model.config.block_size) {
  model.cropBlockSize(config.block_size);
  model_args.block_size = config.block_size; // so that the checkpoint will have the right value
}

// optimizer
const optimizer = model.configureOptimizers(
  config.weight_decay,
  config.learning_rate,
  [config.beta1, config.beta2],
  config.device
);
if (config.init_from === 'resume') {
  // restore both model weights and optimizer state from the checkpoint
  loadCheckpoint(config.out_dir, model, optimizer);
}

if (config.wandb_log) {
  // wandb is not available in the Node.js port; the config keys are kept so
  // Python config files remain compatible
  console.log('note: wandb logging is not available in the Node.js port, continuing without it');
}

// helps estimate an arbitrarily accurate loss over either split using many batches
function estimateLoss() {
  const out = {};
  for (const split of ['train', 'val']) {
    let sum = 0.0;
    for (let k = 0; k < config.eval_iters; k++) {
      const { x, y } = getBatch(split);
      const { loss } = tf.tidy(() => model.forward(x, y, false));
      sum += loss.dataSync()[0];
      loss.dispose();
      x.dispose();
      y.dispose();
    }
    out[split] = sum / config.eval_iters;
  }
  return out;
}

// learning rate decay scheduler (cosine with warmup)
function getLr(it) {
  // 1) linear warmup for warmup_iters steps
  if (it < config.warmup_iters) {
    return config.learning_rate * (it + 1) / (config.warmup_iters + 1);
  }
  // 2) if it > lr_decay_iters, return min learning rate
  if (it > config.lr_decay_iters) {
    return config.min_lr;
  }
  // 3) in between, use cosine decay down to min learning rate
  const decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters);
  if (!(decay_ratio >= 0 && decay_ratio <= 1)) {
    throw new Error(`decay_ratio out of [0, 1]: ${decay_ratio}`);
  }
  const coeff = 0.5 * (1.0 + Math.cos(Math.PI * decay_ratio)); // coeff ranges 0..1
  return config.min_lr + coeff * (config.learning_rate - config.min_lr);
}

function loadCheckpointMetaOnly(outDir) {
  const jsonPath = path.join(outDir, 'ckpt.json');
  if (!fs.existsSync(jsonPath)) {
    throw new Error(`file not found: ${jsonPath}`);
  }
  return JSON.parse(fs.readFileSync(jsonPath, 'utf8'));
}

// training loop
const raw_model = model; // no DDP wrapper in this port
let t0 = performance.now();
let local_iter_num = 0; // number of iterations in the lifetime of this process
let running_mfu = -1.0;
while (true) {
  // determine and set the learning rate for this iteration
  const lr = config.decay_lr ? getLr(iter_num) : config.learning_rate;

  // evaluate the loss on train/val sets and write checkpoints
  if (iter_num % config.eval_interval === 0 && master_process) {
    const losses = estimateLoss();
    console.log(`step ${iter_num}: train loss ${losses.train.toFixed(4)}, val loss ${losses.val.toFixed(4)}`);
    if (losses.val < best_val_loss || config.always_save_checkpoint) {
      best_val_loss = losses.val;
      if (iter_num > 0) {
        console.log(`saving checkpoint to ${config.out_dir}`);
        saveCheckpoint(config.out_dir, {
          model: raw_model,
          optimizer,
          modelArgs: model_args,
          iterNum: iter_num,
          bestValLoss: best_val_loss,
          config: config,
        });
      }
    }
  }
  if (iter_num === 0 && config.eval_only) break;

  // forward backward update, with optional gradient accumulation to simulate
  // larger batch size
  let last_micro_loss = null;
  const accum = {}; // gradient accumulators, keyed by variable name
  for (let micro_step = 0; micro_step < config.gradient_accumulation_steps; micro_step++) {
    // in DDP training, python would only sync grads at the last micro step;
    // not applicable here
    const { x, y } = getBatch('train');
    const { value: lossVal, grads } = tf.variableGrads(() => {
      const { loss } = model.forward(x, y, true);
      // scale the loss to account for gradient accumulation
      return tf.mul(loss, 1.0 / config.gradient_accumulation_steps);
    });
    last_micro_loss = lossVal.dataSync()[0]; // scaled loss, like loss.item()
    lossVal.dispose();
    // backward pass done by variableGrads; accumulate the gradients
    for (const k of Object.keys(grads)) {
      const g = grads[k];
      if (accum[k] == null) {
        accum[k] = g;
      } else {
        const sum = tf.add(accum[k], g);
        accum[k].dispose();
        accum[k] = sum;
        g.dispose();
      }
    }
    // immediately prefetch the next batch, mirroring the asyncio prefetch;
    // our loader is lazy per micro step, so nothing more to do here
    x.dispose();
    y.dispose();
  }
  // clip the gradient (global norm over all params, like
  // torch.nn.utils.clip_grad_norm_)
  if (config.grad_clip !== 0.0) {
    let normSq = 0.0;
    tf.tidy(() => { // keeps the per-grad temp tensors alive only inside this scope
      for (const k of Object.keys(accum)) {
        normSq += tf.sum(tf.square(accum[k])).dataSync()[0];
      }
      return null;
    });
    const norm = Math.sqrt(normSq);
    if (norm > config.grad_clip) {
      const scale = config.grad_clip / norm;
      for (const k of Object.keys(accum)) {
        const scaled = tf.mul(accum[k], scale);
        accum[k].dispose();
        accum[k] = scaled;
      }
    }
  }
  // step the optimizer (and clear grads, like optimizer.zero_grad(set_to_none=True))
  optimizer.step(accum, lr);

  // timing and logging
  const t1 = performance.now();
  const dt = t1 - t0;
  t0 = t1;
  if (iter_num % config.log_interval === 0 && master_process) {
    // scale up to undo the division above, approximating the true total loss
    const lossf = last_micro_loss * config.gradient_accumulation_steps;
    if (local_iter_num >= 5) { // let the training loop settle a bit
      const mfu = raw_model.estimateMfu(config.batch_size * config.gradient_accumulation_steps, dt / 1000);
      running_mfu = running_mfu === -1.0 ? mfu : 0.9 * running_mfu + 0.1 * mfu;
    }
    console.log(`iter ${iter_num}: loss ${lossf.toFixed(4)}, time ${(dt).toFixed(2)}ms, mfu ${(running_mfu * 100).toFixed(2)}%`);
  }
  iter_num += 1;
  local_iter_num += 1;

  // termination conditions
  if (iter_num > config.max_iters) break;
}
