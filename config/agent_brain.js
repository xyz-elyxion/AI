// AgentKit tutorial brain training config.
// Trains a tiny GPT on the agent-protocol corpus produced by
// data/tutorial/prepare.js. Run: node train.js config/agent_brain.js

export default {
  out_dir: 'out-demo',

  dataset: 'tutorial', // data/tutorial (train.bin + meta.json)
  eval_interval: 500,
  eval_iters: 20, // smaller than the usual 200: tiny model, fast evals
  log_interval: 500,
  always_save_checkpoint: true,

  gradient_accumulation_steps: 1,
  batch_size: 8,
  block_size: 64, // transcripts are short lines

  // baby GPT brain — trains in about a minute on CPU
  n_layer: 2,
  n_head: 2,
  n_embd: 64,
  dropout: 0.0,

  learning_rate: 1e-3,
  max_iters: 2000,
  lr_decay_iters: 2000,
  min_lr: 1e-4,
  beta2: 0.99,
  warmup_iters: 100,
};
