// Train a task-doing brain on YOUR OWN corpus (data/own/corpus.txt).
// Run:  node data/own/prepare.js && node train.js config/own_brain.js
//
// Scale up as your corpus grows (each knob is at "tiny so it trains fast"):
//
//   --n_layer=4 --n_embd=128 --n_head=4      a slightly taller brain
//   --max_iters=5000 --lr_decay_iters=5000   longer curriculum
//   --block_size=128 (or 256)                longer memory of the transcript
//     (if you raise block_size, also raise manifest.brain.max_new_tokens!)

export default {
  out_dir: 'out-own',

  dataset: 'own', // data/own (train.bin + meta.json)
  eval_interval: 500,
  eval_iters: 20, // smaller than the usual 200: tiny model, fast evals
  log_interval: 500,
  always_save_checkpoint: true,

  gradient_accumulation_steps: 1,
  batch_size: 8,
  block_size: 64, // transcripts are short lines; raise with your corpus

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
