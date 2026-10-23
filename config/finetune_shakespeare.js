// finetune a pretrained GPT-2 on tiny shakespeare, mirrored from
// config/finetune_shakespeare.py. Requires the shakespeare data:
// $ node data/shakespeare/prepare.js
// then: node train.js config/finetune_shakespeare.js

export default {
  out_dir: 'out-shakespeare',
  eval_interval: 5,
  eval_iters: 40,
  wandb_log: false, // feel free to turn on
  wandb_project: 'shakespeare',
  wandb_run_name: 'ft-' + String(Math.floor(Date.now() / 1000)),

  dataset: 'shakespeare',
  init_from: 'gpt2-xl', // this is the largest GPT-2 model

  // only save checkpoints if the validation loss improves
  always_save_checkpoint: false,

  // the number of examples per iter:
  // 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
  // shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
  batch_size: 1,
  gradient_accumulation_steps: 32,
  max_iters: 20,

  // finetune at constant LR
  learning_rate: 3e-5,
  decay_lr: false,
};
