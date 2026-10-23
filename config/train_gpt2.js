// config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
// The Python version launched with torchrun across 8 GPUs; multi-GPU/DDP has
// no Node.js equivalent, so run this config as-is on a single fast machine:
// $ node train.js config/train_gpt2.js

export default {
  wandb_log: true,
  wandb_project: 'owt',
  wandb_run_name: 'gpt2-124M',

  // these make the total batch size be ~0.5M
  // 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
  batch_size: 12,
  block_size: 1024,
  gradient_accumulation_steps: 5 * 8,

  // this makes total number of tokens be 300B
  max_iters: 600000,
  lr_decay_iters: 600000,

  // eval stuff
  eval_interval: 1000,
  eval_iters: 200,
  log_interval: 10,

  // weight decay
  weight_decay: 1e-1,
};
