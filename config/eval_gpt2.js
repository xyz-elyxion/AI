// evaluate the base gpt2
// n_layer=12, n_head=12, n_embd=768
// 124M parameters
// run with: node train.js config/eval_gpt2.js
export default {
  batch_size: 8,
  eval_iters: 500, // use more iterations to get good estimate
  eval_only: true,
  wandb_log: false,
  init_from: 'gpt2',
};
