// evaluate the large gpt2 (774M parameters)
// n_layer=36, n_head=20, n_embd=1280
// run with: node train.js config/eval_gpt2_large.js
export default {
  batch_size: 8,
  eval_iters: 500, // use more iterations to get good estimate
  eval_only: true,
  wandb_log: false,
  init_from: 'gpt2-large',
};
