// evaluate the medium gpt2 (350M parameters)
// n_layer=24, n_head=16, n_embd=1024
// run with: node train.js config/eval_gpt2_medium.js
export default {
  batch_size: 8,
  eval_iters: 500, // use more iterations to get good estimate
  eval_only: true,
  wandb_log: false,
  init_from: 'gpt2-medium',
};
