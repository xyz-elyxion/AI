// evaluate the xl gpt2 (1.5B parameters)
// n_layer=48, n_head=25, n_embd=1600
// run with: node train.js config/eval_gpt2_xl.js
export default {
  batch_size: 8,
  eval_iters: 500, // use more iterations to get good estimate
  eval_only: true,
  wandb_log: false,
  init_from: 'gpt2-xl',
};
