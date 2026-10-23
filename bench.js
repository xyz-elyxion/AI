/**
 * A much shorter version of train.js for benchmarking
 */

import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import * as tf from '@tensorflow/tfjs-node';

import { GPTConfig, GPT } from './model.js';
import { configureConfig } from './configurator.js';
import { makeGetBatch } from './dataloader.js';

// -----------------------------------------------------------------------------
const DEFAULTS = {
  batch_size: 12,
  block_size: 1024,
  bias: false,
  real_data: true,
  seed: 1337,
  device: 'cpu', // 'cpu'; gpu requires @tensorflow/tfjs-node-gpu
  dtype: 'float32',
  compile: false, // torch.compile is a PyTorch feature; no equivalent
  profile: false, // the pytorch profiler has no Node.js equivalent, ignored
};
const config = await configureConfig(DEFAULTS);
// -----------------------------------------------------------------------------

if (config.dtype !== 'float32') {
  console.warn(`WARNING: the node backend only supports float32, ignoring --dtype=${config.dtype}`);
}
if (config.compile) {
  console.warn('WARNING: torch.compile has no Node.js equivalent, ignoring --compile=true');
}
if (config.profile) {
  console.warn('WARNING: the pytorch profiler has no Node.js equivalent, using simple benchmarking');
}

// data loading init
let getBatchFn;
if (config.real_data) {
  const dataset = 'openwebtext';
  const data_dir = path.join('data', dataset);
  if (!fs.existsSync(path.join(data_dir, 'train.bin'))) {
    throw new Error(`cannot find ${data_dir}/train.bin; run "node data/openwebtext/prepare.js" first`);
  }
  getBatchFn = makeGetBatch(data_dir, {
    split: 'train', // note: split ignored in the benchmarking script
    batch_size: config.batch_size,
    block_size: config.block_size,
  });
  function getBatch() {
    return getBatchFn();
  }
} else {
  // alternatively, if fixed data is desired to not care about data loading
  const x = tf.randomUniform([config.batch_size, config.block_size], 0, 50304, 'int32', 1337);
  const y = tf.randomUniform([config.batch_size, config.block_size], 0, 50304, 'int32', 1338);
  const getBatch = () => ({ x, y });
  getBatchFn = getBatch;
}

// model init
const gptconf = new GPTConfig({
  block_size: config.block_size, // how far back does the model look? i.e. context size
  n_layer: 12, n_head: 12, n_embd: 768, // size of the model
  dropout: 0, // for determinism
  bias: config.bias,
});
const model = new GPT(gptconf);

const optimizer = model.configureOptimizers(1e-2, 1e-4, [0.9, 0.95], config.device);

function stepModel(x, y) {
  // one fused fwd+bwd+opt step, like the body of the python benchmark loop
  const { value: lossVal, grads } = tf.variableGrads(() => model.forward(x, y, false).loss);
  const lossf = lossVal.dataSync()[0];
  lossVal.dispose();
  // note: the python benchmark calls optimizer.zero_grad then backward then
  // step; variableGrads gives us the grads to feed the optimizer directly
  optimizer.step(grads);
  return lossf;
}

// simple benchmarking (torch.cuda.synchronize has no node equivalent; node
// backend op calls are synchronous already)
for (const [stage, num_steps] of [[0, 10], [1, 20]]) { // burnin, then benchmark
  const t0 = performance.now();
  let lossf = 0;
  for (let k = 0; k < num_steps; k++) {
    const { x, y } = getBatchFn();
    lossf = stepModel(x, y);
    x.dispose();
    y.dispose();
    console.log(`${k}/${num_steps} loss: ${lossf.toFixed(4)}`);
  }
  const t1 = performance.now();
  const dt = t1 - t0;  const mfu = model.estimateMfu(config.batch_size * 1 * num_steps, dt / 1000);
  if (stage === 1) {
    console.log(`time per iteration: ${(dt / num_steps * 1000).toFixed(4)}ms, MFU: ${(mfu * 100).toFixed(2)}%`);
  }
}
