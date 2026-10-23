/**
 * Full definition of a GPT Language Model, all of it in this single file.
 *
 * Node.js/TensorFlow.js port of nanoGPT's model.py. References:
 * 1) the official GPT-2 TensorFlow implementation released by OpenAI:
 * https://github.com/openai/gpt-2/blob/master/src/model.py
 * 2) huggingface/transformers PyTorch implementation:
 * https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import * as tf from '@tensorflow/tfjs-node';

const LN_EPS = 1e-5;
const STD_INIT = 0.02;

// -----------------------------------------------------------------------------
// ops helpers (functional equivalents of the nn.Modules in model.py)

/** nn.LayerNorm with optional bias, applied over the last dimension. */
export function layerNorm(x, weight, bias) {
  const lastAxis = x.shape.length - 1;
  const { mean, variance } = tf.moments(x, lastAxis, true);
  const inv = tf.rsqrt(tf.add(variance, LN_EPS));
  let y = tf.mul(tf.sub(x, mean), inv);
  y = tf.mul(y, weight);
  if (bias != null) y = tf.add(y, bias);
  return y;
}

/** nn.Linear(x) with weight stored as [out_features, in_features] like PyTorch. */
export function linear(x, weight, bias) {
  const cIn = x.shape[x.shape.length - 1];
  const flat = tf.matMul(tf.reshape(x, [-1, cIn]), tf.transpose(weight));
  let y = tf.reshape(flat, [...x.shape.slice(0, -1), weight.shape[0]]);
  if (bias != null) y = tf.add(y, bias); // broadcasting [out]
  return y;
}

/** nn.GELU. The tfjs node backend does not export a gelu op, so we use the
 * tanh approximation that GPT-2 shipped with ("gelu_new" in HF,
 * approximate='tanh' in torch). */
export function gelu(x) {
  const inner = tf.add(x, tf.mul(tf.pow(x, 3), 0.044715));
  return tf.mul(
    tf.mul(x, 0.5),
    tf.add(tf.tanh(tf.mul(inner, Math.sqrt(2 / Math.PI))), 1)
  );
}

/** nn.Dropout(x, p=rate) if training, identity otherwise. */
export function dropout(x, rate, training) {
  if (!training || rate === 0) return x;
  return tf.dropout(x, rate);
}

/** torch.nn.functional.cross_entropy(..., ignore_index=-1), mean-reduced.
 * logits: [B, T, V] float32; targets: [B, T] int32 (-1 positions are ignored). */
export function crossEntropy(logits, targets) {
  const V = logits.shape[logits.shape.length - 1];
  const flat = tf.reshape(logits, [-1, V]); // [N, V]
  const tgt = tf.reshape(tf.cast(targets, 'int32'), [-1]); // [N]
  const n = tgt.shape[0];
  // picked[i] = flat[i, tgt[i]]  (gather over the flattened flat logits.
  // tf.gather on a 2D tensor gathers rows, so we gather from a 1D view.)
  const flatIdx = tf.add(
    tf.mul(tf.range(0, n, 1, 'int32'), tf.scalar(V, 'int32')),
    tgt
  );
  const picked = tf.gather(tf.reshape(logits, [-1]), flatIdx); // [N]
  const lse = tf.logSumExp(flat, 1); // [N]
  const ce = tf.sub(lse, picked); // per-position cross entropy, [N]
  const mask = tf.cast(tf.notEqual(tgt, tf.scalar(-1, 'int32')), 'float32');
  const loss = tf.div(tf.sum(tf.mul(ce, mask)), tf.sum(mask));
  return loss;
}

// -----------------------------------------------------------------------------
// manual AdamW (mirrors torch.optim.AdamW with decoupled weight decay groups,
// which tfjs's built-in optimizers don't support per-parameter)

export class AdamW {
  /**
   * paramDescriptors: [{ logical, variable, weightDecay }] — `logical` is the
   * stable nanoGPT-style parameter name (used for checkpointing), `variable`
   * is the tf.Variable (whose engine-level .name keys the variableGrads map).
   * betas: [beta1, beta2]; eps matches torch default 1e-8.
   */
  constructor(paramDescriptors, learningRate, betas = [0.9, 0.999]) {
    this.params = paramDescriptors;
    this.lr = learningRate;
    this.beta1 = betas[0];
    this.beta2 = betas[1];
    this.eps = 1e-8;
    this.t = 0;
    this.m = {}; // per logical name tensor
    this.v = {};
  }

  /**
   * grads: map variableName -> grad tensor (same keying as tf.variableGrads
   * output). Each grad tensor is disposed inside this call (like
   * optimizer.zero_grad()). lr: optional per-step learning rate (LR scheduler).
   */
  step(grads, lr) {
    if (lr != null) this.lr = lr;
    this.t += 1;
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);
    for (const p of this.params) {
      const g = grads[p.variable.name];
      if (!g) continue;
      const { variable, weightDecay } = p;
      const old = { m: this.m[p.logical], v: this.v[p.logical] };
      // m = beta1*m + (1-beta1)*g ; v = beta2*v + (1-beta2)*g^2
      // mhat = m / (1-beta1^t) ; vhat = v / (1-beta2^t)
      // p = p * (1 - lr*wd) - lr * mhat / (sqrt(vhat) + eps)
      let mNew;
      let vNew;
      tf.tidy(() => {
        mNew = old.m
          ? tf.keep(tf.add(tf.mul(old.m, this.beta1), tf.mul(g, 1 - this.beta1)))
          : tf.keep(tf.mul(g, 1 - this.beta1));
        vNew = old.v
          ? tf.keep(tf.add(tf.mul(old.v, this.beta2), tf.mul(tf.square(g), 1 - this.beta2)))
          : tf.keep(tf.square(tf.mul(g, Math.sqrt(1 - this.beta2))));
        const mhat = tf.mul(mNew, 1 / bc1);
        const vhat = tf.mul(vNew, 1 / bc2);
        const update = tf.div(mhat, tf.add(tf.sqrt(vhat), this.eps));
        const next = tf.sub(
          tf.mul(variable, 1 - this.lr * weightDecay),
          tf.mul(update, this.lr)
        );
        variable.assign(next);
        // `next`'s buffer now belongs to the variable (disposeData is
        // refcounted in the node backend); tidy will dispose the temp tensor
        // wrapper on scope exit, which is safe.
        return null;
      });
      if (old.m) old.m.dispose();
      if (old.v) old.v.dispose();
      this.m[p.logical] = mNew;
      this.v[p.logical] = vNew;
      // like optimizer.zero_grad(set_to_none=True) for this param
      g.dispose();
      delete grads[p.variable.name];
    }
  }

  /** free the m/v state tensors */
  disposeState() {
    for (const k of Object.keys(this.m)) if (this.m[k]) this.m[k].dispose();
    for (const k of Object.keys(this.v)) if (this.v[k]) this.v[k].dispose();
    this.m = {};
    this.v = {};
  }

  stateEntries() {
    const entries = [];
    for (const p of this.params) {
      if (this.m[p.logical]) entries.push({ name: `opt.m.${p.logical}`, data: this.m[p.logical].dataSync() });
      if (this.v[p.logical]) entries.push({ name: `opt.v.${p.logical}`, data: this.v[p.logical].dataSync() });
    }
    return entries;
  }

  loadStateEntries(entries) {
    const byName = Object.fromEntries(entries.map((e) => [e.name, e]));
    for (const p of this.params) {
      const m = byName[`opt.m.${p.logical}`];
      const v = byName[`opt.v.${p.logical}`];
      if (this.m[p.logical]) this.m[p.logical].dispose();
      if (this.v[p.logical]) this.v[p.logical].dispose();
      this.m[p.logical] = m ? tf.tensor(m.data, p.variable.shape) : null;
      this.v[p.logical] = v ? tf.tensor(v.data, p.variable.shape) : null;
    }
  }
}

/**
 * tfjs's engine requires globally unique variable names (two GPT instances in
 * one process would collide, e.g. model + training copy). We keep the logical
 * nanoGPT-style names in our own registries and uniquify the tfjs-level name.
 */
const usedTFJSNames = new Set();
function uniqVarName(base) {
  if (!usedTFJSNames.has(base)) {
    usedTFJSNames.add(base);
    return base;
  }
  let i = 1;
  while (usedTFJSNames.has(`${base}#${i}`)) i++;
  const n = `${base}#${i}`;
  usedTFJSNames.add(n);
  return n;
}

// -----------------------------------------------------------------------------
// GPT

export class GPTConfig {
  constructor({
    block_size = 1024,
    vocab_size = 50304, // GPT-2 vocab_size of 50257, padded up to nearest multiple of 64
    n_layer = 12,
    n_head = 12,
    n_embd = 768,
    dropout = 0.0,
    bias = true, // True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
  } = {}) {
    this.block_size = block_size;
    this.vocab_size = vocab_size;
    this.n_layer = n_layer;
    this.n_head = n_head;
    this.n_embd = n_embd;
    this.dropout = dropout;
    this.bias = bias;
  }
}

export class GPT {
  constructor(config) {
    if (config.vocab_size == null) throw new Error('vocab_size must be set');
    if (config.block_size == null) throw new Error('block_size must be set');
    this.config = config;
    this.paramSeed = 1337; // like torch.manual_seed(1337), for weight init
    this.namedParams = []; // [{name, variable}] in registration order
    this.byName = {}; // name -> tf.Variable

    this.wte = this._initEmbedding('transformer.wte.weight', config.vocab_size, config.n_embd);
    this.wpe = this._initEmbedding('transformer.wpe.weight', config.block_size, config.n_embd);
    this.lnF = this._initLayerNorm('transformer.ln_f', config);
    this.blocks = [];
    for (let i = 0; i < config.n_layer; i++) {
      this.blocks.push(this._initBlock(i, config));
    }
    // lm_head is weight-tied to wte: https://paperswithcode.com/method/weight-tying
    this.lmHeadWeight = this.wte;

    // init all weights done above via _initEmbedding/_initLinear; now apply
    // the special scaled init to the residual projections, per GPT-2 paper
    const stdResid = 0.02 / Math.sqrt(2 * config.n_layer);
    for (const { name, variable } of this.namedParams) {
      if (name.endsWith('c_proj.weight')) {
        variable.assign(tf.randomNormal(variable.shape, 0, stdResid, 'float32', this._nextSeed()));
      }
    }

    // report number of parameters
    console.log(`number of parameters: ${(this.getNumParams() / 1e6).toFixed(2)}M`);
  }

  _nextSeed() {
    this.paramSeed += 1;
    return this.paramSeed;
  }

  _register(name, variable) {
    this.namedParams.push({ name, variable });
    this.byName[name] = variable;
    return variable;
  }

  /** nn.Embedding table with normal(0, 0.02) init */
  _initEmbedding(name, rows, cols) {
    return this._register(
      name,
      tf.variable(tf.randomNormal([rows, cols], 0, STD_INIT, 'float32', this._nextSeed()), true, uniqVarName(name))
    );
  }

  /** nn.Linear weight+bias with normal(0, 0.02) weight init */
  _initLinear(prefix, nIn, nOut, bias) {
    const weight = this._register(
      `${prefix}.weight`,
      tf.variable(tf.randomNormal([nOut, nIn], 0, STD_INIT, 'float32', this._nextSeed()), true, uniqVarName(`${prefix}.weight`))
    );
    let biasVar = null;
    if (bias) {
      biasVar = this._register(`${prefix}.bias`, tf.variable(tf.zeros([nOut]), true, uniqVarName(`${prefix}.bias`)));
    }
    return { weight, bias: biasVar };
  }

  _initLayerNorm(prefix, { n_embd, bias }) {
    const weight = this._register(`${prefix}.weight`, tf.variable(tf.ones([n_embd]), true, uniqVarName(`${prefix}.weight`)));
    let biasVar = null;
    if (bias) biasVar = this._register(`${prefix}.bias`, tf.variable(tf.zeros([n_embd]), true, uniqVarName(`${prefix}.bias`)));
    return { weight, bias: biasVar };
  }

  _initBlock(i, config) {
    const p = `transformer.h.${i}`;
    // CausalSelfAttention
    const attn = {
      cAttn: this._initLinear(`${p}.attn.c_attn`, config.n_embd, 3 * config.n_embd, config.bias),
      cProj: this._initLinear(`${p}.attn.c_proj`, config.n_embd, config.n_embd, config.bias),
      masks: {}, // lazily built triangular causal mask, keyed by T
      dropout: config.dropout,
      nHead: config.n_head,
    };
    // MLP
    const mlp = {
      cFc: this._initLinear(`${p}.mlp.c_fc`, config.n_embd, 4 * config.n_embd, config.bias),
      cProj: this._initLinear(`${p}.mlp.c_proj`, 4 * config.n_embd, config.n_embd, config.bias),
      dropout: config.dropout,
    };
    return {
      ln1: this._initLayerNorm(`${p}.ln_1`, config),
      attn,
      ln2: this._initLayerNorm(`${p}.ln_2`, config),
      mlp,
    };
  }

  getNumParams(nonEmbedding = true) {
    /**
     * For non-embedding count (default), the position embeddings get subtracted.
     * The token embeddings would too, except due to the parameter sharing these
     * params are actually used as weights in the final layer, so we include them.
     */
    let n = 0;
    for (const { variable } of this.namedParams) {
      n += variable.size;
    }
    if (nonEmbedding) n -= this.wpe.size;
    return n;
  }

  buildCausalMask(T) {
    // lower triangular: allowed positions carry 0, disallowed -infinity
    const cond = tf.greaterEqual(
      tf.expandDims(tf.range(0, T, 1, 'int32'), 0),
      tf.expandDims(tf.range(0, T, 1, 'int32'), 1)
    );
    const mask = tf.where(
      cond,
      tf.zeros([T, T]),
      tf.fill([T, T], -Infinity)
    );
    return mask;
  }

  _forwardBlock(block, x, training) {
    const [B, T, C] = x.shape;
    // ---- CausalSelfAttention ----
    const attn = block.attn;
    let h = layerNorm(x, block.ln1.weight, block.ln1.bias);
    h = linear(h, attn.cAttn.weight, attn.cAttn.bias); // [B, T, 3C]
    const q = tf.slice(h, [0, 0, 0], [-1, -1, C]);
    const k = tf.slice(h, [0, 0, C], [-1, -1, C]);
    const v = tf.slice(h, [0, 0, 2 * C], [-1, -1, C]);
    const hs = C / attn.nHead;
    const split4d = (t4) =>
      tf.transpose(tf.reshape(t4, [B, T, attn.nHead, hs]), [0, 2, 1, 3]); // (B, nh, T, hs)
    const q4 = split4d(q);
    const k4 = split4d(k);
    const v4 = split4d(v);
    // causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    let att = tf.matMul(q4, tf.transpose(k4, [0, 1, 3, 2]));
    att = tf.mul(att, tf.scalar(1.0 / Math.sqrt(hs)));
    if (!attn.masks[T]) {
      // keep() un-registers the mask from any enclosing tidy scope so the
      // cache survives; masks outlive the ops that created them
      attn.masks[T] = tf.keep(this.buildCausalMask(T));
    }
    att = tf.add(att, attn.masks[T]); // broadcast [B,H,T,T] + [T,T]
    att = tf.softmax(att, -1);
    att = dropout(att, attn.dropout, training);
    let y = tf.matMul(att, v4); // (B, nh, T, hs)
    y = tf.transpose(y, [0, 2, 1, 3]); // (B, T, nh, hs)
    y = tf.reshape(y, [B, T, C]); // re-assemble all head outputs side by side
    // output projection
    y = linear(y, attn.cProj.weight, attn.cProj.bias);
    y = dropout(y, attn.dropout, training);
    x = tf.add(x, y);

    // ---- MLP ----
    let m = layerNorm(x, block.ln2.weight, block.ln2.bias);
    m = linear(m, block.mlp.cFc.weight, block.mlp.cFc.bias);
    m = gelu(m);
    m = linear(m, block.mlp.cProj.weight, block.mlp.cProj.bias);
    m = dropout(m, block.mlp.dropout, training);
    x = tf.add(x, m);
    return x;
  }

  /**
   * Forward the model. idx: [b, t] int32 tensor; targets: [b, t] int32 or null.
   * Returns {logits, loss} (loss is null when targets is null).
   * `training` toggles dropout (PyTorch model.train()/model.eval()).
   *
   * NOTE: intermediate tensors of the forward pass are intentionally NOT
   * disposed here. Callers must wrap forward() in a tf.tidy scope (inference)
   * or rely on tf.variableGrads' internal cleanup (training).
   */
  forward(idx, targets = null, training = false) {
    const cfg = this.config;
    const t = idx.shape[1];
    if (t > cfg.block_size) {
      throw new Error(
        `Cannot forward sequence of length ${t}, block size is only ${cfg.block_size}`
      );
    }
    const pos = tf.range(0, t, 1, 'int32'); // shape (t)
    // forward the GPT model itself
    const tokEmb = tf.gather(this.wte, tf.cast(idx, 'int32')); // (b, t, n_embd)
    const posEmb = tf.gather(this.wpe, pos); // (t, n_embd)
    let x = dropout(tf.add(tokEmb, posEmb), cfg.dropout, training);
    for (const block of this.blocks) {
      x = this._forwardBlock(block, x, training);
    }
    x = layerNorm(x, this.lnF.weight, this.lnF.bias);

    if (targets != null) {
      // if we are given some desired targets also calculate the loss
      const logits = linear(x, this.lmHeadWeight, null);
      const loss = crossEntropy(logits, targets);
      return { logits, loss };
    }
    // inference-time mini-optimization: only forward the lm_head on the very last position
    const lastX = tf.slice(x, [0, t - 1, 0], [-1, 1, -1]); // note: keeping the time dim
    const logits = linear(lastX, this.lmHeadWeight, null); // (b, 1, V)
    return { logits, loss: null };
  }

  cropBlockSize(blockSize) {
    // model surgery to decrease the block size if necessary
    // e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
    // but want to use a smaller block size for some smaller, simpler model
    if (blockSize > this.config.block_size) {
      throw new Error('cannot crop to a larger block size');
    }
    this.config.block_size = blockSize;
    const sliced = this.wpe.slice([0, 0], [blockSize, this.config.n_embd]);
    const name = 'transformer.wpe.weight';
    const newVar = tf.variable(sliced, true, uniqVarName(name));
    // swap the variable in the registry (old variable must no longer be updated)
    const idx = this.namedParams.findIndex((p) => p.name === name);
    this.namedParams[idx].variable = newVar;
    this.byName[name] = newVar;
    this.wpe = newVar;
  }

  static async fromPretrained(modelType, overrideArgs = {}) {
    /**
     * Initialize a GPT from OpenAI's released GPT-2 weights, downloaded from
     * huggingface as model.safetensors (the PyTorch version used the
     * `transformers` library; here we parse safetensors directly and need no
     * other dependency).
     */
    const modelSizes = {
      'gpt2': { n_layer: 12, n_head: 12, n_embd: 768 }, // 124M params
      'gpt2-medium': { n_layer: 24, n_head: 16, n_embd: 1024 }, // 350M params
      'gpt2-large': { n_layer: 36, n_head: 20, n_embd: 1280 }, // 774M params
      'gpt2-xl': { n_layer: 48, n_head: 25, n_embd: 1600 }, // 1558M params
    };
    if (!(modelType in modelSizes)) {
      throw new Error(`unknown model_type ${modelType}, must be one of ${Object.keys(modelSizes)}`);
    }
    for (const k of Object.keys(overrideArgs)) {
      if (k !== 'dropout') {
        throw new Error(`cannot override ${k} in from_pretrained, only dropout`);
      }
    }
    console.log(`loading weights from pretrained gpt: ${modelType}`);
    const cfgArgs = { ...modelSizes[modelType] };
    console.log('forcing vocab_size=50257, block_size=1024, bias=True');
    cfgArgs.vocab_size = 50257; // always 50257 for GPT model checkpoints
    cfgArgs.block_size = 1024; // always 1024 for GPT model checkpoints
    cfgArgs.bias = true; // always True for GPT model checkpoints
    if ('dropout' in overrideArgs) {
      console.log(`overriding dropout rate to ${overrideArgs.dropout}`);
      cfgArgs.dropout = overrideArgs.dropout;
    }
    const config = new GPTConfig(cfgArgs);
    const model = new GPT(config);

    // copy the weights, ensuring all parameters are aligned in name and shape
    const sd = await loadGpt2Safetensors(modelType);
    // discard masks/buffers that are not parameters
    const hfKeys = Object.keys(sd).filter(
      (k) => !k.endsWith('.attn.masked_bias') && !k.endsWith('.attn.bias')
    );
    const transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'];
    // basically the openai checkpoints use a "Conv1D" module, but we only want
    // to use a vanilla Linear; this means that we have to transpose these
    // weights when we import them
    for (const name of hfKeys) {
      // map HF/transformers names onto ours (top-level keys live under transformer)
      let target;
      if (name === 'wte.weight') target = 'transformer.wte.weight'; // tied to lm_head
      else if (name === 'wpe.weight') target = 'transformer.wpe.weight';
      else if (name === 'lm_head.weight') target = 'transformer.wte.weight'; // tied
      else target = `transformer.${name}`;
      if (model.byName[target] == null) {
        throw new Error(`mismatched keys: HF has ${name} but model lacks ${target}`);
      }
      const varShape = model.byName[target].shape;
      let data = sd[name];
      if (transposed.some((w) => name.endsWith(w))) {
        // special treatment for the Conv1D weights we need to transpose
        const expectedRows = varShape[0];
        const expectedCols = varShape[1];
        if (data.length !== expectedRows * expectedCols) {
          throw new Error(`shape mismatch for ${name}`);
        }
        data = transposeLastTwo(data, expectedCols, expectedRows);
      }
      if (data.length !== model.byName[target].size) {
        throw new Error(`shape mismatch for ${name}: expected ${model.byName[target].size} values, got ${data.length}`);
      }
      model.byName[target].assign(tf.tensor(data, varShape));
    }
    return model;
  }

  configureOptimizers(weightDecay, learningRate, betas, deviceType = 'cpu') {
    // start with all of the candidate parameters
    const paramDict = Object.fromEntries(
      this.namedParams.map(({ name, variable }) => [name, variable])
    );
    // create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    // i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    const decayParams = [];
    const nodecayParams = [];
    for (const [name, variable] of Object.entries(paramDict)) {
      const pd = { logical: name, variable };
      if (variable.shape.length >= 2) decayParams.push({ ...pd, weightDecay });
      else nodecayParams.push({ ...pd, weightDecay: 0.0 });
    }
    let numDecay = 0;
    let numNodecay = 0;
    for (const p of decayParams) numDecay += p.variable.size;
    for (const p of nodecayParams) numNodecay += p.variable.size;
    console.log(`num decayed parameter tensors: ${decayParams.length}, with ${numDecay.toLocaleString()} parameters`);
    console.log(`num non-decayed parameter tensors: ${nodecayParams.length}, with ${numNodecay.toLocaleString()} parameters`);
    // torch uses the "fused" AdamW kernel on cuda; our manual AdamW implements
    // the same algorithm (decoupled weight decay, bias-correction) either way
    console.log('using fused AdamW: false');
    const optimizer = new AdamW([...decayParams, ...nodecayParams], learningRate, betas);
    return optimizer;
  }

  estimateMfu(fwDdBwdPerIter, dt) {
    /** estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS */
    // first estimate the number of flops we do per iteration.
    // see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
    const N = this.getNumParams();
    const cfg = this.config;
    const L = cfg.n_layer;
    const H = cfg.n_head;
    const Q = cfg.n_embd / cfg.n_head;
    const T = cfg.block_size;
    const flopsPerToken = 6 * N + 12 * L * H * Q * T;
    const flopsPerFwdBwd = flopsPerToken * T;
    const flopsPerIter = flopsPerFwdBwd * fwDdBwdPerIter;
    // express our flops throughput as ratio of A100 bfloat16 peak flops
    const flopsAch = flopsPerIter * (1.0 / dt); // per second
    const flopsPromised = 312e12; // A100 GPU bfloat16 peak flops is 312 TFLOPS
    const mfu = flopsAch / flopsPromised;
    return mfu;
  }

  generate(idxList, maxNewTokens, { temperature = 1.0, top_k = null } = {}) {
    /**
     * Take a conditioning sequence of indices idx (array of shape (b,t)) and complete
     * the sequence max_new_tokens times, feeding the predictions back into the model
     * each time. Most likely you'll want to make sure to be in eval mode for this
     * (no dropout here, mirroring @torch.no_grad + eval in the Python version).
     * Python returned a LongTensor; here we return an array of number arrays.
     */
    const idx = idxList.map((row) => row.slice()); // (b, t)
    const B = idx.length;
    for (let step = 0; step < maxNewTokens; step++) {
      // if the sequence context is growing too long we must crop it at block_size
      const T = Math.min(idx[0].length, this.config.block_size);
      // forward the model to get the logits for the index in the sequence
      const { logits } = tf.tidy(() => {
        const tensor = tf.tensor2d(
          idx.map((row) => row.slice(-T)),
          [B, T],
          'int32'
        );
        return this.forward(tensor);
      });
      // pluck the logits at the final step (shape (b, 1, V)) and flatten per row
      const lastLogits = tf.tidy(() => tf.slice(logits, [0, 0, 0], [-1, 1, -1]).reshape([-1]));
      const V = this.config.vocab_size;
      // For each sequence in the batch, sample the next token.
      const next = [];
      for (let b = 0; b < B; b++) {
        let row = lastLogits.slice([b * V], [V]).dataSync();
        lastLogits.slice([b * V], [V]).dispose();
        // scale by desired temperature: 1.0 = no change, < 1.0 = less random,
        // > 1.0 = more random
        row = Array.from(row, (v) => v / temperature);
        // optionally crop the logits to only the top k options
        if (top_k != null) {
          const k = Math.min(top_k, row.length);
          const order = row.map((v, i) => [v, i]).sort((a2, b2) => b2[0] - a2[0]).slice(0, k);
          const threshold = order[order.length - 1][0];
          row = row.map((v) => (v < threshold ? -Infinity : v));
        }
        // apply softmax to convert logits to (normalized) probabilities, then
        // sample from the distribution. tf.multinomial takes (log-softmax'd)
        // logits and normalizes internally, matching torch.multinomial(probs).
        const tProbs = tf.tensor1d(row);
        const sample = tf.multinomial(tProbs, 1);
        next.push(sample.dataSync()[0]);
        sample.dispose();
        tProbs.dispose();
      }
      tf.dispose([lastLogits, logits]);
      // append sampled indices to the running sequence and continue
      for (let b = 0; b < B; b++) idx[b].push(next[b]);
    }
    return idx;
  }

  // ---- checkpoint/state dict helpers (serialization) ----

  /** list of {name, shape, data: Float32Array} for all named params */
  stateDict() {
    return this.namedParams.map(({ name, variable }) => ({
      name,
      shape: variable.shape.slice(),
      data: variable.dataSync(),
    }));
  }

  loadStateDict(entries) {
    const byName = Object.fromEntries(entries.map((e) => [e.name, e]));
    for (const { name, variable } of this.namedParams) {
      const e = byName[name];
      if (!e) throw new Error(`missing state dict key ${name}`);
      if (e.data.length !== variable.size) {
        throw new Error(
          `size mismatch for ${name}: expected ${variable.size}, got ${e.data.length}`
        );
      }
      variable.assign(tf.tensor(e.data, variable.shape));
    }
  }
}

function transposeLastTwo(data, rows, cols) {
  // data is length rows*cols stored row-major as [rows, cols]; return as [cols, rows]
  const src = data instanceof Float32Array ? data : Float32Array.from(data);
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      out[j * rows + i] = src[i * cols + j];
    }
  }
  return out;
}

/**
 * Download (with on-disk cache) + parse a GPT-2 model.safetensors file from
 * huggingface. Returns a map name -> Float32Array (raw HF layout;Conv1D
 * weights are [in, out] and the caller transposes where needed).
 */
export async function loadGpt2Safetensors(modelType) {
  const cacheDir = path.join(os.homedir(), '.cache', 'nanogpt-js');
  fs.mkdirSync(cacheDir, { recursive: true });
  const cachePath = path.join(cacheDir, `${modelType}-model.safetensors`);
  if (!fs.existsSync(cachePath)) {
    console.log(`downloading ${modelType} weights from huggingface to ${cachePath}...`);
    const url = `https://huggingface.co/${modelType}/resolve/main/model.safetensors`;
    // download to a .part file first so partial failures don't poison the cache
    const res = await fetch(url, { redirect: 'follow' });
    if (!res.ok || !res.body) {
      throw new Error(`failed to download ${url}: HTTP ${res.status}`);
    }
    const tmp = cachePath + '.part';
    await pipeline(Readable.fromWeb(res.body), fs.createWriteStream(tmp));
    fs.renameSync(tmp, cachePath);
  }
  const buf = fs.readFileSync(cachePath);
  // safetensors: 8-byte little-endian u64 header length, then JSON header
  const headerLen = Number(buf.readBigUInt64LE(0));
  const header = JSON.parse(buf.slice(8, 8 + headerLen).toString('utf8'));
  const dataStart = 8 + headerLen;
  const result = {};
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const { dtype, data_offsets } = info;
    if (dtype !== 'F32') {
      throw new Error(`unexpected dtype ${dtype} for ${name}, only F32 supported`);
    }
    const [start, end] = data_offsets;
    // copy out so we're independent of the big file buffer
    const view = buf.buffer.slice(
      buf.byteOffset + dataStart + start,
      buf.byteOffset + dataStart + end
    );
    result[name] = new Float32Array(view);
  }
  return result;
}

// -----------------------------------------------------------------------------
// checkpoint IO: ckpt.json (metadata + weight manifest) + ckpt.bin (raw floats)
// This replaces torch.save(torch checkpoint dict) of the Python version.

export function saveCheckpoint(outDir, { model, optimizer = null, modelArgs, iterNum, bestValLoss, config }) {
  fs.mkdirSync(outDir, { recursive: true });
  const entries = model.stateDict();
  if (optimizer != null) entries.push(...optimizer.stateEntries());
  // pack all float32 data into one binary file
  let totalFloats = 0;
  for (const e of entries) totalFloats += e.data.length;
  const binBuf = Buffer.alloc(totalFloats * 4);
  let offset = 0;
  const manifest = [];
  for (const e of entries) {
    manifest.push({ name: e.name, shape: e.shape, offset, length: e.data.length });
    writeFloats(binBuf, e.data, offset * 4);
    offset += e.data.length;
  }
  fs.writeFileSync(path.join(outDir, 'ckpt.bin'), binBuf);
  const meta = {
    model_args: modelArgs,
    iter_num: iterNum,
    best_val_loss: bestValLoss,
    config: config,
    weights: manifest,
    format_version: 1,
  };
  fs.writeFileSync(path.join(outDir, 'ckpt.json'), JSON.stringify(meta));
  return path.join(outDir, 'ckpt.json');
}

function writeFloats(buf, f32, byteOffset) {
  if (f32 instanceof Float32Array && buf.byteOffset === 0) {
    new Float32Array(buf.buffer, byteOffset, f32.length).set(f32);
  } else {
    for (let i = 0; i < f32.length; i++) buf.writeFloatLE(f32[i], byteOffset + i * 4);
  }
}

export function loadCheckpoint(outDir, model, optimizer = null) {
  const meta = JSON.parse(fs.readFileSync(path.join(outDir, 'ckpt.json'), 'utf8'));
  const bin = fs.readFileSync(path.join(outDir, 'ckpt.bin'));
  // Float32Array views require 4-byte alignment; copy into our own buffer to be safe
  const ab = new ArrayBuffer(bin.length);
  new Uint8Array(ab).set(bin);
  const allData = new Float32Array(ab);
  const modelEntries = [];
  const optEntries = [];
  for (const w of meta.weights) {
    const e = { name: w.name, data: allData.slice(w.offset, w.offset + w.length) };
    if (w.name.startsWith('opt.m.') || w.name.startsWith('opt.v.')) optEntries.push(e);
    else modelEntries.push(e);
  }
  model.loadStateDict(modelEntries);
  if (optimizer != null) optimizer.loadStateEntries(optEntries);
  return meta;
}
