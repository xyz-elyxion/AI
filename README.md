# nanoGPT (Node.js)

![nanoGPT](assets/nanogpt.jpg)

**Update Nov 2025** nanoGPT has a new and improved cousin called [nanochat](https://github.com/karpathy/nanochat). It is very likely you meant to use/find nanochat instead.

This repository is a **Node.js port of nanoGPT** (originally Python/PyTorch by Andrej Karpathy) that trains/finetunes medium-sized GPTs using [TensorFlow.js](https://www.tensorflow.org/js). The original Python implementation was rewritten, not merely wrapped: `model.js` is the full GPT (layer norms, causally masked multi-head attention, GELU MLPs, weight-tied `lm_head`), `train.js` is the training loop with gradient accumulation, gradient clipping, warmup + cosine learning rate decay, and checkpointing, and `sample.js` generates text — optionally from OpenAI's released GPT-2 weights.

## install

Requires Node.js >= 18 (or Bun). Then:

```sh
bun install   # or npm install
```

Dependencies:

- [`@tensorflow/tfjs-node`](https://www.npmjs.com/package/@tensorflow/tfjs-node) — tensors, ops and autograd (`tf.variableGrads`) on native TensorFlow bindings
- [`js-tiktoken`](https://www.npmjs.com/package/js-tiktoken) — the GPT-2 BPE tokenizer
- [`hyparquet`](https://www.npmjs.com/package/hyparquet) — pure-JS parquet reader (for the OpenWebText prep script)

## layout

| file | role |
| ---- | ---- |
| `model.js` | GPT model, AdamW optimizer, checkpoint I/O, GPT-2 weight loading |
| `train.js` | training loop |
| `sample.js` | sampling / inference |
| `bench.js` | short benchmark version of train.js |
| `configurator.js` | "poor man's configurator" (config files + `--key=value` overrides) |
| `config/*.js` | training configuration presets |
| `data/*/prepare.js` | dataset preparation (download + tokenize to train.bin/val.bin) |
| `dataloader.js` | uint16 binary token reader (np.memmap equivalent) |

## quick start

Get your feet wet by training a character-level GPT on Shakespeare. First download the dataset (1MB) and turn it into a stream of integers:

```sh
node data/shakespeare_char/prepare.js
```

This creates `train.bin`, `val.bin` and `meta.json` in that directory. Then train — the model size very much depends on your compute:

**I have a GPU.** Install `@tensorflow/tfjs-node-gpu` instead of `tfjs-node` and run the preset config:

```sh
node train.js config/train_shakespeare_char.js
```

This preset trains a GPT with context size 256 characters, 384 feature channels, 6-layer Transformer with 6 heads. On a fast GPU this corresponds to the famous ~3 minute run reaching best val loss ~1.47, writing checkpoints to `out-shakespeare-char`. Once training finishes, sample from the best model:

```sh
node sample.js --init_from=resume --out_dir=out-shakespeare-char
```

**I only have a laptop.** Dial things down:

```sh
node train.js config/train_shakespeare_char.js \
    --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 \
    --n_layer=4 --n_head=4 --n_embd=128 --max_iters=2000 \
    --lr_decay_iters=2000 --dropout=0.0
node sample.js --init_from=resume --out_dir=out-shakespeare-char
```

On CPU this still gives you the right character-level gestalt after a few minutes (`block_size=64, n_layer=4, n_head=4, n_embd=128` gets you sampleable text).

## differences vs the PyTorch original

- The node backend trains in **float32 on CPU** (or NVIDIA GPU with `@tensorflow/tfjs-node-gpu`). `--dtype=bfloat16/float16` autocast, `torch.compile` and DDP/torchrun multi-GPU have no equivalent here; the flags are accepted but ignored with a warning.
- `wandb` logging is kept in config keys for compatibility but not wired up.
- Attention uses a manual masked softmax implementation (peers to the "slow attention" path in the original; a flash-attention kernel is not available from TensorFlow.js here).
- GELU uses the tanh approximation that GPT-2 shipped with (same as HF `gelu_new`, `approximate='tanh'`).
- Checkpoints are `ckpt.json` + `ckpt.bin` (weights, optimizer state and metadata) instead of `ckpt.pt`; dataset meta is `meta.json` instead of `meta.pkl`.

## reproducing GPT-2

Tokenize OpenWebText with GPT-2 BPE (full run: ~17GB of parquet, ~9B tokens, takes a long time):

```sh
node data/openwebtext/prepare.js
node train.js config/train_gpt2.js
```

Then sample with `node sample.js`. If you have spare time and want to start from OpenAI's pretrained weights instead, initialize from them (downloaded from HuggingFace, converted on load):

```sh
node train.js config/finetune_shakespeare.js   # finetune gpt2-xl on Shakespeare
```

Evaluation presets for the GPT-2 124M/350M/774M/1.6B baselines also exist:

```sh
node train.js config/eval_gpt2.js
```

## acknowledge

All nanoGPT experiments are powered by GPUs on [Lambda labs](https://lambdalabs.com), and the original idea is by [Andrej Karpathy](https://github.com/karpathy/nanoGPT). For background on GPT and language modeling, see his [Zero To Hero series](https://karpathy.ai/zero-to-hero.html).
