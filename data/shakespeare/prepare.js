/**
 * Prepare the tiny shakespeare dataset encoded with GPT-2 BPE tokens,
 * mirroring data/shakespeare/prepare.py. Writes train.bin and val.bin.
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { encodingForModel } from 'js-tiktoken';

const here = path.dirname(fileURLToPath(import.meta.url));

// download the tiny shakespeare dataset
const input_file_path = path.join(here, 'input.txt');
let data;
if (!fs.existsSync(input_file_path)) {
  const data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt';
  const res = await fetch(data_url);
  if (!res.ok) throw new Error(`failed to download tiny shakespeare: HTTP ${res.status}`);
  data = await res.text();
  fs.writeFileSync(input_file_path, data);
} else {
  data = fs.readFileSync(input_file_path, 'utf8');
}
const n = data.length;
const train_data = data.slice(0, Math.floor(n * 0.9));
const val_data = data.slice(Math.floor(n * 0.9));

// encode with tiktoken gpt2 bpe. js-tiktoken's Tiktoken.encode has no separate
// encode_ordinary, but disallowedSpecial=[] makes special tokens be treated as
// ordinary text, which is exactly encode_ordinary semantics
const enc = encodingForModel('gpt2');
function writeBin(name, text) {
  const ids = enc.encode(text, [], []);
  const buf = Buffer.alloc(ids.length * 2);
  for (let i = 0; i < ids.length; i++) buf.writeUInt16LE(ids[i], i * 2);
  fs.writeFileSync(path.join(here, name), buf);
  return ids.length;
}
console.log(`train has ${writeBin('train.bin', train_data).toLocaleString()} tokens`);
console.log(`val has ${writeBin('val.bin', val_data).toLocaleString()} tokens`);

// train.bin has 301,966 tokens
// val.bin has 36,059 tokens
