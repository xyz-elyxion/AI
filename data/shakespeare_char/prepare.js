/**
 * Prepare the Shakespeare dataset for character-level language modeling.
 * So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
 * Will save train.bin, val.bin containing the ids, and meta.json containing the
 * encoder and decoder and some other related info.
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

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
console.log(`length of dataset in characters: ${data.length.toLocaleString()}`);

// get all the unique characters that occur in this text
const chars = [...new Set([...data])].sort();
const vocab_size = chars.length;
console.log(`all the unique characters: ${chars.join('')}`);
console.log(`vocab size: ${vocab_size.toLocaleString()}`);

// create a mapping from characters to integers
const stoi = Object.fromEntries(chars.map((ch, i) => [ch, i])); // encoder: string -> ints
const itos = Object.fromEntries(chars.map((ch, i) => [i, ch])); // decoder: ints -> string

// create the train and test splits
const n = data.length;
const train_data = data.slice(0, Math.floor(n * 0.9));
const val_data = data.slice(Math.floor(n * 0.9));

// export to bin files
function writeBin(name, text) {
  const ids = [...text].map((c) => stoi[c]);
  const buf = Buffer.alloc(ids.length * 2);
  for (let i = 0; i < ids.length; i++) buf.writeUInt16LE(ids[i], i * 2);
  fs.writeFileSync(path.join(here, name), buf);
  return ids.length;
}
console.log(`train has ${writeBin('train.bin', train_data).toLocaleString()} tokens`);
console.log(`val has ${writeBin('val.bin', val_data).toLocaleString()} tokens`);

// save the meta information as well, to help us encode/decode later
const meta = { vocab_size, itos, stoi };
fs.writeFileSync(path.join(here, 'meta.json'), JSON.stringify(meta));

// length of dataset in characters:  1115394
// all the unique characters:
//  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
// vocab size: 65
// train has 1003854 tokens
// val has 111540 tokens
