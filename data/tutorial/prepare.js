/**
 * Prepare the AgentKit tutorial corpus for character-level language modeling.
 *
 * It tokenizes whatever is in data/tutorial/corpus.txt (regenerate it with
 * make-corpus.mjs, or drop in any text of your own) and writes train.bin,
 * val.bin and meta.json into this directory. No downloads involved.
 *
 *   node data/tutorial/prepare.js
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

const corpusPath = path.join(here, 'corpus.txt');
if (!fs.existsSync(corpusPath)) {
  console.log('corpus.txt not found, generating the default one...');
  await import('./make-corpus.mjs');
}
const data = fs.readFileSync(corpusPath, 'utf8');
console.log(`length of corpus in characters: ${data.length.toLocaleString()}`);

// character-level vocabulary
const chars = [...new Set([...data])].sort();
const vocab_size = chars.length;
console.log(`vocab size: ${vocab_size.toLocaleString()}`);

const stoi = Object.fromEntries(chars.map((ch, i) => [ch, i]));
const itos = Object.fromEntries(chars.map((ch, i) => [String(i), ch]));

// 90/10 train/val split
const n = data.length;
const split = Math.floor(n * 0.9);

function writeBin(name, textPart) {
  const ids = [...textPart].map((c) => stoi[c]);
  const buf = Buffer.alloc(ids.length * 2);
  for (let i = 0; i < ids.length; i++) buf.writeUInt16LE(ids[i], i * 2);
  fs.writeFileSync(path.join(here, name), buf);
  return ids.length;
}
console.log(`train has ${writeBin('train.bin', data.slice(0, split)).toLocaleString()} tokens`);
console.log(`val has ${writeBin('val.bin', data.slice(split)).toLocaleString()} tokens`);

fs.writeFileSync(
  path.join(here, 'meta.json'),
  JSON.stringify({ vocab_size, itos, stoi, source: 'corpus.txt' })
);
console.log('wrote meta.json');
