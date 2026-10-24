/**
 * Prepare YOUR OWN corpus for a task-doing AgentKit brain.
 *
 * Drop your text into data/own/corpus.txt and run:
 *
 *   node data/own/prepare.js
 *
 * It tokenizes corpus.txt (character-level, no downloads) and writes
 * train.bin / val.bin / meta.json into this directory. Then train:
 *
 *   node train.js config/own_brain.js
 *
 * To make the brain capable of doing tasks (not just chatting), shape your
 * paragraphs as transcripts so the model imitates the protocol:
 *
 *   [user] what is 6*7?
 *   [Agent] I can compute that.
 *   act: calc(6*7)
 *   [observed] calc(6*7) -> 42
 *   [Agent] That's 42.
 *
 * [observed] lines are inserted by the runtime during agent runs — include
 * some in the corpus so the model expects them. A handful of calc/exec/
 * read_file examples like the above is usually enough to unlock tool use.
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));

const corpusPath = path.join(here, 'corpus.txt');
if (!fs.existsSync(corpusPath)) {
  console.error(
    'No corpus.txt found in data/own/. Paste your own text there (plain text,\n' +
      'or task transcripts in the [user]/[Agent]/act:/[observed] shape shown in\n' +
      'this file header), then re-run this script.'
  );
  process.exit(1);
}
const data = fs.readFileSync(corpusPath, 'utf8').trim();
if (!data) {
  console.error('data/own/corpus.txt is empty — paste your own text first.');
  process.exit(1);
}
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
