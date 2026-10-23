/**
 * Saves the openwebtext dataset to binary files for training.
 *
 * The Python version used huggingface `datasets`; this port streams the same
 * dataset from the HF parquet conversion (Skylion007/openwebtext, which is
 * where the openwebtext dataset lives these days) using pure-JS parquet reads.
 * It tokenizes each document with the GPT-2 BPE (js-tiktoken) and appends the
 * ids into train.bin / val.bin (uint16 little-endian).
 *
 * Usage: node data/openwebtext/prepare.js [--max-docs N]
 * (pass --max-docs to only fetch a small sample, e.g. for a smoke test)
 *
 * Full run downloads ~17GB of parquet and tokenizes ~9B tokens; like the
 * Python version this takes a long time (a few hours+ depending on cores).
 * A deterministic per-document PRNG routes 0.0005 of documents to val,
 * mirroring the train_test_split(test_size=0.0005, seed=2357) in prepare.py.
 */

import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';
import { encodingForModel } from 'js-tiktoken';
import {
  parquetQuery,
  parquetMetadataAsync,
  asyncBufferFromFile,
  cachedAsyncBuffer,
} from 'hyparquet';

const here = path.dirname(fileURLToPath(import.meta.url));

// parse --max-docs=N from command line (optional smoke-test cap)
const maxDocsArg = process.argv.find((a) => a.startsWith('--max-docs'));
const maxDocs = maxDocsArg ? parseInt(maxDocsArg.split('=')[1], 10) : Infinity;

// gpt2 bpe tokenizer; ordinary encode, then append the eot token (51250 or
// rather 50256 for gpt2) per document, mirroring process() in prepare.py
const enc = encodingForModel('gpt2');
const EOT = 50256;

// dataset parquet shard listing (train split only, like the Python datasets)
const listingUrl = 'https://datasets-server.huggingface.co/parquet?dataset=Skylion007/openwebtext';
const res = await fetch(listingUrl);
if (!res.ok) throw new Error(`failed to list parquet shards: HTTP ${res.status}`);
const parquetFiles = (await res.json()).parquet_files.filter((f) => f.split === 'train');
console.log(`found ${parquetFiles.length} parquet shards for openwebtext train split`);

// deterministic per-index PRNG (mulberry-ish) for the val split
function hashToUnit(seed, i) {
  // xorshift-style integer mix, deterministic, fast
  let x = seed ^ (i + 1);
  x ^= x >>> 16; x = Math.imul(x, 2246822519);
  x ^= x >>> 13; x = Math.imul(x, 3266489917);
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

// descriptors for the two output streams
const splitFds = {
  train: fs.openSync(path.join(here, 'train.bin'), 'w'),
  val: fs.openSync(path.join(here, 'val.bin'), 'w'),
};
const splitTokenCounts = { train: 0, val: 0 };
const splitDocCounts = { train: 0, val: 0 };
let splitFdsClosed = false;
function closeOutputs() {
  if (splitFdsClosed) return;
  for (const fd of Object.values(splitFds)) fs.closeSync(fd);
  splitFdsClosed = true;
}
// close the output files at the end of the (possibly early-exited) run
process.on('exit', closeOutputs);

const cacheDir = path.join(os.homedir(), '.cache', 'nanogpt-js', 'openwebtext');
fs.mkdirSync(cacheDir, { recursive: true });

function writeTokens(split, ids) {
  // append a chunk of uint16 ids to the given output bin
  const buf = Buffer.alloc(ids.length * 2);
  for (let i = 0; i < ids.length; i++) buf.writeUInt16LE(ids[i], i * 2);
  fs.writeSync(splitFds[split], buf);
  splitTokenCounts[split] += ids.length;
  splitDocCounts[split] += 1;
}

let totalDocs = 0;
outer: for (let fi = 0; fi < parquetFiles.length; fi++) {
  const shard = parquetFiles[fi];
  if (splitFdsClosed) break;
  const cachePath = path.join(cacheDir, path.basename(decodeURIComponent(new URL(shard.url).pathname)));
  if (!fs.existsSync(cachePath)) {
    console.log(`downloading shard ${fi + 1}/${parquetFiles.length}: ${shard.url.split('/').pop()} (${(shard.size / 1e9).toFixed(2)} GB)...`);
    const res = await fetch(shard.url);
    if (!res.ok) throw new Error(`failed to download parquet shard: HTTP ${res.status}`);
    await pipeline(Readable.fromWeb(res.body), fs.createWriteStream(cachePath + '.part'));
    fs.renameSync(cachePath + '.part', cachePath);
  }
  // stream the shard row group by row group to keep memory bounded
  const file = cachedAsyncBuffer(await asyncBufferFromFile(cachePath));
  const meta = await parquetMetadataAsync(file);
  // build cumulative row ranges: row group i covers [start, end)
  const ranges = [];
  let cumRows = 0;
  for (const rg of meta.row_groups) {
    const numRowsRg = Number(rg.num_rows);
    ranges.push([cumRows, cumRows + numRowsRg]);
    cumRows += numRowsRg;
  }
  for (const [rowStart, rowEnd] of ranges) {
    const rows = await parquetQuery({
      file,
      columns: ['text'],
      rowStart,
      rowEnd,
    });
    for (const row of rows) {
      const text = row.text;
      // encode_ordinary ignores any special tokens
      const ids = enc.encode(text, [], []);
      ids.push(EOT); // add the end of text token, e.g. 50256 for gpt2 bpe
      const split = hashToUnit(2357, totalDocs) < 0.0005 ? 'val' : 'train';
      writeTokens(split, ids);
      totalDocs += 1;
      if (totalDocs % 100000 === 0) {
        console.log(`docs: ${splitDocCounts.train.toLocaleString()} train / ${splitDocCounts.val.toLocaleString()} val`);
      }
      if (totalDocs >= maxDocs) break outer;
    }
  }
  console.log(`shard ${fi + 1}/${parquetFiles.length} done: tokens so far ${(splitTokenCounts.train + splitTokenCounts.val).toLocaleString()}`);
}

const counts = { train: splitTokenCounts.train, val: splitTokenCounts.val };
console.log('train.bin tokens:', counts.train.toLocaleString());
console.log('val.bin tokens:', counts.val.toLocaleString());
console.log('total docs:', totalDocs.toLocaleString());

// note: train.bin is ~17GB, val.bin ~8.5MB for a full run
// train has ~9B tokens (9,035,582,198)
// val has ~4M tokens (4,434,897)

// to read the bin files later, e.g. with the dataloader:
//   const data = new U16Memmap('data/openwebtext/train.bin')
