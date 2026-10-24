
/**
 * Poor man's data loader: reads uint16 token ids from the train.bin/val.bin
 * files produced by the data prepare.js scripts, mirroring
 * np.memmap(dtype=np.uint16, mode='r') usage in the Python version.
 */

import fs from 'node:fs';
import './node-compat.js';
import * as tf from '@tensorflow/tfjs-node';

export class U16Memmap {
  constructor(filePath) {
    this.fd = fs.openSync(filePath, 'r');
    const stat = fs.fstatSync(this.fd);
    this.length = Math.floor(stat.size / 2); // uint16 = 2 bytes per token
    this.path = filePath;
  }

  /** read tokens [start, start+len) as an Int32Array-like array */
  read(start, len) {
    const buf = Buffer.alloc(len * 2);
    fs.readSync(this.fd, buf, 0, len * 2, start * 2);
    const u16 = new Uint16Array(buf.buffer, buf.byteOffset, len);
    return Int32Array.from(u16); // uint16 fits in int32 safely
  }

  close() {
    fs.closeSync(this.fd);
  }
}

/**
 * Build a getBatch(split) function like the one in train.py.
 * Returns an object {x, y} of int32 tensors shaped [batch_size, block_size].
 */
export function makeGetBatch(datasetDir, {
  split,
  batch_size,
  block_size,
  randomInts, // fn(n) -> array of random ints, for testability
  device = 'cpu', // no-op for the node backend (kept for parity with python)
}) {
  if (device !== 'cpu' && !device.startsWith('gpu')) {
    console.warn(`--device=${device} is not available in the node backend, using cpu`);
  }
  const file = split === 'train' ? 'train.bin' : 'val.bin';
  const data = new U16Memmap(`${datasetDir.replace(/\/$/, '')}/${file}`);
  const randInt = randomInts
    ? (n) => randomInts(n)
    : (n) => Math.floor(Math.random() * n);
  return function getBatch() {
    const xData = new Int32Array(batch_size * block_size);
    const yData = new Int32Array(batch_size * block_size);
    for (let b = 0; b < batch_size; b++) {
      const i = randInt(data.length - block_size);
      const seq = data.read(i, block_size + 1); // one read gives x and y
      xData.set(seq.subarray(0, block_size), b * block_size);
      yData.set(seq.subarray(1, block_size + 1), b * block_size);
    }
    const x = tf.tensor2d(xData, [batch_size, block_size], 'int32');
    const y = tf.tensor2d(yData, [batch_size, block_size], 'int32');
    return { x, y };
  };
}
