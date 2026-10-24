/**
 * Node.js version compatibility shim, imported before any TensorFlow code.
 *
 * @tensorflow/tfjs-node 4.22.0 calls `util.isNullOrUndefined`, which Node
 * removed from `node:util` in Node 24 (deprecated since Node 4). TensorFlow
 * fixed this upstream but never shipped a new npm release (github.com/
 * tensorflow/tfjs#8746, PR #8425), so we restore the two-line behavior here.
 * tfjs reads the property off the module object at call time, so defining it
 * anywhere before the first op fixes every kernel.
 */

import util from 'node:util';

if (typeof util.isNullOrUndefined !== 'function') {
  util.isNullOrUndefined = (v) => v == null;
}
