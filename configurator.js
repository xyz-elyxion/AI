/**
 * Poor Man's Configurator. Probably a terrible idea. Example usage:
 * $ node train.js config/override_file.js --batch_size=32
 * this will first run config/override_file.js, then override batch_size to 32
 *
 * Node.js port of configurator.py. Python config files become ES modules that
 * `export default` a plain object of overrides. Command line overrides work the
 * same as in Python: `--key=value` with values parsed like Python's
 * literal_eval (numbers, booleans, quoted strings; anything else is a string).
 *
 * Usage from an entry script:
 *   const config = configureConfig(DEFAULTS);
 * Each entry script defines its default config object (mirroring the module
 * level globals in the Python scripts).
 */

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

function parseValue(val) {
  // mirror Python's ast.literal_eval attempt
  try {
    return JSON.parse(val);
  } catch {
    // if that goes wrong, just use the string (like the Python fallback)
    return val;
  }
}

function typesCompatible(attempt, current) {
  // Python asserts `type(attempt) == type(globals()[key])`. In JS, numbers are
  // all "number" and booleans "boolean", so mirror that check at type level;
  // also allow int/float confusion between 0.05 vs 5 etc. (both are numbers).
  const first = attempt === null ? 'null' : typeof attempt;
  const second = current === null ? 'null' : typeof current;
  return first === second;
}

export async function configureConfig(defaults) {
  const args = process.argv.slice(2);
  const config = { ...defaults };

  for (const arg of args) {
    if (!arg.includes('=')) {
      // assume it's the name of a config file
      if (arg.startsWith('--')) {
        throw new Error(`cannot have a -- separator before a config file: ${arg}`);
      }
      const configFile = arg;
      console.log(`Overriding config with ${configFile}:`);
      console.log(fs.readFileSync(configFile, 'utf8'));
      const resolved = path.resolve(configFile);
      // eslint-disable-next-line no-await-in-loop
      const mod = await import(pathToFileURL(resolved).href);
      const overrides = mod.default ?? mod;
      for (const [k, v] of Object.entries(overrides)) {
        if (!(k in config)) {
          console.warn(`WARNING: config file sets unknown key "${k}", ignoring it`);
          continue;
        }
        if (!typesCompatible(v, config[k])) {
          throw new Error(
            `config key "${k}" in ${configFile} has type ${typeof v}, ` +
            `but the default is of type ${typeof config[k]}`
          );
        }
        console.log(`Overriding: ${k} = ${JSON.stringify(v)}`);
        config[k] = v;
      }
    } else {
      // assume it's a --key=value argument
      if (!arg.startsWith('--')) {
        throw new Error(`additional configuration arguments must be --key=value: ${arg}`);
      }
      const eqIndex = arg.indexOf('=');
      const key = arg.slice(2, eqIndex);
      const val = arg.slice(eqIndex + 1);
      if (!(key in config)) {
        throw new Error(`Unknown config key: ${key}`);
      }
      let attempt = parseValue(val);
      if (!typesCompatible(attempt, config[key])) {
        // like the Python version: only accept if types line up
        attempt = val;
        if (!typesCompatible(attempt, config[key])) {
          throw new Error(
            `config key "${key}" has type ${typeof attempt}, ` +
            `but the default is of type ${typeof config[key]}`
          );
        }
      }
      console.log(`Overriding: ${key} = ${JSON.stringify(attempt)}`);
      config[key] = attempt;
    }
  }

  return config;
}
