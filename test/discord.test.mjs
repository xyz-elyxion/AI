import test from 'node:test';
import assert from 'node:assert/strict';

import { requireDiscordToken, split } from '../agentkit/discord.js';

test('requireDiscordToken reports a clear error when the token is missing', () => {
  const hadToken = !!process.env.DISCORD_BOT_TOKEN;
  if (hadToken) {
    // token configured: it must return the token, not throw
    assert.equal(typeof requireDiscordToken(), 'string');
  } else {
    assert.throws(() => requireDiscordToken(), /DISCORD_BOT_TOKEN is not set/);
  }
});

test('split respects the reply length limit', () => {
  assert.deepEqual(split('', 10), ['']);
  assert.deepEqual(split('abcdefghij', 10), ['abcdefghij']);
  assert.deepEqual(split('abcdefghijkl', 10), ['abcdefghij', 'kl']);
  assert.deepEqual(split('a'.repeat(21), 10), ['aaaaaaaaaa', 'aaaaaaaaaa', 'a']);
});
