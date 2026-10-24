/**
 * AgentKit Discord channel — run an agent manifest as a Discord bot.
 *
 * The bot joins the gateway, and every message that mentions it (or any DM)
 * is run through the same think-act loop as `agentkit run`, then replied to
 * with the final answer. Tool calls and observations stream to the console.
 *
 * Needs a bot token: DISCORD_BOT_TOKEN (from an application created at
 * https://discord.com/developers/applications with the MESSAGE CONTENT
 * intent enabled).
 */

import {
  Client,
  GatewayIntentBits,
  Partials,
} from 'discord.js';

const MAX_REPLY_LEN = 1900; // Discord's hard limit is 2000 chars

export function requireDiscordToken() {
  const token = process.env.DISCORD_BOT_TOKEN;
  if (!token) {
    throw new Error(
      'DISCORD_BOT_TOKEN is not set — create an application at https://discord.com/developers/applications, enable the MESSAGE CONTENT intent, make a bot token, and put it in Settings → Environment as DISCORD_BOT_TOKEN'
    );
  }
  return token;
}

/** Client + wiring for one agent. Returns the discord.js Client. */
export function createDiscordAgentClient(manifest, { onEvent = null } = {}) {
  const client = new Client({
    intents: [
      GatewayIntentBits.DirectMessages,
      GatewayIntentBits.Guilds,
      GatewayIntentBits.GuildMessages,
      GatewayIntentBits.MessageContent,
    ],
    partials: [Partials.Channel],
  });

  // serialize agent runs so a tiny local brain never runs two generations
  // against itself
  let queue = Promise.resolve();
  const enqueue = (job) =>
    (queue = queue.then(job).catch((e) => e instanceof Error ? e : new Error(String(e))));

  client.on('messageCreate', (message) => {
    if (message.author.bot) return;
    if (message.webhookId) return;
    const mention = message.mentions.has(client.user);
    const dm = !message.guild;
    if (!mention && !dm) return;

    enqueue(() => handle(message));
  });

  async function handle(message) {
    // strip the bot mention and reply-quote noise from the prompt
    let task = message.content
      .replace(new RegExp(`<@!?${client.user.id}>`, 'g'), '')
      .trim();
    if (!task) task = 'hi';
    if (message.reference) {
      const ref = await message.fetchReference().catch(() => null);
      if (ref) task = `${message.author.username} replies to "${ref.content.slice(0, 200)}": ${task}`;
    }

    const who =
      message.guild
        ? `${message.author.username} in #${message.channel.name}`
        : `DM from ${message.author.username}`;
    console.log(`[discord] ${who}: ${task}`);

    const sessionId = `discord-${message.channelId}`;
    let final = '';
    try {
      const { runAgent } = await import('./agent.js');
      for await (const evt of runAgent(manifest, { task, sessionId })) {
        emit(evt);
        if (evt.type === 'final') final = evt.text;
      }
    } catch (e) {
      console.error(`[discord] agent run failed: ${e.message}`);
      await message.reply(`⚠ agent failed: ${e.message}`).catch(() => {});
      return;
    }

    for (const chunk of split(final.trim() || '(no response)', MAX_REPLY_LEN)) {
      await message.reply(chunk).catch((e) =>
        console.error(`[discord] reply failed: ${e.message}`)
      );
    }
  }

  function emit(evt) {
    if (onEvent) return onEvent(evt);
    switch (evt.type) {
      case 'tool':
        console.log(`[tool] ${evt.name}(${evt.args.join(', ')})`);
        break;
      case 'final':
        console.log(`[discord] -> ${evt.text.replace(/\s+/g, ' ').slice(0, 200)}`);
        break;
    }
  }

  return client;
}

export function split(text, limit) {
  const chunks = [];
  for (let i = 0; i < text.length; i += limit) chunks.push(text.slice(i, i + limit));
  return chunks.length ? chunks : [text];
}

/** start the gateway and wait for ready; resolves with the client */
export async function startDiscordAgent(manifest, opts = {}) {
  const client = createDiscordAgentClient(manifest, opts);
  await client.login(requireDiscordToken());
  await new Promise((resolve, reject) => {
    client.once('ready', () => resolve());
    client.once('error', reject);
  });
  const u = client.user;
  console.log(`[discord] ${manifest.name} online as ${u.tag} — mention or DM it (Ctrl-C to stop)`);
  return client;
}
