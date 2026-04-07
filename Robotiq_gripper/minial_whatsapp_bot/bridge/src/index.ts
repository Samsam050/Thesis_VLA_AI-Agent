import { BridgeServer } from './server.js';

const PORT = parseInt(process.env.BRIDGE_PORT || '8765');
const AUTH_DIR = process.env.BRIDGE_AUTH_DIR || './wa_auth';
const TOKEN = process.env.BRIDGE_TOKEN;

const server = new BridgeServer(PORT, AUTH_DIR, TOKEN);

process.on('SIGINT', async () => { await server.stop(); process.exit(0); });
process.on('SIGTERM', async () => { await server.stop(); process.exit(0); });

server.start().catch((err) => { console.error('Bridge failed:', err); process.exit(1); });
