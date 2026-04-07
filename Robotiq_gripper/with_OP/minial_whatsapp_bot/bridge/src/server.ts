/**
 * WebSocket server bridging Python bot ↔ WhatsApp via Baileys.
 * Binds to 127.0.0.1 only. Optional BRIDGE_TOKEN auth.
 */
import { WebSocketServer, WebSocket } from 'ws';
import { WhatsAppClient, InboundMessage } from './whatsapp.js';

interface BridgeMessage { type: string; [key: string]: unknown; }

export class BridgeServer {
  private wss: WebSocketServer | null = null;
  private wa: WhatsAppClient | null = null;
  private clients: Set<WebSocket> = new Set();

  constructor(private port: number, private authDir: string, private token?: string) {}

  async start(): Promise<void> {
    this.wss = new WebSocketServer({ host: '127.0.0.1', port: this.port });
    console.log(`Bridge server listening on ws://127.0.0.1:${this.port}`);
    if (this.token) console.log('Token authentication enabled');

    this.wa = new WhatsAppClient({
      authDir: this.authDir,
      onMessage: (msg) => this.broadcast({ type: 'message', ...msg }),
      onQR: (qr) => this.broadcast({ type: 'qr', qr }),
      onStatus: (status) => this.broadcast({ type: 'status', status }),
    });

    this.wss.on('connection', (ws) => {
      if (this.token) {
        const timeout = setTimeout(() => ws.close(4001, 'Auth timeout'), 5000);
        ws.once('message', (data) => {
          clearTimeout(timeout);
          try {
            const msg = JSON.parse(data.toString());
            if (msg.type === 'auth' && msg.token === this.token) {
              console.log('Python client authenticated');
              this.setupClient(ws);
            } else {
              ws.close(4003, 'Invalid token');
            }
          } catch { ws.close(4003, 'Invalid auth message'); }
        });
      } else {
        console.log('Python client connected');
        this.setupClient(ws);
      }
    });

    await this.wa.connect();
  }

  private setupClient(ws: WebSocket): void {
    this.clients.add(ws);
    ws.on('message', async (data) => {
      try {
        const cmd = JSON.parse(data.toString());
        if (cmd.type === 'send' && this.wa) {
          await this.wa.sendMessage(cmd.to, cmd.text);
          ws.send(JSON.stringify({ type: 'sent', to: cmd.to }));
        } else if (cmd.type === 'send_image' && this.wa) {
          await this.wa.sendImage(cmd.to, cmd.image, cmd.caption);
          ws.send(JSON.stringify({ type: 'sent', to: cmd.to }));
        }
      } catch (error) {
        ws.send(JSON.stringify({ type: 'error', error: String(error) }));
      }
    });
    ws.on('close', () => this.clients.delete(ws));
    ws.on('error', () => this.clients.delete(ws));
  }

  private broadcast(msg: BridgeMessage): void {
    const data = JSON.stringify(msg);
    for (const client of this.clients) {
      if (client.readyState === WebSocket.OPEN) client.send(data);
    }
  }

  async stop(): Promise<void> {
    for (const client of this.clients) client.close();
    this.clients.clear();
    this.wss?.close(); this.wss = null;
    await this.wa?.disconnect(); this.wa = null;
  }
}
