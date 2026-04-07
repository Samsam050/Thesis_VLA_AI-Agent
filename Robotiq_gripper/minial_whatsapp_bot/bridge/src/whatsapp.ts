/**
 * WhatsApp client wrapper using Baileys.
 * Based on nanobot's implementation (https://github.com/HKUDS/nanobot).
 */
/* eslint-disable @typescript-eslint/no-explicit-any */
import makeWASocket, {
  DisconnectReason,
  useMultiFileAuthState,
  fetchLatestBaileysVersion,
  makeCacheableSignalKeyStore,
} from '@whiskeysockets/baileys';
import { Boom } from '@hapi/boom';
import qrcode from 'qrcode-terminal';
import pino from 'pino';

const VERSION = '0.1.0';

export interface InboundMessage {
  id: string;
  sender: string;
  pn: string;
  content: string;
  timestamp: number;
  isGroup: boolean;
}

export interface WhatsAppClientOptions {
  authDir: string;
  onMessage: (msg: InboundMessage) => void;
  onQR: (qr: string) => void;
  onStatus: (status: string) => void;
}

export class WhatsAppClient {
  private sock: any = null;
  private options: WhatsAppClientOptions;
  private reconnecting = false;
  private sentByBot: Map<string, number> = new Map();

  constructor(options: WhatsAppClientOptions) {
    this.options = options;
  }

  async connect(): Promise<void> {
    const logger = pino({ level: 'silent' });
    const { state, saveCreds } = await useMultiFileAuthState(this.options.authDir);
    const { version } = await fetchLatestBaileysVersion();
    console.log(`Using Baileys version: ${version.join('.')}`);

    this.sock = makeWASocket({
      auth: {
        creds: state.creds,
        keys: makeCacheableSignalKeyStore(state.keys, logger),
      },
      version,
      logger,
      printQRInTerminal: false,
      browser: ['mybot', 'cli', VERSION],
      syncFullHistory: false,
      markOnlineOnConnect: false,
    });

    if (this.sock.ws && typeof this.sock.ws.on === 'function') {
      this.sock.ws.on('error', (err: Error) => {
        console.error('WebSocket error:', err.message);
      });
    }

    this.sock.ev.on('connection.update', async (update: any) => {
      const { connection, lastDisconnect, qr } = update;

      if (qr) {
        console.log('\nScan this QR code with WhatsApp (Linked Devices):\n');
        qrcode.generate(qr, { small: true });
        this.options.onQR(qr);
      }

      if (connection === 'close') {
        const statusCode = (lastDisconnect?.error as Boom)?.output?.statusCode;
        const shouldReconnect = statusCode !== DisconnectReason.loggedOut;
        console.log(`Connection closed. Status: ${statusCode}, Will reconnect: ${shouldReconnect}`);
        this.options.onStatus('disconnected');

        if (shouldReconnect && !this.reconnecting) {
          this.reconnecting = true;
          console.log('Reconnecting in 5 seconds...');
          setTimeout(() => { this.reconnecting = false; this.connect(); }, 5000);
        }
      } else if (connection === 'open') {
        console.log('Connected to WhatsApp');
        this.options.onStatus('connected');
      }
    });

    this.sock.ev.on('creds.update', saveCreds);

    this.sock.ev.on('messages.upsert', async ({ messages, type }: { messages: any[]; type: string }) => {
      if (type !== 'notify') return;
      this.pruneSentByBot();
      for (const msg of messages) {
        if (msg.key.fromMe && this.wasSentByBot(msg.key.id || '')) continue;
        if (msg.key.remoteJid === 'status@broadcast') continue;
        const content = this.extractContent(msg);
        if (!content) continue;
        this.options.onMessage({
          id: msg.key.id || '',
          sender: msg.key.remoteJid || '',
          pn: msg.key.remoteJidAlt || '',
          content,
          timestamp: msg.messageTimestamp as number,
          isGroup: msg.key.remoteJid?.endsWith('@g.us') || false,
        });
      }
    });
  }

  private extractContent(msg: any): string | null {
    const m = msg.message;
    if (!m) return null;
    if (m.conversation) return m.conversation;
    if (m.extendedTextMessage?.text) return m.extendedTextMessage.text;
    if (m.imageMessage?.caption) return `[Image] ${m.imageMessage.caption}`;
    if (m.videoMessage?.caption) return `[Video] ${m.videoMessage.caption}`;
    if (m.documentMessage?.caption) return `[Document] ${m.documentMessage.caption}`;
    if (m.audioMessage) return '[Voice Message]';
    return null;
  }

  async sendMessage(to: string, text: string): Promise<void> {
    if (!this.sock) throw new Error('Not connected');
    const result = await this.sock.sendMessage(to, { text });
    const id = result?.key?.id;
    if (id) this.sentByBot.set(id, Date.now());
  }

  async sendImage(to: string, imageBase64: string, caption?: string): Promise<void> {
    if (!this.sock) throw new Error('Not connected');
    const buffer = Buffer.from(imageBase64, 'base64');
    const result = await this.sock.sendMessage(to, { image: buffer, caption: caption ?? '' });
    const id = result?.key?.id;
    if (id) this.sentByBot.set(id, Date.now());
  }

  async disconnect(): Promise<void> {
    if (this.sock) { this.sock.end(undefined); this.sock = null; }
  }

  private wasSentByBot(id: string): boolean {
    if (!id) return false;
    return this.sentByBot.delete(id);
  }

  private pruneSentByBot(): void {
    const cutoff = Date.now() - 5 * 60 * 1000;
    for (const [id, ts] of this.sentByBot) {
      if (ts < cutoff) this.sentByBot.delete(id);
    }
  }
}
