/* ============================================================================
 * Otto — browser client (vanilla ES module, no build step, no dependencies).
 *
 * Pieces
 *   Capture   microphone -> AudioWorklet (16 kHz mono Int16) -> energy VAD
 *   Player    server PCM16 (AUD1 frames) -> gapless scheduled AudioBuffers
 *   Socket    /ws JSON control + binary audio, exponential-backoff reconnect
 *   UI        transcript, tool/job/subagent/confirmation cards, settings sheet, orb
 *   Account   cookie session (a 401 or a 4401 socket close means /login), account menu
 *   Connects  Google OAuth + Slack token rows in Settings > Connections
 *
 * Manual smoke test (when no headless browser is handy)
 *   1. ./.venv/bin/python -m uvicorn otto.main:app --port 8800
 *   2. Open http://127.0.0.1:8800 — header pill must read "Connected" and the
 *      Voice/LLM badges must reflect /api/health.
 *   3. Console: __ottoDev.injectAssistantText('Hello there, I am Otto.')
 *              __ottoDev.simulateTool({ name: 'gmail_search',
 *                args: { query: 'from:sarah' },
 *                display: { kind: 'email_list', query: 'from:sarah', items: [
 *                  { from: 'sarah@acme.com', subject: 'Q3 planning', date: '2025-05-01T09:00:00Z',
 *                    snippet: 'Can we move the sync?', unread: true } ] } })
 *              __ottoDev.simulateConfirm({ tool: 'gmail_send', name: 'Send email',
 *                summary: 'Send the draft to sarah@acme.com?', danger: true,
 *                preview: { kind: 'email_send', to: ['sarah@acme.com'],
 *                           subject: 'Q3 planning', body: 'Sounds good.' } })
 *              __ottoDev.simulateAudio({ seconds: 2 })   // exercises playback + playback.idle
 *              __ottoDev.simulateMicLevel(0.05, 400)     // drive the VAD with no microphone
 *              __ottoDev.simulateMicLevel(0.002, 600)    // ... then a pause that ends the turn
 *              __ottoDev.simulateSubagents()      // subagents tool card
 *              __ottoDev.simulateJobEvent(1)      // queued -> running -> done job card
 *              __ottoDev.simulateJob({ id: 'job_x', status: 'running', title: 'Reading thread' })
 *              __ottoDev.simulateConfirm({ preview: { kind: 'event_update', id: 'evt_1',
 *                title: 'Q3 planning', start: '2025-05-02T15:00:00Z',
 *                end: '2025-05-02T15:30:00Z', location: 'Room 2' } })
 *      Approve the card with the Y key; it must disable and read "approved".
 *      Job card: one card per id, Cancel disables itself, and the activity strip above
 *      the composer shows "Running N · title" and disappears once every job settles.
 *   4. Click the mic button, allow the device, speak: partial captions appear,
 *      the utterance ends only after the configured silence and the reply streams in.
 *      A pause inside a sentence must not end it (endpointing is hysteresis-based: silence
 *      is level well under the one that opened the turn), and a blip under 250 ms is dropped.
 *      Speak again while it talks — sustained speech over the "Interrupt after" window stops
 *      playback (barge-in); a cough or a keystroke must not.
 *      With Debug on a multi-sentence reply logs "segment join" with a gap near 0 ms,
 *      and the joins must be inaudible (no click, no gap).
 *   5. Space (hold) = push-to-talk, Enter in the text box = send, Esc = stop.
 *   6. Settings: change model/voice/personality, reload — values must persist
 *      (localStorage + POST /api/config) and be re-applied on reconnect
 *      ({type:'config'} after every "ready").
 *   7. Settings > Capabilities: both toggles ride along in every config patch.
 *      "Background model" is disabled and reads "not offered for this model" when
 *      /api/bootstrap reports capabilities.background_model.available === false,
 *      and background_model is then always sent as false.
 *   8. Account: the topbar circle shows the signed-in initials and opens a menu with
 *      the email, Settings, Connections and Sign out. Sign out posts
 *      /api/auth/logout and lands on /login; a 401 on any fetch or a socket close
 *      with code 4401 lands there too, without retrying the socket.
 *   9. Settings > Connections: shows the Google/Slack state from /api/bootstrap,
 *      "Connect Google" navigates to /oauth/google/start (full page), Disconnect
 *      DELETEs, Slack saves a write-only bot token. Returning with ?connected=google
 *      toasts and refreshes bootstrap; ?error=google_* toasts quietly.
 *  10. End (topbar), "End conversation" in the account menu and Shift+Esc all stop the
 *      turn, close the socket without reconnecting and land on /dashboard. The actions
 *      badge left of it mirrors bootstrap.actions.open and follows
 *      {"type":"action.created"}; the badge and that toast's Open button go to /dashboard.
 * ========================================================================== */

const TARGET_RATE = 16000; // browser -> server PCM sample rate
const CAPTURE_CHUNK = 512; // 32 ms @ 16 kHz
const ONSET_MS = 140; // unbroken voiced audio before an utterance opens
const MIN_UTTERANCE_MS = 250; // blips shorter than this are dropped
const PREROLL_MS = 320; // audio kept before onset so the first word is intact
const ENDPOINT_SILENCE_RATIO = 0.55; // "silence" is below this share of the threshold that opened the turn
const ENDPOINT_REARM_MS = 40; // speech this long after a dip restarts the end-of-speech clock
const BARGE_IN_MULTIPLIER = 1.6; // barge-in needs this much more level than a normal turn
const BARGE_IN_GUARD_MS = 300; // ignore the first 300 ms of playback: Otto's own tail reaches the mic
const ENDPOINT_FLOOR_RATIO = 1.25; // ...and never "silent" until we are this close to the room's floor
const MAX_UTTERANCE_MS = 120000; // dictation length — a turn always ends, even if the levels never settle
const SILENCE_GROWTH_MS_PER_S = 70; // every second spoken grows the pause budget by this…
const SILENCE_BUDGET_MAX_MS = 2600; // …but never past here — a real stop must still end the turn
const NOISE_FLOOR_FREEZE_MS = 300; // the room is not re-learned for this long after a turn closes
const PLAYBACK_TAIL_FREEZE_MS = 350; // nor for this long after Otto's playback stops
const NOISE_FLOOR_MIN = 0.0006;
const NOISE_FLOOR_MAX = 0.02; // above this the pickup threshold would drown out speech
const SENSITIVITY_MULT_MAX = 4.5; // least sensitive end of the slider ("noisy room")
const SENSITIVITY_MULT_MIN = 1.9; // most sensitive end ("quiet room")
const SENSITIVITY_ABS_CEIL = 0.012; // absolute level floor, least sensitive end
const SENSITIVITY_ABS_FLOOR = 0.005; // absolute level floor, most sensitive end
const DRAIN_GRACE_MS = 260; // wait this long after drain before playback.idle
const PING_INTERVAL_MS = 25000;
const HEALTH_INTERVAL_MS = 30000;
const SETTINGS_KEY = 'otto.settings.v1';
const AUDIO_HEADER_BYTES = 16;
const AUDIO_MAGIC = [0x41, 0x55, 0x44, 0x31]; // "AUD1"
const AUDIO_FADE_S = 0.006; // 6 ms gain envelope at each segment edge (no click on a join)
const AUDIO_JOIN_WINDOW_S = 0.25; // chain onto the previous buffer while it is this fresh
const AUDIO_START_LEAD_S = 0.02; // the first buffer of a turn starts slightly ahead of now
const AUDIO_TRIM_THRESHOLD = 0.005; // |sample| below this counts as silence
const AUDIO_TRIM_MIN_SILENCE_S = 0.03; // only trim a lead-in of at least 30 ms
const AUDIO_TRIM_MARGIN_S = 0.01; // keep 10 ms of it before the first audible sample
const AUDIO_TRIM_MAX_S = 0.25; // never drop more than 250 ms

/* ------------------------------------------------------------------ utils -- */

function $(id) {
  return document.getElementById(id);
}

function clamp(value, lo, hi) {
  return value < lo ? lo : value > hi ? hi : value;
}

function num(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function text(value) {
  return value == null ? '' : String(value);
}

function clear(node) {
  while (node && node.firstChild) node.removeChild(node.firstChild);
}

function safeJson(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch (err) {
    return String(value);
  }
}

function el(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  if (options.class) node.className = options.class;
  if (options.text != null) node.textContent = String(options.text);
  if (options.attrs) {
    for (const [key, value] of Object.entries(options.attrs)) {
      if (value == null || value === false) continue;
      node.setAttribute(key, value === true ? '' : String(value));
    }
  }
  appendChildren(node, children);
  return node;
}

function appendChildren(node, children) {
  for (const child of children) {
    if (child == null || child === false) continue;
    if (Array.isArray(child)) appendChildren(node, child);
    else if (child instanceof Node) node.append(child);
    else node.append(document.createTextNode(String(child)));
  }
}

function isTypingTarget(node) {
  if (!node || !node.tagName) return false;
  const tag = node.tagName.toLowerCase();
  return tag === 'input' || tag === 'textarea' || tag === 'select' || node.isContentEditable === true;
}

function arrayOf(value) {
  return Array.isArray(value) ? value : [];
}

function joinList(value) {
  if (value == null) return '';
  const items = Array.isArray(value) ? value : [value];
  return items
    .map((item) => {
      if (item == null) return '';
      if (typeof item === 'object') {
        return text(item.name || item.displayName || item.email || item.id || safeJson(item));
      }
      return String(item);
    })
    .filter(Boolean)
    .join(', ');
}

function formatWhen(value) {
  if (value == null || value === '') return '';
  let raw = value;
  if (typeof raw === 'object') raw = raw.dateTime || raw.date || '';
  const source = text(raw);
  if (!source) return '';
  const parsed = new Date(source);
  if (Number.isNaN(parsed.getTime())) {
    return source.includes('T') ? source.slice(0, 16).replace('T', ' ') : source;
  }
  const sameYear = parsed.getFullYear() === new Date().getFullYear();
  const date = parsed.toLocaleDateString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    ...(sameYear ? {} : { year: 'numeric' }),
  });
  const hasTime = source.includes('T') && !/^\d{4}-\d{2}-\d{2}$/.test(source);
  if (!hasTime) return date;
  return `${date} · ${parsed.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}`;
}

/* --------------------------------------------------------------- settings -- */

const DEFAULT_SETTINGS = Object.freeze({
  model: '',
  thinking: '',
  voice: '',
  personality: '',
  instruct: '',
  sensitivity: 55,
  silenceMs: 550,
  interruptMs: 250,
  deviceId: '',
  sendTranscriptText: false,
  debug: false,
  parallelAgents: false,
  backgroundModel: false,
});

function loadSettings() {
  const merged = { ...DEFAULT_SETTINGS };
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return merged;
    const saved = JSON.parse(raw);
    if (!saved || typeof saved !== 'object') return merged;
    for (const key of Object.keys(DEFAULT_SETTINGS)) {
      if (saved[key] == null) continue;
      if (typeof DEFAULT_SETTINGS[key] === 'boolean') merged[key] = !!saved[key];
      else if (typeof DEFAULT_SETTINGS[key] === 'number') merged[key] = num(saved[key], DEFAULT_SETTINGS[key]);
      else merged[key] = String(saved[key]);
    }
  } catch (err) {
    console.warn('[otto] could not read saved settings', err);
  }
  return merged;
}

const settings = loadSettings();

// True once the user moves the end-of-speech slider, so "ready" stops overriding it.
let localSilenceOverride = false;

function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch (err) {
    console.warn('[otto] could not persist settings', err);
  }
}

function silenceMs() {
  return clamp(num(settings.silenceMs, 550), 200, 2000);
}

function interruptMs() {
  return clamp(num(settings.interruptMs, 250), 150, 600);
}

/* ------------------------------------------------------------- dom handles -- */

const appEl = $('app');
const mainEl = $('main');
const orbEl = $('orb');
const orbCanvas = $('orb-canvas');
const orbHit = $('orb-hit');
const stateLabel = $('state-label');
const stateDetail = $('state-detail');
const liveCaption = $('live-caption');
const transcriptEl = $('transcript');
const logEmpty = $('log-empty');
const metricsEl = $('metrics');
const connPill = $('conn-pill');
const connLabel = $('conn-label');
const badgeModel = $('badge-model');
const badgeLlm = $('badge-llm');
const badgeTts = $('badge-tts');
const micBtn = $('mic-btn');
const meterEl = $('meter');
const textInput = $('text-input');
const stopBtn = $('stop-btn');
const drawerEl = $('drawer');
const scrimEl = $('scrim');
const toastsEl = $('toasts');
const infoSession = $('info-session');
const infoVersion = $('info-version');
const infoMic = $('info-mic');
const activityStrip = $('activity-strip');
const activityText = $('activity-text');
const activityUnseen = $('activity-unseen');
const composerEl = $('composer');
const sendBtn = $('send-btn');
const brandDot = $('brand-dot');
const accountBtn = $('account-btn');
const accountMenu = $('account-menu');
const accountInitials = $('account-initials');
const accountEmail = $('account-email');
const actionsBadge = $('actions-badge');
const actionsBadgeLabel = $('actions-badge-label');
const endBtn = $('end-btn');
const sheetInitials = $('sheet-initials');
const sheetName = $('sheet-name');
const sheetEmail = $('sheet-email');
const googleBadge = $('google-badge');
const googleConnect = $('google-connect');
const googleDisconnect = $('google-disconnect');
const googleConfigNote = $('google-config-note');
const googleErrorNote = $('google-error-note');
const mcpBadge = $('mcp-badge');
const mcpNote = $('mcp-note');
const mcpDetail = $('mcp-detail');
const mcpCheckBtn = $('mcp-check');
const mcpTestBtn = $('mcp-test');
const googleRedirectUri = $('google-redirect-uri');
const googleRedirectCopy = $('google-redirect-copy');
const slackBadge = $('slack-badge');
const slackToken = $('slack-token');
const slackSaveBtn = $('slack-save');
const slackRemoveBtn = $('slack-remove');

/* ------------------------------------------------------------------ state -- */

const app = {
  connected: false,
  connState: 'connecting',
  phase: 'idle',
  detail: '',
  sessionId: '',
  serverConfig: {},
  models: [],
  voices: [],
  personalities: [],
  muted: false,
  micLevel: 0,
  utteranceOpen: false,
  // VAD decisions since the page loaded: turns opened/closed, barge-ins, dropped blips.
  vadStats: { opened: 0, closed: 0, bargeIns: 0, rejectedBlips: 0 },
  metrics: {},
  pendingConfirm: null,
  ttsDown: false,
  capabilities: { parallelAgents: false, backgroundModel: { available: false, id: '' } },
  mcp: { active: false, tools: 0, servers: [], notices: [], mode: '', authoritative: false },
  user: null,
  sessionEnded: false,
};

// Mirrors GET /api/connections (and /api/bootstrap.connections). Tokens are write-only:
// the server never sends a stored Slack token back, so there is nothing to render.
const connections = {
  google: { configured: false, connected: false, email: '', scopes: [] },
  slack: { configured: false, team: null },
};

const CONNECTION_SECTIONS = {
  connections: 'section-connections',
  account: 'section-account',
  settings: 'section-conversation',
};

const assistantMessages = new Map(); // assistant message id -> {node, body, meta, buffer}
const toolCards = new Map(); // tool call id -> {card, ms, body}
const confirmCards = new Map(); // confirmation id -> {card, approve, deny}
const jobCards = new Map(); // job id -> {card, state, ms, result, cancel, ...}
const seenNotices = new Set();

// Background job bookkeeping for the activity strip: how many jobs settled since the strip
// was last clicked (that count clears when the strip itself is clicked).
const jobActivity = { unseen: 0 };

let socketws = null;
let wsAttempt = 0;
let wsRetryTimer = 0;
let wsPingTimer = 0;
let healthTimer = 0;
let configPostTimer = 0;
let devSeq = 0;

/* ------------------------------------------------------------------ toasts -- */

const toastTimes = new Map();

/**
 * Transient message under the composer. `options.actionLabel` + `options.onAction` add one
 * inline button (action.created uses it to jump to the dashboard) and keep the toast up longer.
 */
function toast(message, level = 'info', options = {}) {
  const body = text(message);
  if (!body) return;
  const now = Date.now();
  const last = toastTimes.get(body) || 0;
  if (now - last < 3000) return;
  toastTimes.set(body, now);

  const actionLabel = text(options.actionLabel);
  const node = el('div', { class: 'toast', attrs: { 'data-level': level } }, [body]);
  let life = 4200;
  if (actionLabel) {
    node.classList.add('has-action');
    life = 6500; // long enough to actually reach the button
    const action = el('button', { class: 'toast-action', text: actionLabel, attrs: { type: 'button' } });
    action.addEventListener('click', () => {
      node.remove();
      if (typeof options.onAction === 'function') options.onAction();
    });
    node.append(action);
  }
  toastsEl.append(node);
  window.setTimeout(() => {
    node.classList.add('leaving');
    window.setTimeout(() => node.remove(), 220);
  }, life);
  while (toastsEl.children.length > 4) toastsEl.firstElementChild.remove();
}

function safe(fn, label) {
  try {
    return fn();
  } catch (err) {
    console.error(`[otto] ${label} failed`, err);
    toast(`${label} failed: ${err && err.message ? err.message : err}`, 'error');
    return undefined;
  }
}

/* --------------------------------------------------------- account/session -- */

/**
 * The cookie session is gone (401 from an API call, or a socket close with code 4401).
 * Go to the login page once, without retrying anything.
 */
function endSession() {
  if (app.sessionEnded) return;
  app.sessionEnded = true;
  socketws = null;
  app.connected = false;
  window.clearTimeout(wsRetryTimer);
  window.clearInterval(wsPingTimer);
  window.clearInterval(healthTimer);
  location.assign('/login');
}

/** fetch() plus the session guard: every API call in this file goes through it. */
async function authFetch(input, init) {
  const response = await fetch(input, init);
  if (response.status === 401) {
    endSession();
    throw new Error('session ended');
  }
  return response;
}

function initialsFor(user) {
  const name = text(user && user.name).trim();
  const emailAddress = text(user && user.email).trim();
  if (name) {
    const words = name.split(/\s+/).filter(Boolean);
    const first = words[0].charAt(0);
    const last = words.length > 1 ? words[words.length - 1].charAt(0) : '';
    return (first + last).toUpperCase();
  }
  return emailAddress ? emailAddress.charAt(0).toUpperCase() : '·';
}

function paintUser(user) {
  if (user && typeof user === 'object') {
    app.user = { id: text(user.id), email: text(user.email), name: text(user.name) };
  }
  const current = app.user || {};
  const initials = initialsFor(current);
  const emailAddress = current.email || '—';
  const displayName = current.name || (current.email ? current.email.split('@')[0] : '—');
  accountInitials.textContent = initials;
  sheetInitials.textContent = initials;
  accountEmail.textContent = emailAddress;
  accountEmail.title = emailAddress;
  sheetEmail.textContent = emailAddress;
  sheetName.textContent = displayName;
}

async function signOut() {
  setAccountMenu(false);
  try {
    await fetch('/api/auth/logout', { method: 'POST', headers: { Accept: 'application/json' } });
  } catch (err) {
    console.warn('[otto] logout request failed, leaving anyway', err);
  }
  endSession();
}

function setAccountMenu(open) {
  accountMenu.hidden = !open;
  accountBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
}

function onAccountButtonClick() {
  const next = accountMenu.hidden;
  setAccountMenu(next);
  if (next) {
    const first = accountMenu.querySelector('.menu-item');
    if (first) first.focus({ preventScroll: true });
  }
}

/**
 * End (topbar), the account menu and Shift+Esc: stop the turn, close the socket for good and
 * hand the user to /dashboard, where whatever Otto produced during the call is waiting.
 * app.sessionEnded is the flag connectSocket()/scheduleReconnect() already honour as
 * "this session is over, never reconnect".
 */
function endConversation() {
  if (app.sessionEnded) return;
  app.sessionEnded = true;
  setAccountMenu(false);
  stopEverything(); // flushes playback and sends cancel while the socket is still open
  window.clearTimeout(wsRetryTimer);
  window.clearInterval(wsPingTimer);
  window.clearInterval(healthTimer);
  const ws = socketws;
  socketws = null; // the close handler bails out when it no longer owns the socket
  if (ws) {
    try {
      ws.close(1000, 'conversation ended');
    } catch (err) {
      /* already closing */
    }
  }
  location.assign('/dashboard');
}

/* ----------------------------------------------------------------- actions -- */

const ACTION_TOAST_LABEL = { draft: 'Draft ready', email: 'Draft ready', event: 'Event ready' };

/** Quiet pill next to the avatar: how many things wait on the dashboard. Hidden at zero. */
function paintActionsBadge(count) {
  const open = Math.max(0, Math.round(num(count)));
  actionsBadge.hidden = open === 0;
  actionsBadgeLabel.textContent = `${open} to finish`;
  actionsBadge.setAttribute('aria-label', `Actions required: ${open} to finish`);
}

/** A tool persisted something for the user: {"type":"action.created", open, action}. */
function onActionCreated(message) {
  paintActionsBadge(message.open);
  const action = message.action && typeof message.action === 'object' ? message.action : {};
  const lead = ACTION_TOAST_LABEL[text(action.kind)] || 'Action ready';
  const title = text(action.title).trim() || 'a new item';
  toast(`${lead} — “${title}” is in your actions`, 'ok', {
    actionLabel: 'Open',
    onAction: () => location.assign('/dashboard'),
  });
}

/* ---------------------------------------------------------------- playback -- */

// Calm, warm-neutral palette: one glow colour and one core colour per state.
const PALETTES = {
  idle: { glow: '150, 160, 186', core: '96, 106, 128' },
  offline: { glow: '122, 126, 136', core: '68, 72, 80' },
  listening: { glow: '116, 208, 180', core: '66, 146, 128' },
  thinking: { glow: '160, 150, 235', core: '100, 92, 168' },
  speaking: { glow: '140, 172, 235', core: '88, 120, 182' },
  awaiting_confirmation: { glow: '226, 196, 130', core: '146, 122, 78' },
};

const PLAYER_LEAD_TRIM_THRESHOLD_INT16 = AUDIO_TRIM_THRESHOLD * 32768;

/**
 * Defensive lead-silence trim for the first buffer of a segment: synthesis can still pad a
 * merged sentence, and a dead patch pushes the join away from its neighbours. Returns how many
 * samples to drop, keeping AUDIO_TRIM_MARGIN_S of the silence in front of the first audible one.
 * A buffer that is silent all the way through is left untouched (the next one may carry the words).
 */
function leadTrimSamples(pcm, rate) {
  const minSilence = Math.round(AUDIO_TRIM_MIN_SILENCE_S * rate);
  let first = 0;
  while (first < pcm.length && Math.abs(pcm[first]) < PLAYER_LEAD_TRIM_THRESHOLD_INT16) first += 1;
  if (first < minSilence || first >= pcm.length) return 0;
  const margin = Math.round(AUDIO_TRIM_MARGIN_S * rate);
  return clamp(first - margin, 0, Math.round(AUDIO_TRIM_MAX_S * rate));
}

class Player {
  constructor() {
    this.ctx = null;
    this.gain = null;
    this.analyser = null;
    this.levelData = null;
    this.playhead = 0;
    this.activeSince = 0; // performance.now() of the moment playback went from idle to busy
    this.scheduled = [];
    this.openSegments = new Set();
    this.seqHigh = 0;
    this.minSeq = 1;
    this.idleSentFor = 0;
    this.drainTimer = 0;
    this.gen = 0;
    this.outputLevel = 0;
    this.warnedMagic = false;
    this.onDrained = null;
    this.lastSeq = 0; // last segment whose buffers were scheduled (for the join log)
    this.joins = 0; // segment joins inside the current turn (debug)
    this.lastTrimSamples = 0; // lead-silence samples dropped from the last first-buffer
  }

  ensureContext() {
    if (this.ctx) return this.ctx;
    const Ctor = window.AudioContext || window.webkitAudioContext;
    this.ctx = new Ctor({ latencyHint: 'interactive' });
    this.gain = this.ctx.createGain();
    this.gain.gain.value = 1;
    this.analyser = this.ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.levelData = new Float32Array(this.analyser.fftSize);
    this.gain.connect(this.analyser);
    this.analyser.connect(this.ctx.destination);
    return this.ctx;
  }

  unlock() {
    if (this.ctx && this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
  }

  isPlaying() {
    return this.scheduled.length > 0 || this.openSegments.size > 0;
  }

  /** How long playback has been busy; Infinity while nothing is playing (no guard in the way). */
  playedForMs() {
    return this.isPlaying() ? performance.now() - this.activeSince : Infinity;
  }

  handleStart(seq, sampleRate) {
    const id = Math.round(num(seq, 0));
    if (id <= 0) return;
    this.seqHigh = Math.max(this.seqHigh, id);
    if (id < this.minSeq) return; // segment was cancelled before it started
    if (!this.isPlaying()) this.activeSince = performance.now();
    this.openSegments.add(id);
    if (settings.debug) console.debug('[otto] audio.start', { seq: id, sampleRate });
  }

  handleFrame(buffer) {
    if (!(buffer instanceof ArrayBuffer) || buffer.byteLength <= AUDIO_HEADER_BYTES) return;
    const magic = new Uint8Array(buffer, 0, 4);
    if (
      magic[0] !== AUDIO_MAGIC[0] ||
      magic[1] !== AUDIO_MAGIC[1] ||
      magic[2] !== AUDIO_MAGIC[2] ||
      magic[3] !== AUDIO_MAGIC[3]
    ) {
      if (!this.warnedMagic) {
        this.warnedMagic = true;
        console.warn('[otto] ignoring binary frame without the AUD1 magic');
      }
      return;
    }
    const view = new DataView(buffer);
    const seq = view.getUint32(4, true);
    const rate = view.getUint32(8, true) || 24000;
    const index = view.getUint32(12, true);
    const frames = (buffer.byteLength - AUDIO_HEADER_BYTES) >> 1;
    if (frames <= 0) return;
    this.seqHigh = Math.max(this.seqHigh, seq);
    if (seq < this.minSeq) return;
    if (!this.openSegments.has(seq)) return; // no audio.start for this segment
    if (settings.debug) console.debug('[otto] audio chunk', { seq, index, rate, frames });
    this.enqueue(new Int16Array(buffer, AUDIO_HEADER_BYTES, frames), rate, seq, index);
  }

  enqueue(pcm, rate, seq, index = 0) {
    const ctx = this.ensureContext();
    if (ctx.state === 'suspended') ctx.resume().catch(() => {});
    // Only the first buffer of a segment can carry a lead-in left by synthesis.
    const trim = index === 0 ? leadTrimSamples(pcm, rate) : 0;
    if (trim) {
      this.lastTrimSamples = trim;
      if (settings.debug) console.debug('[otto] trimmed lead silence', { seq, ms: Math.round((trim / rate) * 1000) });
    }
    const frames = pcm.length - trim;
    if (frames <= 0) return;
    const audioBuffer = ctx.createBuffer(1, frames, rate);
    const channel = audioBuffer.getChannelData(0);
    for (let i = 0; i < frames; i++) channel[i] = pcm[i + trim] / 32768;

    const now = ctx.currentTime;
    // Between the sentences of one turn the previous buffer's end is the next buffer's
    // start: chain onto it with no re-priming lead, so the join carries no extra gap.
    const chaining = this.playhead > 0 && this.playhead >= now - AUDIO_JOIN_WINDOW_S;
    const when = chaining ? Math.max(this.playhead, now) : now + AUDIO_START_LEAD_S;
    if (settings.debug && chaining && this.lastSeq && this.lastSeq !== seq) {
      this.joins += 1;
      console.debug('[otto] segment join', {
        joins: this.joins,
        from: this.lastSeq,
        to: seq,
        gapMs: Math.round((when - this.playhead) * 1000),
      });
    }
    this.lastSeq = seq;

    const source = ctx.createBufferSource();
    source.buffer = audioBuffer;
    // A per-buffer gain node carries the 6 ms edge envelope; mid-segment buffers stay flat.
    const envelope = ctx.createGain();
    envelope.gain.value = 1;
    if (index === 0) {
      envelope.gain.setValueAtTime(0, when);
      envelope.gain.linearRampToValueAtTime(1, when + Math.min(AUDIO_FADE_S, audioBuffer.duration / 2));
    }
    source.connect(envelope);
    envelope.connect(this.gain);
    const entry = { seq, src: source, envelope, end: when + audioBuffer.duration };

    const gen = this.gen;
    source.onended = () => {
      if (gen !== this.gen) return;
      this.release(entry);
      this.maybeDrained();
    };
    this.scheduled.push(entry);
    try {
      source.start(when);
    } catch (err) {
      console.warn('[otto] could not start audio source', err);
      this.release(entry);
      return;
    }
    this.playhead = entry.end;
  }

  release(entry, stop = false) {
    const at = this.scheduled.indexOf(entry);
    if (at >= 0) this.scheduled.splice(at, 1);
    if (stop) {
      try {
        entry.src.stop();
      } catch (err) {
        /* already stopped */
      }
    }
    for (const node of [entry.src, entry.envelope]) {
      try {
        if (node) node.disconnect();
      } catch (err) {
        /* already disconnected */
      }
    }
  }

  handleEnd(seq) {
    const id = Math.round(num(seq, 0));
    this.fadeOutSegmentTail(id);
    this.openSegments.delete(id);
    this.maybeDrained();
  }

  /** Fade the last buffer of a finished segment out so the next segment joins silently. */
  fadeOutSegmentTail(seq) {
    const ctx = this.ctx;
    if (!ctx) return;
    let last = null;
    for (const entry of this.scheduled) {
      if (entry.seq !== seq) continue;
      if (!last || entry.end > last.end) last = entry;
    }
    if (!last || last.faded) return;
    const fadeStart = last.end - AUDIO_FADE_S;
    if (fadeStart <= ctx.currentTime) return; // already audible: leave the tail alone
    last.faded = true;
    last.envelope.gain.setValueAtTime(1, fadeStart);
    last.envelope.gain.linearRampToValueAtTime(0, last.end);
  }

  maybeDrained() {
    if (this.drainTimer) return;
    if (this.scheduled.length || this.openSegments.size) return;
    if (!this.seqHigh || this.idleSentFor >= this.seqHigh) return;
    const gen = this.gen;
    this.drainTimer = window.setTimeout(() => {
      this.drainTimer = 0;
      if (gen !== this.gen) return;
      if (this.scheduled.length || this.openSegments.size) return;
      if (this.idleSentFor >= this.seqHigh) return;
      this.idleSentFor = this.seqHigh;
      this.resetPlayhead(); // the turn is over: the next turn starts a fresh chain
      if (this.onDrained) this.onDrained(this.seqHigh);
    }, DRAIN_GRACE_MS);
  }

  /** cancel.audio {fromSeq}: drop every queued/scheduled segment with seq >= fromSeq. */
  cancelFrom(fromSeq) {
    const cutoff = Math.max(1, Math.round(num(fromSeq, 1)));
    this.minSeq = Math.max(this.minSeq, cutoff);
    this.gen += 1;
    for (const entry of [...this.scheduled]) {
      if (entry.seq >= cutoff) this.release(entry, true);
    }
    for (const seq of [...this.openSegments]) {
      if (seq >= cutoff) this.openSegments.delete(seq);
    }
    window.clearTimeout(this.drainTimer);
    this.drainTimer = 0;
    this.resetPlayhead();
    if (settings.debug) console.debug('[otto] cancel.audio', { fromSeq: cutoff });
  }

  /** Local barge-in flush: silence everything immediately, ignore stale segments. */
  flush() {
    this.gen += 1;
    for (const entry of [...this.scheduled]) this.release(entry, true);
    this.openSegments.clear();
    this.minSeq = Math.max(this.minSeq, this.seqHigh + 1);
    this.idleSentFor = this.seqHigh;
    window.clearTimeout(this.drainTimer);
    this.drainTimer = 0;
    this.resetPlayhead();
  }

  /** Break the chain: the next buffer is scheduled fresh instead of butted onto the last. */
  resetPlayhead() {
    this.playhead = 0;
    this.lastSeq = 0;
    this.joins = 0;
  }

  readLevel() {
    if (!this.analyser) return 0;
    this.analyser.getFloatTimeDomainData(this.levelData);
    let sum = 0;
    for (let i = 0; i < this.levelData.length; i++) {
      const value = this.levelData[i];
      sum += value * value;
    }
    const rms = Math.sqrt(sum / this.levelData.length);
    const target = clamp(Math.pow(rms / 0.06, 0.75), 0, 1);
    this.outputLevel = this.outputLevel * 0.75 + target * 0.25;
    return this.outputLevel;
  }
}

const player = new Player();

/* ----------------------------------------------------------------- capture -- */

class Downsampler {
  constructor(inRate, outRate, chunkSamples, onChunk) {
    this.ratio = Math.max(1, inRate / outRate);
    this.chunkSamples = chunkSamples;
    this.onChunk = onChunk;
    this.acc = 0;
    this.sum = 0;
    this.count = 0;
    this.stage = new Int16Array(chunkSamples);
    this.stageLen = 0;
  }

  push(samples) {
    for (let i = 0; i < samples.length; i++) {
      this.sum += samples[i];
      this.count += 1;
      this.acc += 1;
      if (this.acc >= this.ratio) {
        const value = this.sum / this.count;
        const scaled = value > 1 ? 32767 : value < -1 ? -32768 : (value * 32767) | 0;
        this.stage[this.stageLen++] = scaled;
        this.acc -= this.ratio;
        this.sum = 0;
        this.count = 0;
        if (this.stageLen >= this.chunkSamples) this.flush();
      }
    }
  }

  flush() {
    if (!this.stageLen) return;
    const out = this.stage.slice(0, this.stageLen);
    this.stageLen = 0;
    this.onChunk(out);
  }
}

/** Dev hook helper: a capture-sized chunk whose RMS is `level` (a 180 Hz tone). */
function syntheticChunk(level, samples) {
  const out = new Int16Array(Math.max(1, samples));
  const amplitude = Math.min(1, level * Math.SQRT2);
  for (let i = 0; i < out.length; i++) {
    out[i] = Math.round(Math.sin((2 * Math.PI * 180 * i) / TARGET_RATE) * amplitude * 32767);
  }
  return out;
}

class Capture {
  constructor(handlers) {
    this.handlers = handlers || {};
    this.ctx = null;
    this.stream = null;
    this.node = null;
    this.source = null;
    this.sink = null;
    this.mode = 'idle';
    this.running = false;
    this.noiseFloor = 0.0035;
    this.wasOttoSpeaking = false;
    this.calibrating = 0;
    this.preroll = [];
    this.prerollMs = 0;
    this.open = false;
    this.openAt = 0;
    this.openThreshold = 0; // level threshold of the frame that opened the turn (endpointing reference)
    this.speechAcc = 0;
    this.voiceRun = 0; // ms of continuous re-voicing inside an open turn
    this.speechMs = 0; // total voiced ms in this turn — drives the adaptive pause budget
    this.silenceAcc = 0;
    this.bargeAcc = 0; // ms of continuous speech loud enough to interrupt Otto
    this.floorFreezeUntil = 0;
    this.level = 0;
    this.forceHold = false;
    this.asrMode = false;
  }

  async start(deviceId) {
    if (this.running) return true;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('This browser cannot capture audio (getUserMedia missing).');
    }
    const base = {
      echoCancellation: true,
      noiseSuppression: true,
      // AGC ramps the gain up in silence, which makes a quiet room look like speech and
      // is the main reason a turn never ends. Levels must stay comparable.
      autoGainControl: false,
      channelCount: 1,
    };
    const constraints = { audio: deviceId ? { ...base, deviceId: { exact: deviceId } } : base };
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (err) {
      if (deviceId && (err.name === 'OverconstrainedError' || err.name === 'NotFoundError')) {
        stream = await navigator.mediaDevices.getUserMedia({ audio: base });
      } else {
        throw err;
      }
    }
    this.stream = stream;

    let ctx;
    try {
      ctx = new AudioContext({ sampleRate: TARGET_RATE, latencyHint: 'interactive' });
    } catch (err) {
      ctx = new AudioContext({ latencyHint: 'interactive' });
    }
    this.ctx = ctx;
    if (ctx.state === 'suspended') {
      try {
        await ctx.resume();
      } catch (err) {
        /* unlock() retries on the next gesture */
      }
    }
    this.source = ctx.createMediaStreamSource(stream);
    this.sink = ctx.createGain();
    this.sink.gain.value = 0;
    this.sink.connect(ctx.destination);

    let mode = 'worklet';
    try {
      await ctx.audioWorklet.addModule('/static/audio-worklet.js');
      const node = new AudioWorkletNode(ctx, 'otto-capture', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        channelCount: 1,
        processorOptions: { targetRate: TARGET_RATE, chunkSamples: CAPTURE_CHUNK },
      });
      node.port.onmessage = (event) => {
        if (event.data instanceof Int16Array) this.onInt16(event.data);
      };
      this.source.connect(node);
      node.connect(this.sink);
      this.node = node;
    } catch (err) {
      mode = 'script';
      console.warn('[otto] AudioWorklet unavailable, using ScriptProcessorNode', err);
      const processor = ctx.createScriptProcessor(4096, 1, 1);
      const sampler = new Downsampler(ctx.sampleRate, TARGET_RATE, CAPTURE_CHUNK, (chunk) => this.onInt16(chunk));
      processor.onaudioprocess = (event) => sampler.push(event.inputBuffer.getChannelData(0));
      this.source.connect(processor);
      processor.connect(this.sink);
      this.node = processor;
    }

    this.mode = mode;
    this.running = true;
    this.forceHold = false;
    this.calibrating = 300;
    this.resetVad();
    this.preroll = [];
    this.prerollMs = 0;
    return true;
  }

  async stop() {
    this.running = false;
    this.open = false;
    this.forceHold = false;
    this.preroll = [];
    this.prerollMs = 0;
    this.level = 0;
    try {
      if (this.node && this.node.port) this.node.port.onmessage = null;
    } catch (err) {
      /* ignore */
    }
    for (const node of [this.node, this.source, this.sink]) {
      try {
        if (node) node.disconnect();
      } catch (err) {
        /* ignore */
      }
    }
    if (this.stream) {
      for (const track of this.stream.getTracks()) {
        try {
          track.stop();
        } catch (err) {
          /* ignore */
        }
      }
    }
    this.stream = null;
    this.node = null;
    this.source = null;
    this.sink = null;
    if (this.ctx) {
      const ctx = this.ctx;
      this.ctx = null;
      try {
        await ctx.close();
      } catch (err) {
        /* ignore */
      }
    }
    this.mode = 'idle';
  }

  resetVad() {
    this.open = false;
    this.openThreshold = 0;
    this.speechAcc = 0;
    this.voiceRun = 0;
    this.speechMs = 0;
    this.silenceAcc = 0;
    this.bargeAcc = 0;
  }

  threshold() {
    const sensitivity = clamp(num(settings.sensitivity, 55), 0, 100) / 100;
    const multiplier = SENSITIVITY_MULT_MAX - sensitivity * (SENSITIVITY_MULT_MAX - SENSITIVITY_MULT_MIN);
    const absoluteMin = SENSITIVITY_ABS_CEIL - sensitivity * (SENSITIVITY_ABS_CEIL - SENSITIVITY_ABS_FLOOR);
    return Math.max(this.noiseFloor * multiplier, absoluteMin);
  }

  pushPreroll(chunk, ms) {
    this.preroll.push({ chunk, ms });
    this.prerollMs += ms;
    while (this.preroll.length > 1 && this.prerollMs - this.preroll[0].ms >= PREROLL_MS) {
      this.prerollMs -= this.preroll[0].ms;
      this.preroll.shift();
    }
  }

  /** True while Otto owns the audio path (playback queued or a turn in flight). */
  ottoSpeaking() {
    return !!(this.handlers.isOttoSpeaking && this.handlers.isOttoSpeaking());
  }

  /** How long Otto's playback has been busy (Infinity when nothing is playing). */
  ottoPlayedForMs() {
    return this.handlers.ottoPlayedForMs ? this.handlers.ottoPlayedForMs() : Infinity;
  }

  onInt16(chunk) {
    if (!this.running || !chunk || !chunk.length) return;
    this.processChunk(chunk);
  }

  /** One VAD tick. Every chunk — captured or simulated — comes through here. */
  processChunk(chunk) {
    const durationMs = (chunk.length / TARGET_RATE) * 1000;
    let sum = 0;
    for (let i = 0; i < chunk.length; i++) {
      const value = chunk[i] / 32768;
      sum += value * value;
    }
    const rms = Math.sqrt(sum / chunk.length);
    const target = clamp(Math.pow(rms / 0.05, 0.7), 0, 1);
    this.level = this.level * 0.7 + target * 0.3;
    if (this.handlers.onLevel) this.handlers.onLevel(this.level);

    if (this.asrMode) return; // browser speech recognition owns the turn boundary

    this.pushPreroll(chunk, durationMs);
    const openThreshold = this.threshold();
    const voiced = rms >= openThreshold;
    // The room is never re-learned while Otto is talking: his own voice reaching
    // the mic would teach the VAD that speech is noise, inflating every threshold
    // until barge-in becomes impossible and turns close mid-word.
    const ottoSpeaking = this.ottoSpeaking();
    if (this.wasOttoSpeaking && !ottoSpeaking) {
      // playback just stopped — the speaker tail is still in the room
      this.floorFreezeUntil = Math.max(this.floorFreezeUntil, performance.now() + PLAYBACK_TAIL_FREEZE_MS);
    }
    this.wasOttoSpeaking = ottoSpeaking;
    const floorFrozen = this.open || ottoSpeaking || performance.now() < this.floorFreezeUntil;

    if (this.calibrating > 0) {
      this.calibrating -= durationMs;
      // Learn the room from the first ~300 ms, but never let a noisy start
      // push the floor high enough to hide speech.
      this.noiseFloor = clamp(Math.max(this.noiseFloor * 0.7, rms * 0.9), NOISE_FLOOR_MIN, 0.02);
    } else if (!voiced && !floorFrozen) {
      this.noiseFloor = clamp(this.noiseFloor * 0.985 + rms * 0.015, NOISE_FLOOR_MIN, NOISE_FLOOR_MAX);
    } else if (this.open && rms < openThreshold) {
      // while tracking speech: learn only from frames that are *below* the pickup level,
      // very slowly — this is what lets a turn end in a room with a steady hiss
      this.noiseFloor = clamp(this.noiseFloor * 0.995 + rms * 0.005, NOISE_FLOOR_MIN, NOISE_FLOOR_MAX);
    }

    if (!this.open) {
      if (this.ottoSpeaking()) {
        // Interrupting Otto uses its own, stricter rule: sustained speech well above the
        // pickup level, and nothing at all during the guard after playback starts.
        this.speechAcc = 0;
        this.bargeAcc = rms >= openThreshold * BARGE_IN_MULTIPLIER ? this.bargeAcc + durationMs : 0;
        if (this.bargeAcc >= interruptMs() && this.ottoPlayedForMs() >= BARGE_IN_GUARD_MS) {
          if (settings.debug) {
            console.debug(
              `[otto] vad barge: rms ${rms.toFixed(3)} >` +
                ` 1.6·thr ${(openThreshold * BARGE_IN_MULTIPLIER).toFixed(3)} for ${Math.round(this.bargeAcc)}ms`,
            );
          }
          if (this.handlers.onBargeIn) this.handlers.onBargeIn();
          this.beginSpeech();
        }
        return;
      }

      this.bargeAcc = 0;
      // The whole onset window must be voiced: one loud frame never opens the mic.
      this.speechAcc = voiced ? this.speechAcc + durationMs : 0;

      if (this.forceHold || this.speechAcc >= ONSET_MS) {
        if (settings.debug) {
          console.debug(
            `[otto] vad open: rms ${rms.toFixed(3)} > thr ${openThreshold.toFixed(3)}` +
              ` for ${Math.round(this.speechAcc)}ms`,
          );
        }
        this.beginSpeech();
      }
      return;
    }

    if (this.handlers.onChunk) this.handlers.onChunk(chunk);
    // Hysteresis: "silence" is below ENDPOINT_SILENCE_RATIO of the threshold that opened
    // the turn — but never below the live noise floor, or a hissy room would hold the
    // turn open forever (the AGC/floor trap).
    const silenceThreshold = Math.max(
      this.openThreshold * ENDPOINT_SILENCE_RATIO,
      this.noiseFloor * ENDPOINT_FLOOR_RATIO,
    );
    if (voiced) {
      this.voiceRun += durationMs;
      this.speechMs += durationMs;
      if (this.voiceRun >= ENDPOINT_REARM_MS) this.silenceAcc = 0; // speech again: restart the clock
    } else if (rms >= silenceThreshold) {
      this.voiceRun = 0;
      this.silenceAcc = 0; // a dip, not silence: keep the turn open
    } else {
      this.voiceRun = 0;
      this.silenceAcc += durationMs;
    }
    const openFor = performance.now() - (this.openAt || 0);
    if (this.open && openFor >= MAX_UTTERANCE_MS) {
      if (settings.debug) console.debug(`[otto] vad close: turn hit the ${MAX_UTTERANCE_MS} ms cap`);
      this.endSpeech();
      return;
    }
    if (!this.forceHold && this.silenceAcc >= this.silenceBudget()) {
      if (settings.debug) {
        console.debug(
          `[otto] vad close: ${Math.round(this.silenceAcc)}ms ≥ budget ${Math.round(this.silenceBudget())}ms` +
            ` (spoken ${Math.round(this.speechMs / 1000)}s, thr ${silenceThreshold.toFixed(3)}, rms ${rms.toFixed(3)})`,
        );
      }
      this.endSpeech();
    }
  }

  /** End-of-turn pause budget: grows with how much has been said, so a long
   *  monologue survives natural thinking pauses while a quick command stays snappy. */
  silenceBudget() {
    const base = silenceMs();
    const growth = (this.speechMs / 1000) * SILENCE_GROWTH_MS_PER_S;
    return clamp(base + growth, base, SILENCE_BUDGET_MAX_MS);
  }

  beginSpeech() {
    if (this.open) return;
    this.open = true;
    this.openAt = performance.now();
    this.openThreshold = this.threshold();
    this.speechAcc = 0;
    this.silenceAcc = 0;
    this.speechMs = 0;
    this.voiceRun = 0;
    this.bargeAcc = 0;
    const preroll = this.preroll.slice();
    this.preroll = [];
    this.prerollMs = 0;
    if (this.handlers.onSpeechStart) this.handlers.onSpeechStart(preroll.map((item) => item.chunk));
  }

  endSpeech() {
    if (!this.open) return;
    this.open = false;
    const durationMs = Math.round(performance.now() - this.openAt);
    this.speechAcc = 0;
    this.silenceAcc = 0;
    this.speechMs = 0;
    this.voiceRun = 0;
    this.bargeAcc = 0;
    this.openThreshold = 0;
    this.floorFreezeUntil = performance.now() + NOISE_FLOOR_FREEZE_MS;
    this.preroll = [];
    this.prerollMs = 0;
    if (this.handlers.onSpeechEnd) this.handlers.onSpeechEnd({ durationMs });
  }

  /** Push-to-talk: open an utterance immediately, ignore VAD silence until keyup. */
  holdStart() {
    if (!this.running || this.asrMode) return false;
    this.forceHold = true;
    if (this.handlers.onBargeIn) this.handlers.onBargeIn();
    this.beginSpeech();
    return true;
  }

  holdEnd() {
    const wasHolding = this.forceHold;
    this.forceHold = false;
    if (this.open) this.endSpeech();
    return wasHolding;
  }

  /**
   * Dev hook: feed `ms` of synthetic audio at `rms` through processChunk, one capture chunk at
   * a time and paced like the real device, so onset/endpoint/barge timing is measured for real.
   */
  async simulateLevel(rms, ms) {
    const level = clamp(num(rms, 0), 0, 1);
    const chunkMs = (CAPTURE_CHUNK / TARGET_RATE) * 1000;
    let remaining = Math.max(0, num(ms, 0));
    while (remaining > 0) {
      const slice = Math.min(chunkMs, remaining);
      remaining -= slice;
      // The chunk arrives at the end of its slice, exactly like a device buffer.
      await new Promise((resolve) => window.setTimeout(resolve, slice));
      this.processChunk(syntheticChunk(level, Math.round((slice / 1000) * TARGET_RATE)));
    }
  }
}

/* ------------------------------------------------------- barge-in policy -- */

/** True while Otto owns the audio path: playback queued, or a turn being produced. */
function ottoSpeaking() {
  // NOT while a confirmation is pending: the question has been asked, the floor
  // is the user's — their answer must open the mic like any turn, not barge in
  return (
    player.isPlaying() ||
    app.phase === 'speaking' ||
    app.phase === 'thinking'
  );
}

/**
 * Stop Otto mid-answer: silence playback synchronously and tell the server, in that order.
 * The caller starts the new utterance right after, so `barge_in` always precedes
 * `utterance.start` on the wire. Returns false when there was nothing to interrupt.
 */
function bargeIn(reason) {
  if (!ottoSpeaking()) return false;
  app.vadStats.bargeIns += 1;
  player.flush();
  sendJson({ type: 'barge_in', ts: Date.now() });
  if (settings.debug) console.debug('[otto] barge-in', { reason: text(reason) || 'speech' });
  return true;
}

const mic = new Capture({
  onLevel: (value) => {
    app.micLevel = value;
  },
  /** The VAD needs to know when Otto owns the audio path: the barge rule is stricter there. */
  isOttoSpeaking: () => ottoSpeaking(),
  ottoPlayedForMs: () => player.playedForMs(),
  /** Barge-in flush happens before utterance.start, so the server sees the order it expects. */
  onBargeIn: () => bargeIn('speech'),
  onSpeechStart: (chunks) => {
    app.vadStats.opened += 1;
    if (!app.connected) return;
    sendJson({ type: 'utterance.start', ts: Date.now() });
    app.utteranceOpen = true;
    for (const chunk of chunks) sendAudio(chunk);
  },
  onChunk: (chunk) => {
    if (app.utteranceOpen) sendAudio(chunk);
  },
  onSpeechEnd: ({ durationMs }) => {
    app.utteranceOpen = false;
    clearCaption();
    if (durationMs < MIN_UTTERANCE_MS) {
      // Too short to be speech: stay quiet and let the next utterance.start
      // reset the server's capture buffer.
      app.vadStats.rejectedBlips += 1;
      if (settings.debug) console.debug('[otto] dropped short utterance', durationMs);
      return;
    }
    app.vadStats.closed += 1;
    sendJson({ type: 'utterance.end', ts: Date.now(), durationMs });
  },
});

/* ------------------------------------------------- browser speech (opt-in) -- */

const SpeechRecognitionCtor = window.SpeechRecognition || window.webkitSpeechRecognition || null;

class BrowserASR {
  constructor(handlers) {
    this.handlers = handlers || {};
    this.recognition = null;
    this.enabled = false;
  }

  static get available() {
    return !!SpeechRecognitionCtor;
  }

  start() {
    if (!SpeechRecognitionCtor || this.enabled) return false;
    let recognition;
    try {
      recognition = new SpeechRecognitionCtor();
    } catch (err) {
      this.fail(err);
      return false;
    }
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = navigator.language || 'en-US';
    recognition.onresult = (event) => {
      let finalText = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) finalText += result[0].transcript;
      }
      const interim = event.results[event.results.length - 1];
      if (interim && !interim.isFinal && this.handlers.onPartial) {
        this.handlers.onPartial(text(interim[0] && interim[0].transcript));
      }
      const cleaned = finalText.trim();
      if (cleaned && this.handlers.onFinal) this.handlers.onFinal(cleaned);
    };
    recognition.onspeechstart = () => {
      if (this.handlers.onSpeechStart) this.handlers.onSpeechStart();
    };
    recognition.onerror = (event) => {
      const code = text(event && event.error) || 'speech-recognition-error';
      if (code === 'no-speech' || code === 'aborted') return;
      this.fail(new Error(code));
    };
    recognition.onend = () => {
      if (!this.enabled) return;
      try {
        recognition.start();
      } catch (err) {
        /* restart can throw while a previous session is closing */
      }
    };
    try {
      recognition.start();
    } catch (err) {
      this.fail(err);
      return false;
    }
    this.recognition = recognition;
    this.enabled = true;
    return true;
  }

  fail(err) {
    console.warn('[otto] browser speech recognition failed', err);
    this.stop();
    if (this.handlers.onError) this.handlers.onError(err);
  }

  stop() {
    this.enabled = false;
    const recognition = this.recognition;
    this.recognition = null;
    if (!recognition) return;
    try {
      recognition.onend = null;
      recognition.onresult = null;
      recognition.stop();
    } catch (err) {
      /* ignore */
    }
  }
}

const browserAsr = new BrowserASR({
  onSpeechStart: () => {
    if (!app.connected) return;
    bargeIn('browser speech');
  },
  onPartial: (value) => showCaption(value),
  onFinal: (value) => {
    clearCaption();
    sendJson({ type: 'text', text: value });
  },
  onError: (err) => {
    settings.sendTranscriptText = false;
    mic.asrMode = false;
    browserAsr.stop();
    saveSettings();
    applySettingsToUi();
    toast(`Browser speech recognition stopped: ${err && err.message ? err.message : 'unavailable'}`, 'warn');
  },
});

/* ------------------------------------------------------------------ socket -- */

function connectSocket() {
  window.clearTimeout(wsRetryTimer);
  if (app.sessionEnded) return;
  if (socketws && (socketws.readyState === WebSocket.OPEN || socketws.readyState === WebSocket.CONNECTING)) {
    return;
  }
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/ws`;
  setConn(wsAttempt ? 'connecting' : 'connecting', wsAttempt ? 'Reconnecting' : 'Connecting');

  let ws;
  try {
    ws = new WebSocket(url);
  } catch (err) {
    console.error('[otto] could not open websocket', err);
    setConn('down', 'Offline');
    scheduleReconnect();
    return;
  }
  ws.binaryType = 'arraybuffer';
  socketws = ws;

  ws.addEventListener('open', () => {
    if (socketws !== ws) return;
    wsAttempt = 0;
    app.connected = true;
    setConn('up', 'Live');
    sendJson({ type: 'hello', sampleRate: TARGET_RATE });
    window.clearInterval(wsPingTimer);
    wsPingTimer = window.setInterval(() => sendJson({ type: 'ping' }), PING_INTERVAL_MS);
  });

  ws.addEventListener('message', (event) => {
    if (typeof event.data === 'string') {
      let message;
      try {
        message = JSON.parse(event.data);
      } catch (err) {
        console.warn('[otto] dropped unparsable control frame', err);
        return;
      }
      if (settings.debug) console.debug('[otto] <-', message);
      safe(() => handleServerMessage(message), 'handling server message');
      return;
    }
    if (event.data instanceof ArrayBuffer) {
      safe(() => player.handleFrame(event.data), 'handling audio frame');
    }
  });

  ws.addEventListener('close', (event) => {
    if (socketws !== ws) return;
    socketws = null;
    app.connected = false;
    app.utteranceOpen = false;
    window.clearInterval(wsPingTimer);
    if (settings.debug) console.debug('[otto] socket closed', { code: event && event.code });
    // 4401 is the server saying the session is gone: never retry that loop.
    if (event && event.code === 4401) {
      endSession();
      return;
    }
    setConn('down', 'Offline');
    scheduleReconnect();
  });

  ws.addEventListener('error', () => {
    // A close event always follows; nothing to do here beyond logging.
    console.warn('[otto] websocket error');
  });
}

function scheduleReconnect() {
  window.clearTimeout(wsRetryTimer);
  wsAttempt += 1;
  const delay = Math.min(15000, 600 * Math.pow(1.8, wsAttempt - 1)) + Math.random() * 250;
  setConn('connecting', `Reconnecting in ${Math.max(1, Math.round(delay / 1000))}s`);
  wsRetryTimer = window.setTimeout(connectSocket, delay);
}

function forceReconnect() {
  window.clearTimeout(wsRetryTimer);
  wsAttempt = 0;
  const ws = socketws;
  socketws = null;
  if (ws) {
    try {
      ws.close();
    } catch (err) {
      /* ignore */
    }
  }
  connectSocket();
}

function setConn(state, label) {
  app.connState = state;
  connPill.dataset.state = state;
  connLabel.textContent = label;
  brandDot.dataset.state = state;
  orb.setPhase(app.connected ? app.phase : state === 'down' ? 'offline' : 'idle');
  refreshStateLabel();
}

function sendJson(message) {
  if (!socketws || socketws.readyState !== WebSocket.OPEN) return false;
  try {
    socketws.send(JSON.stringify(message));
    if (settings.debug) console.debug('[otto] ->', message);
    return true;
  } catch (err) {
    console.warn('[otto] send failed', err);
    return false;
  }
}

function sendAudio(view) {
  if (!socketws || socketws.readyState !== WebSocket.OPEN) return;
  try {
    socketws.send(view);
  } catch (err) {
    console.warn('[otto] audio send failed', err);
  }
}

/* --------------------------------------------------------- server messages -- */

function handleServerMessage(message) {
  if (!message || typeof message !== 'object') return;
  switch (message.type) {
    case 'ready':
      onReady(message);
      break;
    case 'state':
      setPhase(text(message.state) || 'idle', text(message.detail));
      break;
    case 'stt.partial':
      showCaption(text(message.text));
      break;
    case 'stt.final':
      clearCaption();
      break;
    case 'user.text':
      addUserMessage(text(message.text), text(message.source) || 'voice');
      break;
    case 'assistant.delta':
      streamAssistant(text(message.id) || 'assistant', text(message.text));
      break;
    case 'assistant.message':
      finishAssistant(text(message.id) || 'assistant', text(message.text), message.spoken !== false);
      break;
    case 'assistant.thinking':
      showThinking(text(message.id) || 'assistant');
      break;
    case 'audio.start':
      player.handleStart(message.seq, message.sampleRate);
      break;
    case 'audio.end':
      player.handleEnd(message.seq);
      break;
    case 'cancel.audio':
      player.cancelFrom(message.fromSeq);
      break;
    case 'tool.call':
      onToolCall(message);
      break;
    case 'tool.result':
      onToolResult(message);
      break;
    case 'confirm.request':
      onConfirmRequest(message);
      break;
    case 'confirm.resolved':
      onConfirmResolved(message);
      break;
    case 'job.update':
      onJobUpdate(message);
      break;
    case 'job.cancelled':
      onJobCancelled(message);
      break;
    case 'action.created':
      onActionCreated(message);
      break;
    case 'notice':
      addNotice(text(message.level) || 'info', text(message.message));
      break;
    case 'error':
      addNotice('error', text(message.message));
      toast(text(message.message) || 'The turn failed.', 'error');
      break;
    case 'metrics':
      app.metrics = message.latency && typeof message.latency === 'object' ? message.latency : {};
      renderMetrics();
      break;
    case 'config':
      if (message.config) app.serverConfig = message.config;
      break;
    case 'pong':
      break;
    default:
      if (settings.debug) console.debug('[otto] unhandled message', message.type);
  }
}

function onReady(message) {
  app.sessionId = text(message.session);
  infoSession.textContent = app.sessionId || '—';
  if (message.config) applyServerConfig(message.config);
  if (message.personality && message.personality.id && !settings.personality) {
    settings.personality = text(message.personality.id);
  }
  if (message.vad && message.vad.silenceMs != null && !localSilenceOverride) {
    settings.silenceMs = clamp(num(message.vad.silenceMs, settings.silenceMs), 200, 2000);
  }
  populateSelects();
  applySettingsToUi();
  saveSettings();
  // Re-apply whatever the UI holds so a reconnect keeps the user's session config.
  sendJson(configPatch());
  setPhase(text(message.state) || app.phase || 'idle', '');
  updateModelBadge();
}

function applyServerConfig(config) {
  if (!config || typeof config !== 'object') return;
  app.serverConfig = config;
  if (!settings.model && config.model) settings.model = text(config.model);
  if (!settings.thinking && config.reasoning_effort) settings.thinking = text(config.reasoning_effort);
  if (!settings.voice && config.voice) settings.voice = text(config.voice);
  if (!settings.personality && config.personality) settings.personality = text(config.personality);
  if (!settings.instruct && config.instruct) settings.instruct = text(config.instruct);
  // The two capability flags have no "empty" value to defer to, and both stores (localStorage
  // and POST /api/config) are written together, so adopting the server value on connect keeps
  // them in step; anything the user flips afterwards rides the next config patch.
  if (typeof config.parallel_agents === 'boolean') settings.parallelAgents = config.parallel_agents;
  if (typeof config.background_model === 'boolean') settings.backgroundModel = config.background_model;
}

/** Normalise /api/bootstrap capabilities and drop a background-model choice this model cannot honour. */
function applyCapabilities(caps) {
  if (!caps || typeof caps !== 'object') return;
  const background = caps.background_model && typeof caps.background_model === 'object' ? caps.background_model : {};
  app.capabilities = {
    parallelAgents: caps.parallel_agents !== false,
    backgroundModel: { available: background.available === true, id: text(background.id) },
  };
  if (!app.capabilities.backgroundModel.available) settings.backgroundModel = false;
}

function configPatch() {
  const patch = { type: 'config' };
  if (settings.model) patch.model = settings.model;
  patch.reasoning_effort = settings.thinking || '';
  if (settings.voice) patch.voice = settings.voice;
  if (settings.personality) patch.personality = settings.personality;
  if (settings.instruct != null) patch.instruct = settings.instruct;
  // The server ignores patch fields it does not know: sending interrupt_ms is safe today and
  // keeps the browser's barge-in window visible in the same place as the rest of the config.
  patch.interrupt_ms = interruptMs();
  patch.parallel_agents = !!settings.parallelAgents;
  // never ask for the cheaper variant when this model has none
  patch.background_model = app.capabilities.backgroundModel.available && !!settings.backgroundModel;
  return patch;
}

/* --------------------------------------------------------------- transcript -- */

function shouldStick() {
  return mainEl.scrollHeight - mainEl.scrollTop - mainEl.clientHeight < 200;
}

function scrollLog() {
  mainEl.scrollTop = mainEl.scrollHeight;
}

function appendToLog(node) {
  const stick = shouldStick();
  logEmpty.hidden = true;
  appEl.classList.add('has-messages');
  transcriptEl.append(node);
  if (stick) scrollLog();
}

function addUserMessage(message, source) {
  if (!message) return;
  clearCaption();
  const wrap = el('div', { class: 'msg msg-user' });
  wrap.append(el('p', { class: 'msg-text', text: message }));
  wrap.append(el('span', { class: 'tag', text: source === 'typed' ? 'typed' : 'voice' }));
  appendToLog(wrap);
}

function assistantEntry(id) {
  const existing = assistantMessages.get(id);
  if (existing) return existing;
  const node = el('div', { class: 'msg msg-assistant streaming' });
  const body = el('p', { class: 'msg-text' });
  const meta = el('div', { class: 'msg-meta' });
  node.append(body, meta);
  const entry = { node, body, meta, buffer: '' };
  assistantMessages.set(id, entry);
  appendToLog(node);
  return entry;
}

function streamAssistant(id, delta) {
  if (!delta) {
    assistantEntry(id);
    return;
  }
  const stick = shouldStick();
  const entry = assistantEntry(id);
  entry.buffer += delta;
  entry.body.textContent = entry.buffer;
  if (stick) scrollLog();
}

function finishAssistant(id, message, spoken) {
  const stick = shouldStick();
  const entry = assistantEntry(id);
  if (message || !entry.buffer) entry.buffer = message || entry.buffer;
  entry.body.textContent = entry.buffer;
  entry.node.classList.remove('streaming');
  for (const node of entry.meta.querySelectorAll('.thinking')) node.remove();
  if (spoken === false && entry.buffer) {
    entry.meta.append(el('span', { text: 'text only' }));
  }
  if (stick) scrollLog();
}

function showThinking(id) {
  const entry = assistantEntry(id);
  if (entry.meta.querySelector('.thinking')) return;
  entry.meta.append(el('span', { class: 'thinking', text: 'reasoning…' }));
}

function showCaption(value) {
  liveCaption.textContent = value || '';
}

function clearCaption() {
  liveCaption.textContent = '';
}

function addNotice(level, message) {
  if (!message) return;
  const node = el('div', { class: 'notice', attrs: { 'data-level': level } });
  node.append(el('span', { class: 'notice-dot', attrs: { 'aria-hidden': 'true' } }));
  node.append(el('span', { text: message }));
  appendToLog(node);
}

function noticeOnce(key, level, message) {
  if (seenNotices.has(key)) return;
  seenNotices.add(key);
  addNotice(level, message);
}

/* ------------------------------------------------------------ display cards -- */

function summarizeArgs(args) {
  if (args == null) return 'no arguments';
  if (typeof args !== 'object') return String(args);
  const entries = Object.entries(args);
  if (!entries.length) return 'no arguments';
  return entries.map(([key, value]) => `${key}: ${shortValue(value)}`).join('  ·  ');
}

function shortValue(value) {
  if (value == null) return '—';
  if (typeof value === 'string') return value.length > 90 ? `"${value.slice(0, 89)}…"` : `"${value}"`;
  if (typeof value === 'object') {
    if (Array.isArray(value)) return `[${value.length} items]`;
    const encoded = JSON.stringify(value);
    return encoded && encoded.length > 90 ? `${encoded.slice(0, 89)}…` : text(encoded);
  }
  return String(value);
}

function rowNode({ title, meta = [], note, unread = false }) {
  const row = el('div', { class: unread ? 'row unread' : 'row' });
  row.append(el('span', { class: 'row-mark', attrs: { 'aria-hidden': 'true' } }));
  const body = el('div', { class: 'row-body' });
  if (title) body.append(el('span', { class: 'row-title', text: title }));
  const metaParts = meta.filter(Boolean);
  if (metaParts.length) {
    const metaEl = el('span', { class: 'row-meta' });
    metaParts.forEach((part, index) => {
      if (index) metaEl.append(el('span', { text: '·' }));
      metaEl.append(el('span', { text: part }));
    });
    body.append(metaEl);
  }
  if (note) body.append(el('span', { class: 'row-note', text: note }));
  row.append(body);
  return row;
}

function rowsList(rows) {
  const list = el('div', { class: 'rows' });
  for (const row of rows) if (row) list.append(row);
  return list;
}

function kvList(pairs) {
  const list = el('dl', { class: 'kv-list' });
  for (const [label, value] of pairs) {
    const rendered = Array.isArray(value) ? joinList(value) : text(value);
    if (!rendered) continue;
    const row = el('div', { class: 'kv-row' });
    row.append(el('dt', { text: label }));
    row.append(el('dd', { text: rendered }));
    list.append(row);
  }
  return list;
}

function mailCard(label, payload) {
  return el('div', { class: 'disp' }, [
    el('div', { class: 'disp-head', text: label }),
    kvList([
      ['To', joinList(payload.to)],
      ['Subject', text(payload.subject)],
      ['Body', text(payload.body)],
    ]),
  ]);
}

function eventRow(item) {
  if (!item || typeof item !== 'object') return null;
  const when = item.all_day
    ? `all day · ${formatWhen(item.start)}`
    : [formatWhen(item.start), formatWhen(item.end)].filter(Boolean).join(' → ');
  return rowNode({
    title: text(item.title) || '(no title)',
    meta: [when, text(item.location)],
  });
}

/**
 * Long text clamped to three lines with a "show more" toggle. The toggle is dropped again
 * when the text already fits, so short results stay clean.
 */
function clampedText(value) {
  const body = el('p', { class: 'clamp-body', text: text(value) });
  const toggle = el('button', { class: 'clamp-toggle', attrs: { type: 'button' }, text: 'show more' });
  toggle.hidden = true;
  toggle.addEventListener('click', () => {
    toggle.textContent = body.classList.toggle('clamped') ? 'show more' : 'show less';
  });
  const wrap = el('div', {}, [body, toggle]);
  window.requestAnimationFrame(() => {
    // line-clamp does not report overflow through scrollHeight, so compare the heights
    const full = body.offsetHeight;
    body.classList.add('clamped');
    if (full - body.offsetHeight <= 1) {
      body.classList.remove('clamped');
      return;
    }
    toggle.hidden = false;
  });
  return wrap;
}

function subagentRow(item) {
  const data = item && typeof item === 'object' ? item : {};
  const ok = data.ok !== false;
  const row = el('div', { class: 'sub-row', attrs: { 'data-state': ok ? 'ok' : 'fail' } });
  const head = el('div', { class: 'sub-head' });
  head.append(el('span', { class: 'sub-mark', attrs: { 'aria-hidden': 'true' } }));
  head.append(el('span', { class: 'sub-title', text: text(data.title) || 'subagent' }));
  const steps = Math.round(num(data.steps));
  const bits = [];
  if (data.ms != null) bits.push(`${Math.round(num(data.ms))} ms`);
  if (steps > 0) bits.push(`${steps} step${steps === 1 ? '' : 's'}`);
  if (bits.length) head.append(el('span', { class: 'sub-ms', text: bits.join(' · ') }));
  row.append(head);
  if (data.result) row.append(clampedText(data.result));
  const tools = arrayOf(data.tools).map((name) => text(name)).filter(Boolean);
  if (tools.length) {
    const list = el('div', { class: 'sub-tools' });
    for (const name of tools) list.append(el('span', { class: 'sub-tool', text: name }));
    row.append(list);
  }
  if (data.error) row.append(el('span', { class: 'sub-error', text: text(data.error) }));
  return row;
}

function jobRow(item) {
  if (!item || typeof item !== 'object') return null;
  const status = text(item.status) || 'queued';
  const settled = status === 'done' || status === 'failed' || status === 'cancelled';
  return rowNode({
    title: text(item.title) || text(item.id) || 'background task',
    meta: [status, settled && item.ms != null ? `${Math.round(num(item.ms))} ms` : '', text(item.kind)],
    note: text(status === 'failed' ? item.error : item.result),
    unread: !settled,
  });
}

const DISPLAY_RENDERERS = {
  email_list(payload) {
    const items = arrayOf(payload.items);
    const wrap = el('div', { class: 'disp' });
    const suffix = items.length === 1 ? 'message' : 'messages';
    wrap.append(
      el('div', {
        class: 'disp-head',
        text: payload.query ? `${items.length} ${suffix} · “${payload.query}”` : `${items.length} ${suffix}`,
      }),
    );
    if (!items.length) {
      wrap.append(el('p', { class: 'disp-empty', text: 'Nothing matched.' }));
      return wrap;
    }
    wrap.append(
      rowsList(
        items.map((item) =>
          rowNode({
            title: text(item.subject) || '(no subject)',
            meta: [text(item.from), formatWhen(item.date)],
            note: text(item.snippet),
            unread: !!item.unread,
          }),
        ),
      ),
    );
    return wrap;
  },

  email(payload) {
    const item = payload.item || {};
    const attachments = arrayOf(item.attachments).map((file) => text(file && (file.filename || file.name)) || 'file');
    return el('div', { class: 'disp' }, [
      el('div', { class: 'disp-head', text: text(item.subject) || '(no subject)' }),
      kvList([
        ['From', text(item.from)],
        ['To', text(item.to)],
        ['Date', formatWhen(item.date)],
        ['Files', attachments.join(', ')],
      ]),
      text(item.body) ? el('p', { class: 'disp-text', text: text(item.body) }) : null,
    ]);
  },

  email_draft(payload) {
    return mailCard('Draft', payload);
  },

  email_sent(payload) {
    return mailCard('Sent', payload);
  },

  event_list(payload) {
    const items = arrayOf(payload.items);
    const wrap = el('div', { class: 'disp' });
    wrap.append(el('div', { class: 'disp-head', text: `${items.length} ${items.length === 1 ? 'event' : 'events'}` }));
    if (!items.length) {
      wrap.append(el('p', { class: 'disp-empty', text: 'The calendar is clear.' }));
      return wrap;
    }
    wrap.append(rowsList(items.map(eventRow)));
    return wrap;
  },

  event_created(payload) {
    return el('div', { class: 'disp' }, [
      el('div', { class: 'disp-head', text: 'Added' }),
      rowsList([eventRow(payload.item)]),
    ]);
  },

  event_updated(payload) {
    return el('div', { class: 'disp' }, [
      el('div', { class: 'disp-head', text: 'Updated' }),
      rowsList([eventRow(payload.item)]),
    ]);
  },

  event_deleted(payload) {
    return el('div', { class: 'disp' }, [
      el('div', { class: 'disp-head', text: 'Deleted' }),
      el('p', { class: 'disp-text', text: text(payload.title) || text(payload.id) || 'event removed' }),
    ]);
  },

  slack_messages(payload) {
    const items = arrayOf(payload.items);
    const wrap = el('div', { class: 'disp' });
    const where = text(payload.channel) ? `#${text(payload.channel)}` : 'Slack';
    wrap.append(el('div', { class: 'disp-head', text: `${items.length} ${items.length === 1 ? 'message' : 'messages'} · ${where}` }));
    if (!items.length) {
      wrap.append(el('p', { class: 'disp-empty', text: 'Nothing there.' }));
      return wrap;
    }
    wrap.append(
      rowsList(
        items.map((item) =>
          rowNode({
            title: text(item.user) || 'someone',
            meta: [text(item.channel), formatWhen(item.ts)],
            note: text(item.text),
          }),
        ),
      ),
    );
    return wrap;
  },

  slack_sent(payload) {
    return el('div', { class: 'disp' }, [
      el('div', { class: 'disp-head', text: `Posted · #${text(payload.channel)}` }),
      el('p', { class: 'disp-text', text: text(payload.text) }),
    ]);
  },

  slack_channels(payload) {
    const items = arrayOf(payload.items);
    const wrap = el('div', { class: 'disp' });
    wrap.append(el('div', { class: 'disp-head', text: `${items.length} channels` }));
    if (!items.length) {
      wrap.append(el('p', { class: 'disp-empty', text: 'No channels visible.' }));
      return wrap;
    }
    const list = el('div', { class: 'tag-list' });
    for (const item of items) list.append(el('span', { class: 'tag-item', text: `#${text(item.name) || text(item.id)}` }));
    wrap.append(list);
    return wrap;
  },

  list(payload) {
    const items = arrayOf(payload.items);
    const wrap = el('div', { class: 'disp' });
    if (!items.length) {
      wrap.append(el('p', { class: 'disp-empty', text: 'Nothing to show.' }));
      return wrap;
    }
    wrap.append(rowsList(items.map((item) => rowNode({ title: typeof item === 'object' ? text(item.text || item.name || safeJson(item)) : text(item) }))));
    return wrap;
  },

  subagents(payload) {
    const items = arrayOf(payload.items);
    const wrap = el('div', { class: 'disp' });
    const elapsed = payload.elapsedMs != null ? ` · ${(num(payload.elapsedMs) / 1000).toFixed(1)} s` : '';
    wrap.append(el('div', { class: 'disp-head', text: `${items.length} subagent${items.length === 1 ? '' : 's'}${elapsed}` }));
    if (!items.length) {
      wrap.append(el('p', { class: 'disp-empty', text: 'No subagents ran.' }));
      return wrap;
    }
    const list = el('div', { class: 'sub-list' });
    for (const item of items) list.append(subagentRow(item));
    wrap.append(list);
    return wrap;
  },

  job_started(payload) {
    return rowsList([
      rowNode({
        title: `started ${text(payload.title) || 'background task'}`,
        meta: [text(payload.status) || 'queued', text(payload.id)],
        unread: true,
      }),
    ]);
  },

  jobs(payload) {
    const items = arrayOf(payload.items);
    const wrap = el('div', { class: 'disp' });
    wrap.append(el('div', { class: 'disp-head', text: `${items.length} background task${items.length === 1 ? '' : 's'}` }));
    if (!items.length) {
      wrap.append(el('p', { class: 'disp-empty', text: 'Nothing has run in the background.' }));
      return wrap;
    }
    wrap.append(rowsList(items.map(jobRow)));
    return wrap;
  },

  text(payload) {
    return el('p', { class: 'disp-text', text: text(payload.text) });
  },
};

function renderDisplay(display) {
  const fragment = document.createDocumentFragment();
  if (display == null) return fragment;
  if (typeof display !== 'object') {
    fragment.append(el('p', { class: 'disp-text', text: String(display) }));
    return fragment;
  }
  const kind = text(display.kind);
  const renderer = DISPLAY_RENDERERS[kind];
  if (!renderer) {
    fragment.append(el('pre', { class: 'disp-pre', text: safeJson(display) }));
    return fragment;
  }
  try {
    fragment.append(renderer(display));
  } catch (err) {
    console.warn('[otto] display renderer failed', kind, err);
    fragment.append(el('pre', { class: 'disp-pre', text: safeJson(display) }));
  }
  return fragment;
}

function renderPreview(preview) {
  const wrap = el('div', { class: 'confirm-preview' });
  if (preview == null) return wrap;
  if (typeof preview !== 'object') {
    wrap.append(el('p', { class: 'disp-text', text: String(preview) }));
    return wrap;
  }
  const kind = text(preview.kind);
  if (kind === 'email_send' || kind === 'email_draft') {
    wrap.append(
      kvList([
        ['To', joinList(preview.to)],
        ['Subject', text(preview.subject)],
        ['Body', text(preview.body)],
      ]),
    );
  } else if (kind === 'event_create') {
    wrap.append(
      kvList([
        ['Title', text(preview.title)],
        ['Start', formatWhen(preview.start)],
        ['End', formatWhen(preview.end)],
        ['Where', text(preview.location)],
        ['Notes', text(preview.description)],
        ['Guests', joinList(preview.attendees)],
      ]),
    );
  } else if (kind === 'event_update') {
    wrap.append(
      kvList([
        ['Event', text(preview.title) || text(preview.id)],
        ['Start', formatWhen(preview.start)],
        ['End', formatWhen(preview.end)],
        ['Where', text(preview.location)],
      ]),
    );
  } else if (kind === 'event_delete') {
    wrap.append(kvList([['Event', text(preview.title) || text(preview.id)]]));
  } else if (kind === 'slack_send') {
    wrap.append(kvList([['Channel', text(preview.channel)], ['Message', text(preview.text)]]));
  } else {
    wrap.append(el('pre', { class: 'disp-pre', text: safeJson(preview) }));
  }
  return wrap;
}

function onToolCall(message) {
  const id = text(message.id) || `tool-${++devSeq}`;
  const card = el('article', {
    class: 'card tool-card',
    attrs: { 'data-state': 'running', 'data-tool-id': id },
  });
  const head = el('header', { class: 'tool-head' });
  head.append(el('span', { class: 'tool-status', attrs: { 'aria-hidden': 'true' } }));
  head.append(el('span', { class: 'tool-name', text: text(message.name) || 'tool' }));
  if (message.category) head.append(el('span', { class: 'tool-cat', text: text(message.category) }));
  const ms = el('span', { class: 'tool-ms' });
  head.append(ms);
  const args = el('div', { class: 'tool-args', text: summarizeArgs(message.args) });
  const body = el('div', { class: 'tool-body' });
  card.append(head, args, body);
  toolCards.set(id, { card, ms, body });
  appendToLog(card);
}

function onToolResult(message) {
  const id = text(message.id);
  let entry = toolCards.get(id);
  if (!entry) {
    onToolCall({ id, name: message.name, args: {}, category: '' });
    entry = toolCards.get(id);
  }
  if (!entry) return;
  const stick = shouldStick();
  const ok = message.ok !== false;
  entry.card.dataset.state = ok ? 'ok' : 'fail';
  entry.ms.textContent = message.ms != null ? `${Math.round(num(message.ms))} ms` : '';
  if (message.display) entry.body.append(renderDisplay(message.display));
  if (message.summary) entry.card.append(el('p', { class: 'tool-summary', text: text(message.summary) }));
  if (message.error) entry.card.append(el('div', { class: 'tool-error', text: text(message.error) }));
  if (stick) scrollLog();
}

function onConfirmRequest(message) {
  const id = text(message.id) || `confirm-${++devSeq}`;
  const danger = message.danger === true;
  const card = el('article', {
    class: 'card confirm-card',
    attrs: { 'data-danger': danger ? 'true' : 'false', 'data-confirm-id': id },
  });
  const head = el('div', { class: 'confirm-head' });
  head.append(el('span', { class: 'confirm-badge', text: danger ? 'Confirm · irreversible' : 'Confirm' }));
  head.append(el('span', { class: 'tool-name', text: text(message.name) || text(message.tool) || 'tool' }));
  card.append(head);
  card.append(el('p', { class: 'confirm-summary', text: text(message.summary) || 'Run this action?' }));
  card.append(renderPreview(message.preview));

  const actions = el('div', { class: 'confirm-actions' });
  const approve = el('button', { class: danger ? 'btn danger' : 'btn primary', attrs: { type: 'button' } });
  approve.append(document.createTextNode('Approve'), el('kbd', { text: 'Y' }));
  const deny = el('button', { class: 'btn ghost', attrs: { type: 'button' } });
  deny.append(document.createTextNode('Deny'), el('kbd', { text: 'N' }));
  approve.addEventListener('click', () => answerConfirm(true, id));
  deny.addEventListener('click', () => answerConfirm(false, id));
  actions.append(approve, deny);
  card.append(actions);

  const entry = { card, approve, deny };
  confirmCards.set(id, entry);
  app.pendingConfirm = { id, ...entry };
  appendToLog(card);
}

function answerConfirm(approved, id) {
  const pending = app.pendingConfirm;
  if (!pending || (id && pending.id !== id)) {
    toast('Nothing to confirm right now.', 'info');
    return;
  }
  sendJson({ type: 'confirm', id: pending.id, approved: !!approved });
  markConfirm(pending, approved ? 'approved · running…' : 'denied');
  app.pendingConfirm = null;
}

function markConfirm(entry, label) {
  if (!entry) return;
  entry.card.dataset.resolved = 'true';
  for (const button of [entry.approve, entry.deny]) {
    if (button) button.disabled = true;
  }
  const existing = entry.card.querySelector('.confirm-resolved');
  if (existing) existing.textContent = label;
  else entry.card.append(el('p', { class: 'confirm-resolved', text: label }));
}

function onConfirmResolved(message) {
  const id = text(message.id);
  const entry = confirmCards.get(id);
  if (entry) markConfirm(entry, message.approved ? 'approved' : 'denied');
  if (app.pendingConfirm && app.pendingConfirm.id === id) app.pendingConfirm = null;
}

/* --------------------------------------------------------------- job cards -- */

const JOB_KIND_LABEL = 'background';

function onJobUpdate(message) {
  const id = text(message.id) || `job-${++devSeq}`;
  upsertJobCard(id, {
    title: text(message.title),
    kind: text(message.kind),
    status: text(message.status) || 'running',
    ms: message.ms,
    result: text(message.result),
    error: text(message.error),
  });
}

/** One card per job id: created by the first job.update, updated in place afterwards. */
function upsertJobCard(id, update) {
  const stick = shouldStick();
  let entry = jobCards.get(id);
  if (!entry) {
    const card = el('article', { class: 'card job-card', attrs: { 'data-job-id': id, 'data-state': update.status } });
    const head = el('div', { class: 'job-head' });
    head.append(el('span', { class: 'job-dot', attrs: { 'aria-hidden': 'true' } }));
    const title = el('span', { class: 'job-title', text: update.title || 'background task' });
    // the chip names the card type; the server's job kind stays available as its tooltip
    const kind = el('span', { class: 'tool-cat', text: JOB_KIND_LABEL });
    const ms = el('span', { class: 'job-ms' });
    const cancel = el('button', { class: 'job-cancel', attrs: { type: 'button' }, text: 'Cancel' });
    cancel.addEventListener('click', () => cancelJob(id, cancel));
    head.append(title, kind, ms, cancel);
    const body = el('div', { class: 'job-body' });
    card.append(head, body);
    entry = { card, title, kind, ms, cancel, body, status: '', settled: false, errorNode: null };
    jobCards.set(id, entry);
    appendToLog(card);
  }

  const status = update.status;
  const settled = status === 'done' || status === 'failed' || status === 'cancelled';
  entry.status = status;
  entry.card.dataset.state = status;
  if (update.title) entry.title.textContent = update.title;
  if (update.kind) entry.kind.title = update.kind;
  entry.ms.textContent = settled && update.ms != null ? `${Math.round(num(update.ms))} ms` : '';
  entry.cancel.hidden = settled;

  clear(entry.body);
  if (status === 'done' && update.result) entry.body.append(clampedText(update.result));
  else if (status === 'cancelled') entry.body.append(el('span', { class: 'job-state-label', text: 'cancelled' }));

  if (update.error) {
    if (!entry.errorNode) {
      entry.errorNode = el('div', { class: 'job-error' });
      entry.card.append(entry.errorNode);
    }
    entry.errorNode.textContent = update.error;
  } else if (entry.errorNode) {
    entry.errorNode.remove();
    entry.errorNode = null;
  }

  if (settled && !entry.settled) {
    entry.settled = true;
    jobActivity.unseen += 1;
  }
  renderActivity();
  if (stick) scrollLog();
}

function cancelJob(id, button) {
  if (button) button.disabled = true;
  if (sendJson({ type: 'job.cancel', id })) return;
  if (button) button.disabled = false;
  toast('Not connected — the task was not cancelled.', 'warn');
}

function onJobCancelled(message) {
  const entry = jobCards.get(text(message.id));
  if (!entry) return;
  entry.cancel.disabled = true;
  if (message.ok === true) return; // a job.update with status "cancelled" follows
  toast('That background task had already finished.', 'info');
}

/** Compact "Running 2 · title (+1)" strip above the composer; hidden when nothing runs. */
function renderActivity() {
  const running = [...jobCards.values()].filter((job) => job.status === 'running' || job.status === 'queued');
  if (!running.length) {
    activityStrip.hidden = true;
    activityText.textContent = '';
    activityUnseen.hidden = true;
    return;
  }
  const extra = running.length - 1;
  const label = `Running ${running.length} · ${running[0].title.textContent}`;
  activityText.textContent = extra > 0 ? `${label} (+${extra})` : label;
  activityUnseen.hidden = jobActivity.unseen <= 0;
  activityUnseen.textContent = jobActivity.unseen > 0 ? `${jobActivity.unseen} finished` : '';
  activityStrip.hidden = false;
}

function focusRunningJob() {
  jobActivity.unseen = 0;
  renderActivity();
  const job = [...jobCards.values()].find((entry) => entry.status === 'running' || entry.status === 'queued');
  if (job) job.card.scrollIntoView({ block: 'center', behavior: 'smooth' });
}

/* --------------------------------------------------------------- orb + frame -- */

class Orb {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.dpr = 1;
    this.width = 0;
    this.height = 0;
    this.phase = 'idle';
    this.palette = PALETTES.idle;
    this.level = 0;
    this.time = 0;
    this.reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (window.ResizeObserver) {
      this.observer = new ResizeObserver(() => this.resize());
      this.observer.observe(canvas.parentElement || canvas);
    }
    window.addEventListener('resize', () => this.resize());
    this.resize();
  }

  resize() {
    const box = this.canvas.parentElement || this.canvas;
    const rect = box.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    this.dpr = dpr;
    this.width = width;
    this.height = height;
    this.canvas.width = Math.round(width * dpr);
    this.canvas.height = Math.round(height * dpr);
  }

  setPhase(phase) {
    const next = PALETTES[phase] ? phase : 'idle';
    if (next === this.phase) return;
    this.phase = next;
    this.palette = PALETTES[next];
  }

  setLevel(level) {
    this.level = clamp(level, 0, 1);
  }

  step(dt) {
    if (!this.reduced) this.time += dt;
  }

  /**
   * One soft gradient body inside a wide ambient glow. Idle breathes on a slow 6 s cycle,
   * listening/speaking move with the audio level, thinking adds a slow rotating shimmer.
   * No ticks, no rings: the sphere is the whole signal.
   */
  draw() {
    const ctx = this.ctx;
    if (!ctx || !this.width || !this.height) return;
    const w = this.width;
    const h = this.height;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2;
    // the body fills ~65% of the box; the rest is glow
    const base = Math.min(w, h) * 0.325;
    const t = this.time;
    const level = this.level;
    const glow = this.palette.glow;
    const core = this.palette.core;
    const breath = this.phase === 'idle' ? Math.sin((t / 6) * Math.PI * 2) : 0;
    const radius = base * (1 + breath * 0.018 + level * 0.07);

    // ambient halo
    const halo = ctx.createRadialGradient(cx, cy, radius * 0.72, cx, cy, radius * 1.85);
    halo.addColorStop(0, `rgba(${glow}, ${0.1 + level * 0.16})`);
    halo.addColorStop(0.45, `rgba(${glow}, ${0.035 + level * 0.06})`);
    halo.addColorStop(1, `rgba(${glow}, 0)`);
    ctx.fillStyle = halo;
    ctx.fillRect(0, 0, w, h);

    // the body: lit from the upper left, dissolving into the canvas at the edge
    const body = ctx.createRadialGradient(
      cx - radius * 0.3,
      cy - radius * 0.36,
      radius * 0.04,
      cx,
      cy,
      radius * 1.02,
    );
    body.addColorStop(0, `rgba(${glow}, ${0.46 + level * 0.22})`);
    body.addColorStop(0.36, `rgba(${core}, ${0.46 + level * 0.16})`);
    body.addColorStop(0.74, `rgba(${core}, 0.14)`);
    body.addColorStop(0.93, 'rgba(18, 20, 24, 0.94)');
    body.addColorStop(1, 'rgba(12, 13, 15, 0)');
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = body;
    ctx.fill();

    // thinking: a slow rotating shimmer inside the body
    if (this.phase === 'thinking') {
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, radius * 0.99, 0, Math.PI * 2);
      ctx.clip();
      ctx.globalCompositeOperation = 'lighter';
      for (let i = 0; i < 2; i++) {
        const angle = t * 0.5 + i * Math.PI;
        const sx = cx + Math.cos(angle) * radius * 0.55;
        const sy = cy + Math.sin(angle) * radius * 0.5;
        const sweep = ctx.createRadialGradient(sx, sy, 0, sx, sy, radius * 1.1);
        sweep.addColorStop(0, `rgba(${glow}, 0.15)`);
        sweep.addColorStop(1, `rgba(${glow}, 0)`);
        ctx.fillStyle = sweep;
        ctx.fillRect(cx - radius, cy - radius, radius * 2, radius * 2);
      }
      ctx.restore();
    }

    // rim light keeps the edge readable against the near-black canvas
    ctx.lineWidth = 1;
    ctx.strokeStyle = `rgba(255, 255, 255, ${0.05 + level * 0.07})`;
    ctx.beginPath();
    ctx.arc(cx, cy, radius - 0.5, 0, Math.PI * 2);
    ctx.stroke();

    // one soft specular sheen, a little stronger while Otto speaks
    const sheen = ctx.createRadialGradient(
      cx - radius * 0.34,
      cy - radius * 0.42,
      0,
      cx - radius * 0.3,
      cy - radius * 0.38,
      radius * 0.92,
    );
    sheen.addColorStop(0, `rgba(255, 255, 255, ${0.09 + level * 0.1})`);
    sheen.addColorStop(0.66, 'rgba(255, 255, 255, 0)');
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = sheen;
    ctx.fill();
  }
}

const orb = new Orb(orbCanvas);

const METER_BARS = 16;
function buildMeter() {
  clear(meterEl);
  for (let i = 0; i < METER_BARS; i++) meterEl.append(el('i'));
}

function updateMeter(level) {
  const bars = meterEl.children;
  const count = bars.length;
  const scaled = clamp(Math.pow(level, 0.75) * 1.2, 0, 1);
  for (let i = 0; i < count; i++) {
    const distance = Math.abs(i - (count - 1) / 2) / ((count - 1) / 2);
    const shape = 1 - distance * 0.55;
    const height = clamp(scaled * shape, 0.1, 1);
    bars[i].style.transform = `scaleY(${height.toFixed(3)})`;
    bars[i].style.opacity = String(0.3 + height * 0.65);
  }
  micBtn.style.setProperty('--level', scaled.toFixed(3));
}

let lastFrame = 0;
function tick(timestamp) {
  const dt = lastFrame ? Math.min(0.05, (timestamp - lastFrame) / 1000) : 0.016;
  lastFrame = timestamp;

  const speaking = app.phase === 'speaking';
  const live = mic.running && !app.muted;
  let level;
  if (speaking) {
    level = player.readLevel() * 0.9 + (live ? app.micLevel * 0.25 : 0);
  } else if (live) {
    level = app.micLevel;
  } else {
    level = 0;
  }
  // idle breathing lives in the orb itself (a slow 6 s cycle), not in the level
  orb.setLevel(clamp(level, 0, 1));
  orb.step(dt);
  orb.draw();
  updateMeter(live || speaking ? level : 0);

  window.requestAnimationFrame(tick);
}

/* ------------------------------------------------------------------- phase -- */

const PHASE_LABELS = {
  idle: 'Ready',
  listening: 'Listening…',
  thinking: 'Thinking…',
  speaking: 'Speaking…',
  awaiting_confirmation: 'Waiting for you',
};

function setPhase(phase, detail) {
  app.phase = PHASE_LABELS[phase] ? phase : 'idle';
  app.detail = detail || '';
  orb.setPhase(app.connected ? app.phase : app.connState === 'down' ? 'offline' : 'idle');
  refreshStateLabel();
  // awaiting_confirmation is the user's floor: keep the stop button hidden and
  // the composer in mic mode so answering feels like talking, not interrupting
  const busy = app.phase === 'speaking' || app.phase === 'thinking';
  stopBtn.hidden = !busy;
  // while Otto is talking the composer's mic slot becomes the stop control
  composerEl.dataset.busy = busy ? 'true' : 'false';
}

function refreshStateLabel() {
  let label = PHASE_LABELS[app.phase] || 'Ready';
  let detailText = '';
  if (!app.connected) {
    label = app.connState === 'down' ? 'Offline' : 'Connecting';
  } else if (app.detail === 'transcribing') {
    label = 'Transcribing';
  } else if (app.detail && app.detail.startsWith('tool:')) {
    detailText = `running ${app.detail.slice(5)}`;
  } else if (app.detail) {
    detailText = app.detail;
  }
  stateLabel.textContent = label;
  stateDetail.textContent = detailText;
}

function stopEverything() {
  player.flush();
  sendJson({ type: 'cancel' });
  if (app.phase === 'speaking' || app.phase === 'thinking') setPhase('listening', '');
}

/* ------------------------------------------------------------------ metrics -- */

function renderMetrics() {
  const latency = app.metrics || {};
  const parts = [];
  if (latency.stt_ms != null) parts.push(`stt ${Math.round(num(latency.stt_ms))} ms`);
  if (latency.llm_ttft_ms != null) parts.push(`ttft ${Math.round(num(latency.llm_ttft_ms))} ms`);
  if (latency.first_audio_ms != null) parts.push(`audio ${Math.round(num(latency.first_audio_ms))} ms`);
  if (latency.turn_ms != null) parts.push(`turn ${(num(latency.turn_ms) / 1000).toFixed(2)} s`);
  metricsEl.textContent = parts.length ? parts.join('  ·  ') : 'no metrics for this turn yet';
  metricsEl.hidden = !settings.debug;
}

/* ------------------------------------------------------------------- health -- */

function setBadge(node, label, state, title) {
  node.hidden = false;
  node.textContent = label;
  node.dataset.state = state;
  node.title = title || label;
}

async function refreshHealth() {
  try {
    const response = await authFetch('/api/health', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    applyHealth(await response.json());
  } catch (err) {
    console.warn('[otto] health check failed', err);
  }
}

function applyHealth(status) {
  if (!status || typeof status !== 'object') return;
  const lunaroute = status.lunaroute || {};
  const llmOk = lunaroute.configured === true;
  setBadge(
    badgeLlm,
    llmOk ? 'LLM' : 'No API key',
    llmOk ? 'ok' : 'bad',
    llmOk ? `Lunaroute configured · ${text(lunaroute.model) || 'model'}` : 'LUNAROUTE_API_KEY is empty — replies will fail',
  );
  const tts = status.tts || {};
  const ttsOk = tts.ok === true;
  app.ttsDown = !ttsOk;
  setBadge(
    badgeTts,
    ttsOk ? 'Voice' : 'No voice',
    ttsOk ? 'ok' : 'warn',
    ttsOk ? `Local TTS ready · ${text(tts.model) || 'qwen3-tts'}` : `TTS offline: ${text(tts.detail) || 'not reachable'}`,
  );
  if (!ttsOk) {
    noticeOnce('tts-down', 'warn', 'Local voice is offline — replies will be text only until the TTS service is running.');
  }
  if (!llmOk) {
    noticeOnce('llm-missing', 'warn', 'No Lunaroute API key configured — the assistant cannot reply until LUNAROUTE_API_KEY is set.');
  }
  if (status.version) infoVersion.textContent = text(status.version);
  updateModelBadge();
}

function updateModelBadge() {
  const model = settings.model || text(app.serverConfig.model) || 'model';
  badgeModel.textContent = model;
  badgeModel.title = `Active model · ${model}`;
}

/* ------------------------------------------------------------------- mic ui -- */

function micErrorMessage(err) {
  const name = err && err.name;
  if (name === 'NotAllowedError' || name === 'SecurityError') {
    return 'Microphone access was blocked. Allow it in the browser and try again.';
  }
  if (name === 'NotFoundError' || name === 'DevicesNotFoundError') return 'No microphone was found.';
  if (name === 'NotReadableError') return 'The microphone is in use by another application.';
  return `Microphone unavailable: ${err && err.message ? err.message : err}`;
}

let micStarting = false;

async function enableMic() {
  if (mic.running || micStarting) return;
  micStarting = true;
  infoMic.textContent = 'starting…';
  try {
    await mic.start(settings.deviceId || '');
    mic.asrMode = !!settings.sendTranscriptText && browserAsr.enabled;
    app.muted = false;
    updateMicUi();
    orbEl.classList.remove('needs-permission');
  } catch (err) {
    console.warn('[otto] microphone start failed', err);
    const message = micErrorMessage(err);
    toast(message, 'error');
    noticeOnce(`mic-${err && err.name ? err.name : 'error'}`, 'warn', message);
    app.muted = true;
    orbEl.classList.add('needs-permission');
    infoMic.textContent = 'blocked';
    updateMicUi();
  } finally {
    micStarting = false;
  }
}

async function toggleMute() {
  if (app.muted || !mic.running) {
    await enableMic();
    return;
  }
  app.muted = true;
  await mic.stop();
  mic.asrMode = false;
  infoMic.textContent = 'muted';
  updateMicUi();
}

function updateMicUi() {
  const live = mic.running && !app.muted;
  micBtn.setAttribute('aria-pressed', live ? 'true' : 'false');
  micBtn.classList.toggle('live', live);
  micBtn.setAttribute('aria-label', live ? 'Mute microphone' : 'Unmute microphone');
  orbEl.classList.toggle('needs-permission', !live);
  if (live) infoMic.textContent = mic.mode === 'worklet' ? 'capturing' : 'capturing (fallback)';
}

async function autoStartMic() {
  try {
    if (!navigator.permissions || !navigator.permissions.query) {
      orbEl.classList.add('needs-permission');
      return;
    }
    const status = await navigator.permissions.query({ name: 'microphone' });
    if (status.state === 'granted') await enableMic();
    else orbEl.classList.add('needs-permission');
  } catch (err) {
    orbEl.classList.add('needs-permission');
  }
}

async function refreshDevices() {
  const select = $('set-device');
  if (!select) return;
  let devices = [];
  try {
    devices = await navigator.mediaDevices.enumerateDevices();
  } catch (err) {
    console.warn('[otto] enumerateDevices failed', err);
  }
  const inputs = devices.filter((device) => device.kind === 'audioinput');
  const options = [{ value: '', label: 'System default' }];
  inputs.forEach((device, index) => {
    options.push({ value: device.deviceId, label: device.label || `Microphone ${index + 1}` });
  });
  fillSelect(select, options, settings.deviceId || '');
}

/* ----------------------------------------------------------------- settings -- */

function fillSelect(node, options, value) {
  clear(node);
  const list = options.slice();
  if (value && !list.some((option) => option.value === value)) {
    // Keep an unknown current value selectable instead of silently switching it.
    list.unshift({ value, label: list.length ? `${value} (unknown)` : value });
  }
  if (!list.length) list.push({ value: '', label: '—' });
  for (const option of list) {
    node.append(el('option', { text: option.label, attrs: { value: option.value } }));
  }
  node.value = value && list.some((option) => option.value === value) ? value : list[0].value;
}


/* ------------------------------------------------------- personality editor */
async function openPersonalityEditor() {
  const id = text(settings.personality) || text($('set-personality').value);
  if (!id) {
    toast('No personality selected yet', 'error');
    return;
  }
  try {
    const response = await authFetch(`/api/personalities/${encodeURIComponent(id)}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    $('persona-name').value = text(data.name) || id;
    $('persona-prompt').value = text(data.system_prompt);
    $('persona-id').textContent = `id: ${id} · ${text(data.source) || 'builtin'} · ${text(data.tagline)}`.trim();
    $('persona-error').textContent = '';
    $('persona-error').removeAttribute('data-level');
    $('persona-dialog').showModal();
  } catch (err) {
    toast(`Could not load personality: ${err.message}`, 'error');
  }
}

async function savePersonality(event) {
  event.preventDefault();
  const id = text(settings.personality) || text($('set-personality').value);
  const name = $('persona-name').value.trim();
  const prompt = $('persona-prompt').value.trim();
  const errorEl = $('persona-error');
  if (!id || !prompt) {
    errorEl.textContent = 'The system prompt cannot be empty.';
    errorEl.dataset.level = 'error';
    return;
  }
  const saveBtn = $('persona-save');
  saveBtn.disabled = true;
  try {
    const response = await authFetch(`/api/personalities/${encodeURIComponent(id)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name || id, system_prompt: prompt }),
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    $('persona-dialog').close();
    // refresh the picker labels and make the running session reload the prompt
    const list = await authFetch('/api/personalities').then((r) => r.json()).catch(() => null);
    if (list && Array.isArray(list.personalities)) {
      app.personalities = list.personalities;
      populateSelects();
    }
    sendJson({ type: 'config', personality: id });
    $('personality-status').textContent = 'saved';
    toast(`${name || id} updated`, 'ok');
    window.setTimeout(() => {
      $('personality-status').textContent = '';
    }, 2500);
  } catch (err) {
    errorEl.textContent = `Could not save: ${err.message}`;
    errorEl.dataset.level = 'error';
  } finally {
    saveBtn.disabled = false;
  }
}

function populateSelects() {
  const models = app.models.length
    ? app.models.map((model) => ({ value: text(model.id), label: text(model.label) || text(model.id) }))
    : [];
  fillSelect($('set-model'), models, settings.model);
  const thinking = $('set-thinking');
  if (thinking) thinking.value = settings.thinking || '';

  const personalities = app.personalities.map((personality) => ({
    value: text(personality.id),
    label: text(personality.name) || text(personality.id),
  }));
  fillSelect($('set-personality'), personalities, settings.personality);

  const voices = app.voices.map((voice) => ({ value: text(voice.id), label: text(voice.id) }));
  fillSelect($('set-voice'), voices, settings.voice);

  // Adopt whatever the selects settled on so the first send/ready matches the UI.
  if (!settings.model) settings.model = $('set-model').value || '';
  if (!settings.personality) settings.personality = $('set-personality').value || '';
  if (!settings.voice) settings.voice = $('set-voice').value || '';
  saveSettings();
  updateModelBadge();
  updatePersonalityNote();
}

function updatePersonalityNote() {
  const note = $('personality-note');
  const match = app.personalities.find((personality) => text(personality.id) === settings.personality);
  note.textContent = match && match.tagline ? text(match.tagline) : '';
}

function applySettingsToUi() {
  $('set-sensitivity').value = String(clamp(num(settings.sensitivity, 55), 0, 100));
  $('sen-value').textContent = String(Math.round(num(settings.sensitivity, 55)));
  $('set-silence').value = String(clamp(num(settings.silenceMs, 550), 200, 1500));
  $('silence-value').textContent = `${Math.round(num(settings.silenceMs, 550))} ms`;
  $('set-interrupt').value = String(clamp(num(settings.interruptMs, 250), 150, 600));
  $('interrupt-value').textContent = `${Math.round(num(settings.interruptMs, 250))} ms`;
  $('set-instruct').value = text(settings.instruct);
  $('set-transcript-text').checked = !!settings.sendTranscriptText;
  $('set-debug').checked = !!settings.debug;
  $('set-parallel-agents').checked = !!settings.parallelAgents;
  updateBackgroundModelToggle();
  metricsEl.hidden = !settings.debug;
  updateModelBadge();
  updatePersonalityNote();
}

/** The background-model toggle only works when this model has a *-background sibling. */
function updateBackgroundModelToggle() {
  const capability = app.capabilities.backgroundModel;
  const input = $('set-background-model');
  const note = $('set-background-model-note');
  const field = input.closest('.field') || input.parentElement;
  input.disabled = !capability.available;
  input.checked = capability.available && !!settings.backgroundModel;
  if (field) field.classList.toggle('disabled', !capability.available);
  note.textContent = capability.available
    ? `Run subagents and background jobs on ${capability.id || 'the cheaper variant'}.`
    : 'not offered for this model';
}

function scheduleConfigPersist() {
  window.clearTimeout(configPostTimer);
  configPostTimer = window.setTimeout(async () => {
    try {
      await authFetch('/api/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: settings.model,
          voice: settings.voice,
          personality: settings.personality,
          instruct: settings.instruct,
          reasoning_effort: settings.thinking || '',
          parallel_agents: !!settings.parallelAgents,
          background_model: app.capabilities.backgroundModel.available && !!settings.backgroundModel,
        }),
      });
    } catch (err) {
      console.warn('[otto] could not persist config on the server', err);
    }
  }, 500);
}

function onControlChanged({ persistServer = true, pushSocket = true } = {}) {
  saveSettings();
  updateModelBadge();
  updatePersonalityNote();
  if (pushSocket) sendJson(configPatch());
  if (persistServer) scheduleConfigPersist();
}

async function previewVoice() {
  const button = $('preview-voice');
  const status = $('preview-status');
  const voice = settings.voice;
  if (!voice) {
    toast('Pick a voice first.', 'warn');
    return;
  }
  button.disabled = true;
  status.textContent = 'generating…';
  try {
    const response = await authFetch(`/api/voice-sample?voice=${encodeURIComponent(voice)}`, { cache: 'no-store' });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail.slice(0, 160) || `HTTP ${response.status}`);
    }
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const audio = new Audio(url);
    audio.addEventListener('ended', () => URL.revokeObjectURL(url));
    audio.addEventListener('error', () => URL.revokeObjectURL(url));
    await audio.play();
    status.textContent = 'playing';
  } catch (err) {
    status.textContent = 'unavailable';
    toast(`Voice preview failed: ${err && err.message ? err.message : err}`, 'warn');
  } finally {
    button.disabled = false;
  }
}

/* ------------------------------------------------------------ connections -- */

/** Paint Settings > Connections from /api/bootstrap.connections or /api/connections. */
function applyConnections(payload) {
  if (!payload || typeof payload !== 'object') return;
  if (payload.google && typeof payload.google === 'object') {
    const google = payload.google;
    connections.google = {
      configured: google.configured === true,
      connected: google.connected === true,
      email: text(google.email),
      scopes: arrayOf(google.scopes).map((scope) => text(scope)).filter(Boolean),
      redirectUri: text(google.redirectUri) || `${location.origin}/oauth/google/callback`,
      lastError: google.lastError && typeof google.lastError === 'object' ? google.lastError : null,
    };
  }
  if (payload.slack && typeof payload.slack === 'object') {
    const slack = payload.slack;
    connections.slack = {
      configured: slack.configured === true || !!text(slack.team),
      team: text(slack.team) || null,
    };
  }
  renderConnections();
}

function renderConnections() {
  const google = connections.google;
  googleBadge.textContent = google.connected ? (google.email || 'Connected') : 'Not connected';
  googleBadge.dataset.state = google.connected ? 'on' : 'off';
  googleBadge.title = google.connected && google.scopes.length ? google.scopes.join('\n') : '';
  googleConnect.hidden = google.connected;
  googleConnect.disabled = !google.configured;
  googleConnect.title = google.configured ? '' : 'The OAuth client is not configured on the server yet.';
  googleDisconnect.hidden = !google.connected;
  googleConfigNote.hidden = google.configured;
  if (googleRedirectUri) googleRedirectUri.textContent = google.redirectUri;

  // a failed connect leaves the exact reason from the server on the card
  const lastError = google.lastError;
  if (lastError && lastError.detail) {
    googleErrorNote.hidden = false;
    googleErrorNote.textContent = lastError.detail;
    googleErrorNote.dataset.level = 'error';
  } else {
    googleErrorNote.hidden = true;
    googleErrorNote.textContent = '';
  }

  const slack = connections.slack;
  slackBadge.textContent = slack.team || (slack.configured ? 'Connected' : 'Not connected');
  slackBadge.dataset.state = slack.configured ? 'on' : 'off';
  slackRemoveBtn.hidden = !slack.configured;
  slackToken.placeholder = slack.configured ? 'Paste a new token to replace' : 'xoxb-…';
}

async function copyGoogleRedirect() {
  const value = connections.google.redirectUri || `${location.origin}/oauth/google/callback`;
  try {
    await navigator.clipboard.writeText(value);
    toast('Redirect URI copied', 'ok');
  } catch (err) {
    // clipboard needs a secure context; select the text so it can be copied by hand
    const node = googleRedirectUri;
    if (node) {
      const range = document.createRange();
      range.selectNodeContents(node);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
    }
    toast('Select the highlighted URI and copy it', 'info');
  }
}

/** Workspace MCP: did Google's servers hand us tools for this account? */
function applyMcp(payload) {
  if (!payload || typeof payload !== 'object') return;
  app.mcp = {
    active: payload.active === true,
    tools: num(payload.tools, 0),
    servers: arrayOf(payload.servers),
    notices: arrayOf(payload.notices).map((n) => text(n)).filter(Boolean),
    mode: text(payload.mode),
    lastError: payload.lastError && typeof payload.lastError === 'object' ? payload.lastError : null,
    authoritative: payload.authoritative === true,
    strict: payload.strict === true || payload.authoritative === true,
  };
  renderMcp();
}

function renderMcp() {
  const mcp = app.mcp;
  const failing = mcp.servers.filter((server) => server && server.ok === false);
  const toolLabel = `${mcp.tools} tool${mcp.tools === 1 ? '' : 's'}`;
  if (mcp.lastError && mcp.lastError.detail) {
    // tools exist and the account is connected, but Google refuses the calls
    mcpBadge.textContent = 'Refused by Google';
    mcpBadge.dataset.state = 'off';
  } else if (mcp.active) {
    mcpBadge.textContent = toolLabel;
    mcpBadge.dataset.state = 'on';
  } else if (failing.length) {
    mcpBadge.textContent = 'Unavailable';
    mcpBadge.dataset.state = 'off';
  } else if (mcp.tools > 0) {
    // the servers answered, but this account has not connected Google yet
    mcpBadge.textContent = toolLabel;
    mcpBadge.dataset.state = 'off';
  } else {
    mcpBadge.textContent = mcp.mode === 'rest' ? 'Off (REST mode)' : 'Not connected';
    mcpBadge.dataset.state = 'off';
  }

  const parts = [];
  for (const server of mcp.servers) {
    if (!server) continue;
    const label = text(server.name) || 'server';
    parts.push(server.ok ? `${label}: ${num(server.tools, 0)} tools` : `${label}: ${text(server.reason) || 'error'}`);
  }
  const strict = app.mcp.strict === true;
  if (refusal) {
    mcpNote.textContent = 'Google is connected, but its MCP server is refusing these calls:';
  } else if (mcp.active) {
    mcpNote.textContent = strict
      ? 'Gmail and Calendar run only through Google\u2019s MCP servers \u2014 no legacy REST APIs.'
      : 'MCP first \u2014 when Google\u2019s MCP servers refuse a call, the classic Gmail/Calendar APIs answer it (mcp.allow_rest_fallback in config).';
  } else if (mcp.tools > 0) {
    mcpNote.textContent = `Google\u2019s MCP servers answered (${toolLabel}) \u2014 connect Google to use them.`;
  } else if (strict) {
    mcpNote.textContent = 'MCP only: no legacy REST APIs are used, so these tools need Google\u2019s MCP servers to accept this project.';
  } else {
    mcpNote.textContent = 'Reading mail and managing events needs a working MCP connection.';
  }

  const refusal = mcp.lastError && mcp.lastError.detail ? text(mcp.lastError.detail) : '';
  const detail = refusal || mcp.notices[0] || (failing[0] ? text(failing[0].detail) : '');
  mcpDetail.hidden = !detail;
  mcpDetail.textContent = detail;
  mcpDetail.dataset.level = detail && (!mcp.active || refusal) ? 'error' : '';

  const summary = parts.join(' · ');
  mcpDetail.title = summary;
  mcpCheckBtn.disabled = false;
}

/** One real read call per server — the only honest check that the token works. */
async function testMcp() {
  mcpTestBtn.disabled = true;
  const label = mcpTestBtn.textContent;
  mcpTestBtn.textContent = 'Testing…';
  try {
    const response = await authFetch('/api/mcp/selftest', { method: 'POST' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    const servers = result.servers && typeof result.servers === 'object' ? Object.entries(result.servers) : [];
    const failures = servers.filter(([, value]) => value && value.ok === false);
    if (result.ok) {
      const detail = servers.map(([name, value]) => `${name}: ${text(value.tool) || 'ok'}`).join(' · ');
      mcpDetail.hidden = false;
      mcpDetail.dataset.level = '';
      mcpDetail.textContent = `Live check passed — ${detail}`;
      toast('MCP is working', 'ok');
    } else {
      const first = failures[0] || [];
      const message = text((first[1] && first[1].detail) || result.detail || 'the check did not pass');
      mcpDetail.hidden = false;
      mcpDetail.dataset.level = 'error';
      mcpDetail.textContent = message;
      toast(text(first[0]) ? `${first[0]}: check failed` : 'MCP check failed', 'error');
    }
    if (result.status) applyMcp(result.status);
  } catch (err) {
    if (app.sessionEnded) return;
    toast(`MCP check failed: ${err.message}`, 'error');
  } finally {
    mcpTestBtn.disabled = false;
    mcpTestBtn.textContent = label;
  }
}

async function probeMcp() {
  mcpCheckBtn.disabled = true;
  mcpBadge.textContent = 'Checking…';
  try {
    const response = await authFetch('/api/mcp/probe', { method: 'POST' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    applyMcp(await response.json());
    toast(app.mcp.active ? `MCP ready — ${app.mcp.tools} tools` : 'MCP is not reachable yet', app.mcp.active ? 'ok' : 'error');
    await refreshBootstrapQuietly();
  } catch (err) {
    if (app.sessionEnded) return;
    toast(`MCP check failed: ${err.message}`, 'error');
  } finally {
    mcpCheckBtn.disabled = false;
  }
}

/** Re-read /api/bootstrap so newly discovered tools show up in the pickers. */
async function refreshBootstrapQuietly() {
  await loadBootstrap();
}

async function refreshConnections() {
  try {
    const response = await authFetch('/api/connections', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    applyConnections(await response.json());
  } catch (err) {
    if (app.sessionEnded) return;
    console.warn('[otto] could not read connections', err);
  }
}

/** Full page navigation on purpose: the server 302s to Google and back to /?connected=google. */
function connectGoogle() {
  player.unlock();
  location.assign('/oauth/google/start');
}

async function disconnectGoogle() {
  googleDisconnect.disabled = true;
  try {
    const response = await authFetch('/api/connections/google', { method: 'DELETE' });
    if (!response.ok) throw new Error(detailFrom(await readJson(response), `HTTP ${response.status}`));
    toast('Google disconnected', 'ok');
    await refreshConnections();
  } catch (err) {
    if (!app.sessionEnded) toast(`Could not disconnect Google: ${err.message}`, 'error');
  } finally {
    googleDisconnect.disabled = false;
  }
}

async function saveSlackToken() {
  const token = slackToken.value.trim();
  if (!token) {
    toast('Paste a Slack bot token first.', 'warn');
    slackToken.focus({ preventScroll: true });
    return;
  }
  slackSaveBtn.disabled = true;
  try {
    const response = await authFetch('/api/connections/slack', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ token }),
    });
    if (!response.ok) throw new Error(detailFrom(await readJson(response), `HTTP ${response.status}`));
    slackToken.value = ''; // tokens are write-only: never render one back
    toast('Slack connected', 'ok');
    await refreshConnections();
  } catch (err) {
    if (!app.sessionEnded) toast(`Could not save the Slack token: ${err.message}`, 'error');
  } finally {
    slackSaveBtn.disabled = false;
  }
}

async function removeSlackToken() {
  slackRemoveBtn.disabled = true;
  try {
    const response = await authFetch('/api/connections/slack', { method: 'DELETE' });
    if (!response.ok) throw new Error(detailFrom(await readJson(response), `HTTP ${response.status}`));
    slackToken.value = '';
    toast('Slack disconnected', 'ok');
    await refreshConnections();
  } catch (err) {
    if (!app.sessionEnded) toast(`Could not remove the Slack token: ${err.message}`, 'error');
  } finally {
    slackRemoveBtn.disabled = false;
  }
}

async function readJson(response) {
  try {
    return await response.json();
  } catch (err) {
    return null;
  }
}

/** FastAPI answers with {detail}; show it verbatim. */
function detailFrom(payload, fallback) {
  if (!payload || typeof payload !== 'object') return fallback;
  if (typeof payload.detail === 'string' && payload.detail.trim()) return payload.detail;
  if (Array.isArray(payload.detail) && payload.detail.length) {
    const parts = payload.detail.map((item) => text(item && item.msg)).filter(Boolean);
    if (parts.length) return parts.join(' · ');
  }
  return fallback;
}

/** /oauth/google/start returns to /?connected=google, or /?error=google_<reason>. */
function consumeAuthQuery() {
  const params = new URLSearchParams(location.search);
  const connected = params.get('connected');
  const error = params.get('error');
  if (!connected && !error) return;
  params.delete('connected');
  params.delete('error');
  const query = params.toString();
  history.replaceState({}, '', `${location.pathname}${query ? `?${query}` : ''}${location.hash}`);

  if (connected === 'google') {
    toast('Google connected', 'ok');
    loadBootstrap();
    return;
  }
  if (error) {
    const reason = error.replace(/^google_?/, '').replace(/_/g, ' ');
    toast(`Google connection failed${reason ? ` · ${reason}` : ''} — see Settings → Connections`, 'error');
    // pull the exact sentence the server recorded, then repaint the card
    refreshConnections();
  }
}

/* -------------------------------------------------------------- settings -- */

function openDrawer(sectionKey) {
  setAccountMenu(false);
  drawerEl.classList.add('open');
  drawerEl.setAttribute('aria-hidden', 'false');
  scrimEl.hidden = false;
  window.requestAnimationFrame(() => scrimEl.classList.add('show'));
  refreshDevices();
  const target = sectionKey ? document.getElementById(CONNECTION_SECTIONS[sectionKey] || sectionKey) : null;
  if (target) target.scrollIntoView({ block: 'start' });
  window.setTimeout(() => {
    const first = drawerEl.querySelector('.drawer-head .icon-btn');
    if (first && !target) first.focus({ preventScroll: true });
  }, 180);
}

function closeDrawer() {
  drawerEl.classList.remove('open');
  drawerEl.setAttribute('aria-hidden', 'true');
  scrimEl.classList.remove('show');
  window.setTimeout(() => {
    if (!drawerEl.classList.contains('open')) scrimEl.hidden = true;
  }, 220);
  accountBtn.focus({ preventScroll: true });
}

/* ------------------------------------------------------------------ wiring -- */

function sendTypedText() {
  const value = textInput.value.trim();
  if (!value) return;
  if (!app.connected) {
    toast('Not connected to the server yet.', 'warn');
    return;
  }
  sendJson({ type: 'text', text: value });
  textInput.value = '';
  clearCaption();
  updateSendVisibility();
}

/** The send arrow only exists while the field holds text. */
function updateSendVisibility() {
  composerEl.classList.toggle('has-text', textInput.value.trim().length > 0);
}

function wireUi() {
  micBtn.addEventListener('click', () => {
    player.unlock();
    toggleMute();
  });
  orbHit.addEventListener('click', () => {
    player.unlock();
    enableMic();
  });
  orbEl.addEventListener('click', (event) => {
    if (event.target === orbHit) return;
    if (!mic.running) enableMic();
  });

  $('composer').addEventListener('submit', (event) => {
    event.preventDefault();
    sendTypedText();
  });
  stopBtn.addEventListener('click', () => stopEverything());
  textInput.addEventListener('input', updateSendVisibility);

  accountBtn.addEventListener('click', onAccountButtonClick);
  endBtn.addEventListener('click', () => endConversation());
  actionsBadge.addEventListener('click', () => location.assign('/dashboard'));
  $('menu-end').addEventListener('click', () => endConversation());
  $('menu-settings').addEventListener('click', () => openDrawer('settings'));
  $('menu-connections').addEventListener('click', () => openDrawer('connections'));
  $('menu-signout').addEventListener('click', () => signOut());
  $('sheet-signout').addEventListener('click', () => signOut());
  document.addEventListener('pointerdown', (event) => {
    if (accountMenu.hidden) return;
    if (accountBtn.contains(event.target) || accountMenu.contains(event.target)) return;
    setAccountMenu(false);
  });

  googleRedirectCopy?.addEventListener('click', copyGoogleRedirect);
  mcpCheckBtn?.addEventListener('click', probeMcp);
  mcpTestBtn?.addEventListener('click', testMcp);
  googleConnect.addEventListener('click', connectGoogle);
  googleDisconnect.addEventListener('click', disconnectGoogle);
  slackSaveBtn.addEventListener('click', saveSlackToken);
  slackRemoveBtn.addEventListener('click', removeSlackToken);
  slackToken.addEventListener('keydown', (event) => {
    if (event.key !== 'Enter') return;
    event.preventDefault();
    saveSlackToken();
  });

  $('drawer-close').addEventListener('click', closeDrawer);
  scrimEl.addEventListener('click', closeDrawer);
  $('btn-reconnect').addEventListener('click', () => forceReconnect());
  $('preview-voice').addEventListener('click', previewVoice);
  $('edit-personality').addEventListener('click', openPersonalityEditor);
  $('persona-close').addEventListener('click', () => $('persona-dialog').close());
  $('persona-cancel').addEventListener('click', () => $('persona-dialog').close());
  $('persona-form').addEventListener('submit', savePersonality);

  $('set-model').addEventListener('change', (event) => {
    settings.model = event.target.value;
    onControlChanged();
  });
  $('set-thinking')?.addEventListener('change', (event) => {
    settings.thinking = event.target.value;
    onControlChanged();
  });
  $('set-personality').addEventListener('change', (event) => {
    settings.personality = event.target.value;
    onControlChanged();
  });
  $('set-voice').addEventListener('change', (event) => {
    settings.voice = event.target.value;
    onControlChanged();
    $('preview-status').textContent = '';
  });
  $('set-instruct').addEventListener('input', (event) => {
    settings.instruct = event.target.value;
    saveSettings(); // persisted locally on every keystroke, pushed on blur
  });
  $('set-instruct').addEventListener('change', () => {
    onControlChanged();
  });

  $('set-sensitivity').addEventListener('input', (event) => {
    settings.sensitivity = clamp(num(event.target.value, 55), 0, 100);
    $('sen-value').textContent = String(Math.round(settings.sensitivity));
    onControlChanged({ persistServer: false, pushSocket: false });
  });
  $('set-silence').addEventListener('input', (event) => {
    settings.silenceMs = clamp(num(event.target.value, 550), 200, 2000);
    localSilenceOverride = true;
    $('silence-value').textContent = `${Math.round(settings.silenceMs)} ms`;
    onControlChanged({ persistServer: false, pushSocket: false });
  });
  $('set-interrupt').addEventListener('input', (event) => {
    settings.interruptMs = clamp(num(event.target.value, 250), 150, 600);
    $('interrupt-value').textContent = `${Math.round(settings.interruptMs)} ms`;
    onControlChanged({ persistServer: false });
  });
  $('set-device').addEventListener('change', async (event) => {
    settings.deviceId = event.target.value;
    onControlChanged({ persistServer: false, pushSocket: false });
    if (mic.running) {
      await mic.stop();
      app.muted = true;
      updateMicUi();
      await enableMic();
    }
  });
  $('set-transcript-text').addEventListener('change', async (event) => {
    const wanted = event.target.checked;
    if (wanted && !BrowserASR.available) {
      event.target.checked = false;
      settings.sendTranscriptText = false;
      saveSettings();
      toast('This browser has no speech recognition — audio streaming stays on.', 'warn');
      return;
    }
    settings.sendTranscriptText = wanted;
    saveSettings();
    if (wanted) {
      if (!mic.running) await enableMic();
      const started = browserAsr.start();
      mic.asrMode = started;
      if (!started) {
        settings.sendTranscriptText = false;
        event.target.checked = false;
        saveSettings();
        toast('Could not start browser speech recognition.', 'warn');
      }
    } else {
      browserAsr.stop();
      mic.asrMode = false;
    }
  });
  $('set-debug').addEventListener('change', (event) => {
    settings.debug = event.target.checked;
    saveSettings();
    metricsEl.hidden = !settings.debug;
    renderMetrics();
    console.info(`[otto] debug ${settings.debug ? 'on' : 'off'}`);
  });
  $('set-parallel-agents').addEventListener('change', (event) => {
    settings.parallelAgents = event.target.checked;
    onControlChanged();
  });
  $('set-background-model').addEventListener('change', (event) => {
    settings.backgroundModel = event.target.checked;
    onControlChanged();
  });

  activityStrip.addEventListener('click', () => focusRunningJob());

  document.querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      textInput.value = chip.dataset.prompt || chip.textContent || '';
      sendTypedText();
      textInput.focus();
    });
  });

  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);
  window.addEventListener('blur', () => mic.holdEnd());

  const unlock = () => {
    player.unlock();
    if (mic.ctx && mic.ctx.state === 'suspended') mic.ctx.resume().catch(() => {});
  };
  document.addEventListener('pointerdown', unlock, { once: true });
  document.addEventListener('keydown', unlock, { once: true });

  window.addEventListener('error', (event) => {
    console.error('[otto] uncaught error', event.error || event.message);
    toast(`Unexpected error: ${event.message || 'see console'}`, 'error');
  });
  window.addEventListener('unhandledrejection', (event) => {
    console.error('[otto] unhandled rejection', event.reason);
    toast(`Unexpected error: ${event.reason && event.reason.message ? event.reason.message : event.reason}`, 'error');
  });

  player.onDrained = (seq) => {
    // tell the server the audio was actually heard; the server decides when the
    // turn is over (a pause between sentences must not flip the UI to idle)
    sendJson({ type: 'playback.idle', seq });
  };
}

function onKeyDown(event) {
  if (event.shiftKey && event.key === 'Escape') {
    // Shift+Esc ends the conversation and goes to the dashboard, whatever else is open.
    event.preventDefault();
    endConversation();
    return;
  }
  if ((event.metaKey || event.ctrlKey) && (event.key === 'k' || event.key === 'K')) {
    event.preventDefault();
    if (drawerEl.classList.contains('open')) closeDrawer();
    else openDrawer('settings');
    return;
  }
  if (event.key === 'Escape') {
    if (!accountMenu.hidden) {
      event.preventDefault();
      setAccountMenu(false);
      accountBtn.focus({ preventScroll: true });
      return;
    }
    if (drawerEl.classList.contains('open')) {
      event.preventDefault();
      closeDrawer();
      return;
    }
    event.preventDefault();
    stopEverything();
    return;
  }
  if (isTypingTarget(event.target)) return;

  if (app.pendingConfirm) {
    if (event.key === 'y' || event.key === 'Y') {
      event.preventDefault();
      answerConfirm(true);
      return;
    }
    if (event.key === 'n' || event.key === 'N') {
      event.preventDefault();
      answerConfirm(false);
      return;
    }
  }

  if (event.code === 'Space' || event.key === ' ') {
    if (event.repeat) {
      event.preventDefault();
      return;
    }
    if (!mic.running || app.muted) return;
    event.preventDefault();
    mic.holdStart();
  }
}

function onKeyUp(event) {
  if (event.code === 'Space' || event.key === ' ') {
    if (mic.forceHold) {
      event.preventDefault();
      mic.holdEnd();
    }
  }
}

/* ---------------------------------------------------------------- bootstrap -- */

async function loadBootstrap() {
  try {
    const response = await authFetch('/api/bootstrap', { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    app.models = arrayOf(data.models);
    app.voices = arrayOf(data.voices);
    app.personalities = arrayOf(data.personalities);
    if (data.config) applyServerConfig(data.config);
    // capabilities come after the config: an unavailable *-background variant clears the toggle
    applyCapabilities(data.capabilities);
    if (data.status) applyHealth(data.status);
    if (data.user) paintUser(data.user);
    if (data.connections) applyConnections(data.connections);
    if (data.actions) paintActionsBadge(data.actions.open);
    if (data.mcp) applyMcp(data.mcp);
    populateSelects();
    applySettingsToUi();
    await refreshDevices();
  } catch (err) {
    if (app.sessionEnded) return;
    console.warn('[otto] bootstrap unavailable, relying on "ready"', err);
    // The websocket "ready" message carries config/personality, so the UI still
    // works; selects fall back to whatever the server session reports.
  }
}

/* ------------------------------------------------------------------ dev hook -- */

function installDevHook() {
  window.__ottoDev = {
    send(message) {
      return sendJson(message);
    },
    /** Feed any server message through the normal handler (protocol debugging). */
    server(message) {
      handleServerMessage(message);
    },
    /** Pretend the VAD just heard speech onset: drives barge-in + utterance.start. */
    simulateSpeechOnset() {
      bargeIn('dev hook');
      mic.handlers.onSpeechStart([]);
      app.utteranceOpen = false;
      return true;
    },
    /**
     * Drive the VAD without a device: feed `ms` of audio at `rms` (0..1), paced one capture
     * chunk at a time through the same tick the microphone uses. Resolves when the feed ends.
     */
    simulateMicLevel(rms, ms) {
      return mic.simulateLevel(rms, ms);
    },
    /** Stream text into the transcript exactly like assistant.delta would. */
    injectAssistantText(value, id) {
      const messageId = id || `dev-${Date.now()}`;
      const full = text(value);
      const words = full.match(/\S+\s*/g) || [];
      handleServerMessage({ type: 'assistant.delta', id: messageId, text: '' });
      let index = 0;
      const timer = window.setInterval(() => {
        if (index >= words.length) {
          window.clearInterval(timer);
          handleServerMessage({ type: 'assistant.message', id: messageId, text: full, spoken: true });
          return;
        }
        handleServerMessage({ type: 'assistant.delta', id: messageId, text: words[index++] });
      }, 45);
      return messageId;
    },
    /** Render a tool card (running -> finished) with whatever display payload you pass. */
    simulateTool(spec = {}) {
      const id = spec.id || `dev-tool-${++devSeq}`;
      const name = spec.name || 'gmail_search';
      handleServerMessage({
        type: 'tool.call',
        id,
        name,
        args: spec.args || {},
        category: spec.category || 'email',
      });
      window.setTimeout(() => {
        handleServerMessage({
          type: 'tool.result',
          id,
          name,
          ok: spec.ok !== false,
          summary: spec.summary || 'Simulated tool result.',
          display: spec.display,
          error: spec.error,
          ms: spec.ms == null ? 312 : spec.ms,
        });
      }, spec.delayMs == null ? 420 : spec.delayMs);
      return id;
    },
    /** Render a confirmation card and leave it pending for Y/N or a click. */
    simulateConfirm(spec = {}) {
      const id = spec.id || `dev-confirm-${++devSeq}`;
      handleServerMessage({
        type: 'confirm.request',
        id,
        tool: spec.tool || 'gmail_send',
        name: spec.name || 'Send email',
        args: spec.args || {},
        summary: spec.summary || 'Send this email?',
        preview: spec.preview,
        danger: !!spec.danger,
      });
      return id;
    },
    /** Push synthetic AUD1 frames through the player (no microphone needed). */
    simulateAudio(spec = {}) {
      const sampleRate = spec.sampleRate || 24000;
      const seq = spec.seq || player.seqHigh + 1;
      const seconds = spec.seconds || 1.2;
      const freq = spec.freq || 220;
      const amplitude = spec.amplitude == null ? 0.22 : spec.amplitude;
      // leadSilenceMs pads the first buffer with silence so the client-side trim can be observed
      const leadFrames = spec.leadSilenceMs ? Math.round((spec.leadSilenceMs / 1000) * sampleRate) : 0;
      const chunkFrames = 1200;
      setPhase('speaking', 'simulated audio');
      player.handleStart(seq, sampleRate);
      const total = Math.round(seconds * sampleRate);
      let offset = 0;
      let index = 0;
      const step = () => {
        if (offset >= total) {
          player.handleEnd(seq);
          return;
        }
        const audioFrames = Math.min(chunkFrames, total - offset);
        const lead = index === 0 ? leadFrames : 0;
        const frames = audioFrames + lead;
        const buffer = new ArrayBuffer(AUDIO_HEADER_BYTES + frames * 2);
        const view = new DataView(buffer);
        view.setUint8(0, AUDIO_MAGIC[0]);
        view.setUint8(1, AUDIO_MAGIC[1]);
        view.setUint8(2, AUDIO_MAGIC[2]);
        view.setUint8(3, AUDIO_MAGIC[3]);
        view.setUint32(4, seq, true);
        view.setUint32(8, sampleRate, true);
        view.setUint32(12, index, true);
        const pcm = new Int16Array(buffer, AUDIO_HEADER_BYTES, frames);
        for (let i = 0; i < frames; i++) {
          if (i < lead) continue; // silent lead-in
          const n = offset + i - lead;
          const envelope = Math.min(1, Math.min(n, total - n) / (sampleRate * 0.04));
          const value = Math.sin((2 * Math.PI * freq * n) / sampleRate) * amplitude * envelope;
          pcm[i] = Math.max(-32768, Math.min(32767, Math.round(value * 32767)));
        }
        player.handleFrame(buffer);
        offset += audioFrames;
        index += 1;
        window.setTimeout(step, ((audioFrames / sampleRate) * 1000) / 2);
      };
      step();
      return seq;
    },
    /** Render a job card straight from a job.update payload (one card per id). */
    simulateJob(spec = {}) {
      const id = spec.id || `dev-job-${++devSeq}`;
      handleServerMessage({
        type: 'job.update',
        id,
        title: spec.title || 'background task',
        kind: spec.kind || 'subagent',
        status: spec.status || 'running',
        ms: spec.ms,
        result: spec.result,
        error: spec.error,
      });
      return id;
    },
    /** Replay queued -> running -> done for one job id, ~2.5 s end to end. */
    simulateJobEvent(seq) {
      const id = `job_sim${num(seq, ++devSeq)}`;
      const title = 'Summarising the Q3 thread';
      const send = (status, extra = {}) =>
        handleServerMessage({ type: 'job.update', id, title, kind: 'subagent', status, ms: 0, ...extra });
      send('queued', { ms: null });
      window.setTimeout(() => send('running', { ms: null }), 700);
      window.setTimeout(
        () =>
          send('done', {
            ms: 1840,
            result:
              'Sarah wants the sync moved to Thursday. Two threads mention the Q3 deck: the finance review ' +
              'is ready, the product review is still open, and the design hand-off has not been booked yet. ' +
              'Nothing else needs a reply today, but the pricing thread now has three unanswered questions, ' +
              'and the vendor comparison in #procurement should be read before the Thursday sync so the ' +
              'agenda can carry a decision instead of another round of questions.',
          }),
        2400,
      );
      return id;
    },
    /** Inject a finished `subagents` tool card (three parallel subagents). */
    simulateSubagents() {
      const id = `dev-subagents-${++devSeq}`;
      const display = {
        kind: 'subagents',
        elapsedMs: 4200,
        items: [
          {
            title: 'invoice 1',
            ok: true,
            steps: 2,
            tools: ['gmail_search', 'gmail_read'],
            ms: 3100,
            result: 'Acme raised the unit price by 4% and kept the 30 day terms.',
          },
          {
            title: 'invoice 2',
            ok: true,
            steps: 3,
            tools: ['gmail_search', 'gmail_read'],
            ms: 4050,
            result:
              'Northwind matched the old price but shortened support to business hours. The discount only ' +
              'applies if the contract is signed this quarter, so it is worth comparing against the Acme ' +
              'offer before replying.',
          },
          { title: 'invoice 3', ok: false, steps: 1, tools: ['gmail_search'], ms: 900, error: 'no matching mail' },
        ],
      };
      handleServerMessage({
        type: 'tool.call',
        id,
        name: 'delegate_task',
        args: { tasks: [{ title: 'invoice 1' }, { title: 'invoice 2' }, { title: 'invoice 3' }] },
        category: 'agents',
      });
      window.setTimeout(() => {
        handleServerMessage({
          type: 'tool.result',
          id,
          name: 'delegate_task',
          ok: true,
          summary: '3 subagents finished in 4.2s.',
          display,
          ms: 4200,
        });
      }, 420);
      return id;
    },
    /** Empty the transcript (keeps the socket and settings untouched). */
    clear() {
      clear(transcriptEl);
      assistantMessages.clear();
      toolCards.clear();
      confirmCards.clear();
      jobCards.clear();
      jobActivity.unseen = 0;
      renderActivity();
      app.pendingConfirm = null;
      appEl.classList.remove('has-messages');
      logEmpty.hidden = false;
    },
    state() {
      return {
        connected: app.connected,
        phase: app.phase,
        session: app.sessionId,
        mic: {
          running: mic.running,
          mode: mic.mode,
          muted: app.muted,
          open: mic.open,
          threshold: Number(mic.threshold().toFixed(4)),
        },
        level: Number(app.micLevel.toFixed(4)),
        vadStats: { ...app.vadStats },
        player: {
          scheduled: player.scheduled.length,
          openSegments: player.openSegments.size,
          minSeq: player.minSeq,
          joins: player.joins,
          lastTrimSamples: player.lastTrimSamples,
        },
        jobs: [...jobCards.values()].map((entry) => ({ id: entry.card.dataset.jobId, status: entry.status })),
        activity: { hidden: activityStrip.hidden, text: activityText.textContent, unseen: jobActivity.unseen },
        capabilities: app.capabilities,
        pendingConfirm: app.pendingConfirm ? app.pendingConfirm.id : null,
        settings: { ...settings },
      };
    },
  };
}

/* ---------------------------------------------------------------- lifecycle -- */

function boot() {
  buildMeter();
  wireUi();
  populateSelects();
  applySettingsToUi();
  installDevHook();
  updateMicUi();
  updateSendVisibility();
  renderMetrics();
  paintUser(null);
  setConn('connecting', 'Connecting');
  setPhase('idle', '');
  window.requestAnimationFrame(tick);
  loadBootstrap().then(consumeAuthQuery);
  connectSocket();
  refreshHealth();
  window.clearInterval(healthTimer);
  healthTimer = window.setInterval(refreshHealth, HEALTH_INTERVAL_MS);
  autoStartMic();
}

safe(boot, 'starting Otto');
