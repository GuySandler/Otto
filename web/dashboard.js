/* ============================================================================
 * Otto — dashboard (vanilla ES module, no dependencies).
 *
 * The calm "after the call" view: the things a tool created for the user that
 * nobody has finished yet. The voice app hands over to /dashboard (End button,
 * account menu, Shift+Esc, or the actions badge).
 *
 * Pieces
 *   Actions   GET /api/actions?status=open, POST /api/actions/{id} {status}
 *   Recent    <details>, lazily GET /api/actions?status=done on first expand
 *   Account   /api/auth/me summary, avatar menu, POST /api/auth/logout -> /login
 *
 * A 401 from any call means the cookie session is gone: go to /login, never retry.
 * Nothing polls; the open list refreshes on window focus and after a mutation.
 * ========================================================================== */

function byId(id) {
  return document.getElementById(id);
}

const accountBtn = byId('account-btn');
const accountMenu = byId('account-menu');
const accountInitials = byId('account-initials');
const accountEmail = byId('account-email');
const summaryEl = byId('dash-summary');
const actionsTitle = byId('actions-title');
const actionList = byId('action-list');
const actionsEmpty = byId('actions-empty');
const actionsError = byId('actions-error');
const recentEl = byId('recent');
const doneList = byId('done-list');
const doneEmpty = byId('done-empty');

// Action kinds the server knows today; anything else is chipped uppercased.
const KIND_CHIP = { draft: 'DRAFT', email: 'EMAIL', event: 'EVENT' };
const OPEN_LABEL = { draft: 'Open in Gmail ↗', email: 'Open in Gmail ↗', event: 'Open in Calendar ↗' };
const DONE_LIMIT = 5;

const state = {
  user: null,
  open: [],
  done: [],
  counts: { open: 0 },
  doneLoaded: false,
  busy: new Set(),
};

/* ------------------------------------------------------------------ utils -- */

function text(value) {
  return value == null ? '' : String(value);
}

function num(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function arrayOf(value) {
  return Array.isArray(value) ? value : [];
}

function clear(node) {
  while (node && node.firstChild) node.removeChild(node.firstChild);
}

function el(tag, options = {}, children = []) {
  const node = document.createElement(tag);
  if (options.class) node.className = options.class;
  if (options.text != null) node.textContent = String(options.text);
  for (const [key, value] of Object.entries(options.attrs || {})) {
    if (value == null || value === false) continue;
    node.setAttribute(key, value === true ? '' : String(value));
  }
  for (const [key, handler] of Object.entries(options.on || {})) node.addEventListener(key, handler);
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

function plural(count, word) {
  return `${count} ${word}${count === 1 ? '' : 's'}`;
}

/** "2 minutes ago" for fresh things, an absolute stamp once a week has passed. */
function relativeTime(value) {
  const at = num(value) * 1000;
  if (!at) return '';
  const diff = Date.now() - at;
  if (diff < 0) return 'just now';
  if (diff < 45000) return 'just now';
  const minutes = Math.round(diff / 60000);
  if (minutes < 60) return `${plural(minutes, 'minute')} ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${plural(hours, 'hour')} ago`;
  const days = Math.round(hours / 24);
  if (days < 7) return `${plural(days, 'day')} ago`;
  return stamp(value);
}

/** Local, compact, unambiguous: "Nov 25, 14:32". */
function stamp(value) {
  const at = num(value) * 1000;
  if (!at) return '';
  try {
    return new Date(at).toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch (err) {
    return new Date(at).toISOString().slice(0, 16).replace('T', ' ');
  }
}

function isHttpUrl(value) {
  return /^https?:\/\//i.test(text(value));
}

/* -------------------------------------------------------------------- api -- */

/** fetch() plus the session guard: a 401 sends the browser to /login and stops the caller. */
async function api(path, init = {}) {
  const response = await fetch(path, {
    cache: 'no-store',
    ...init,
    headers: { Accept: 'application/json', ...(init.headers || {}) },
  });
  if (response.status === 401) {
    location.assign('/login');
    throw new Error('session ended');
  }
  return response;
}

/** FastAPI puts the useful message in {detail}; fall back to the status code. */
async function errorText(response, fallback) {
  try {
    const payload = await response.json();
    if (payload && typeof payload.detail === 'string' && payload.detail.trim()) return payload.detail;
  } catch (err) {
    /* not JSON */
  }
  return fallback || `HTTP ${response.status}`;
}

function showError(message) {
  actionsError.textContent = text(message);
  actionsError.hidden = !message;
}

/* ------------------------------------------------------------------ paint -- */

function paintAccount(user) {
  const address = text(user && user.email);
  accountInitials.textContent = text(user && user.initials) || address.charAt(0).toUpperCase() || '·';
  accountEmail.textContent = address || '—';
  accountEmail.title = address;
}

function paintSummary() {
  const parts = [];
  const address = text(state.user && state.user.email);
  if (address) parts.push(`Signed in as ${address}`);
  parts.push(`${plural(num(state.counts.open), 'action')} waiting`);
  summaryEl.textContent = parts.join(' · ');
}

function paintTitle() {
  const count = num(state.counts.open, state.open.length);
  actionsTitle.textContent = `Actions required (${count})`;
}

function actionCard(action) {
  const id = text(action.id);
  const kind = text(action.kind);
  const card = el('article', { class: 'acard', attrs: { 'data-id': id } });

  card.append(
    el('div', { class: 'acard-head' }, [
      el('span', { class: 'acard-kind', text: KIND_CHIP[kind] || text(kind || 'item').toUpperCase() }),
      el('span', {
        class: 'acard-time',
        text: relativeTime(action.createdAt),
        attrs: { title: stamp(action.createdAt) },
      }),
    ]),
  );
  card.append(el('h3', { class: 'acard-title', text: text(action.title) || 'Untitled action' }));
  if (text(action.subtitle)) card.append(el('p', { class: 'acard-sub', text: action.subtitle }));
  if (text(action.detail)) card.append(el('p', { class: 'acard-detail', text: action.detail }));

  const row = el('div', { class: 'acard-actions' });
  if (isHttpUrl(action.url)) {
    row.append(
      el('a', {
        class: 'btn primary',
        text: OPEN_LABEL[kind] || 'Open ↗',
        attrs: { href: action.url, target: '_blank', rel: 'noreferrer noopener' },
      }),
    );
  }
  row.append(
    el('button', {
      class: 'btn',
      text: 'Mark done',
      attrs: { type: 'button' },
      on: { click: () => setStatus(id, 'done', card) },
    }),
  );
  row.append(
    el('button', {
      class: 'btn ghost',
      text: 'Dismiss',
      attrs: { type: 'button' },
      on: { click: () => setStatus(id, 'dismissed', card) },
    }),
  );
  card.append(row);
  return card;
}

function doneRow(action) {
  const id = text(action.id);
  const row = el('div', { class: 'done-row' });
  row.append(el('span', { class: 'done-title', text: text(action.title) || 'Untitled action' }));
  row.append(el('span', { class: 'done-when', text: stamp(action.updatedAt || action.createdAt) }));
  row.append(
    el('button', {
      class: 'done-reopen',
      text: 'Reopen',
      attrs: { type: 'button' },
      on: { click: () => setStatus(id, 'open', row) },
    }),
  );
  return row;
}

function renderOpen() {
  clear(actionList);
  for (const action of state.open) actionList.append(actionCard(action));
  actionsEmpty.hidden = state.open.length > 0;
  paintTitle();
}

function renderDone() {
  clear(doneList);
  for (const action of state.done.slice(0, DONE_LIMIT)) doneList.append(doneRow(action));
  doneEmpty.hidden = state.done.length > 0;
}

/* ------------------------------------------------------------------ loads -- */

async function loadMe() {
  try {
    const response = await api('/api/auth/me');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    state.user = data && data.user ? data.user : null;
    paintAccount(state.user);
    paintSummary();
  } catch (err) {
    if (text(err && err.message) === 'session ended') return;
    console.warn('[otto] could not read the account', err);
  }
}

async function loadOpen() {
  try {
    const response = await api('/api/actions?status=open');
    if (!response.ok) throw new Error(await errorText(response));
    const data = await response.json();
    state.open = arrayOf(data.actions);
    state.counts.open = num(data.counts && data.counts.open, state.open.length);
    renderOpen();
    paintSummary();
    showError('');
  } catch (err) {
    if (text(err && err.message) === 'session ended') return;
    showError(`Could not load your actions: ${err.message}`);
  }
}

async function loadDone() {
  try {
    const response = await api('/api/actions?status=done');
    if (!response.ok) throw new Error(await errorText(response));
    const data = await response.json();
    state.done = arrayOf(data.actions);
    renderDone();
  } catch (err) {
    if (text(err && err.message) === 'session ended') return;
    showError(`Could not load the finished actions: ${err.message}`);
  }
}

/** The open list always, the done list only once it has been asked for. */
async function refresh() {
  await loadOpen();
  if (state.doneLoaded) await loadDone();
}

/* -------------------------------------------------------------- mutations -- */

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function setStatus(id, status, node) {
  if (!id || state.busy.has(id)) return;
  state.busy.add(id);
  node?.classList.add('leaving');
  try {
    const response = await api(`/api/actions/${encodeURIComponent(id)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    });
    if (!response.ok) throw new Error(await errorText(response));
    const data = await response.json();
    if (data && data.counts && data.counts.open != null) state.counts.open = num(data.counts.open);
    paintTitle();
    paintSummary();
    showError('');
    // let the card fade out before the list is rebuilt from the server
    await wait(node ? 180 : 0);
    await refresh();
  } catch (err) {
    if (text(err && err.message) === 'session ended') return;
    node?.classList.remove('leaving');
    showError(`Could not update that action: ${err.message}`);
  } finally {
    state.busy.delete(id);
  }
}

/* ---------------------------------------------------------------- account -- */

function setAccountMenu(open) {
  accountMenu.hidden = !open;
  accountBtn.setAttribute('aria-expanded', open ? 'true' : 'false');
}

async function signOut() {
  setAccountMenu(false);
  try {
    await fetch('/api/auth/logout', { method: 'POST', headers: { Accept: 'application/json' } });
  } catch (err) {
    console.warn('[otto] logout request failed, leaving anyway', err);
  }
  location.assign('/login');
}

function wireAccount() {
  accountBtn.addEventListener('click', () => setAccountMenu(accountMenu.hidden));
  byId('menu-signout').addEventListener('click', signOut);
  document.addEventListener('pointerdown', (event) => {
    if (accountMenu.hidden) return;
    if (accountBtn.contains(event.target) || accountMenu.contains(event.target)) return;
    setAccountMenu(false);
  });
  document.addEventListener('keydown', (event) => {
    if (event.key !== 'Escape' || accountMenu.hidden) return;
    event.preventDefault();
    setAccountMenu(false);
    accountBtn.focus({ preventScroll: true });
  });
}

/* -------------------------------------------------------------- lifecycle -- */

function boot() {
  wireAccount();
  recentEl.addEventListener('toggle', () => {
    if (!recentEl.open || state.doneLoaded) return;
    state.doneLoaded = true;
    loadDone();
  });
  // Cheap freshness: a card marked done in another tab shows up when you come back.
  window.addEventListener('focus', () => refresh());
  loadMe();
  loadOpen();
}

boot();
