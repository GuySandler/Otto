/* ============================================================================
 * Otto — login page (vanilla ES module, no dependencies).
 *
 * One page, two modes:
 *   sign in  POST /api/auth/login  {email, password}
 *   sign up  POST /api/auth/signup {email, password, name?}
 *
 * Both answer {user:{id,email,name}} on success and {detail} otherwise; the
 * session cookie is HttpOnly and owned by the browser, so nothing here touches
 * cookies. A visitor who is already signed in is sent straight to /.
 * ========================================================================== */

const MODES = {
  signin: {
    tagline: 'Sign in to continue',
    submit: 'Sign in',
    switchText: 'New here?',
    switchLabel: 'Create account',
    passwordAutocomplete: 'current-password',
    endpoint: '/api/auth/login',
  },
  signup: {
    tagline: 'Create your account',
    submit: 'Create account',
    switchText: 'Already have an account?',
    switchLabel: 'Sign in',
    passwordAutocomplete: 'new-password',
    endpoint: '/api/auth/signup',
  },
};

const form = document.getElementById('auth-form');
const tagline = document.getElementById('auth-tagline');
const nameField = document.getElementById('auth-name-field');
const nameInput = document.getElementById('auth-name');
const emailInput = document.getElementById('auth-email');
const passwordInput = document.getElementById('auth-password');
const errorEl = document.getElementById('auth-error');
const submitBtn = document.getElementById('auth-submit');
const switchText = document.getElementById('auth-switch-text');
const switchBtn = document.getElementById('auth-switch');

let mode = 'signin';

function applyMode(next) {
  mode = next;
  const spec = MODES[mode];
  tagline.textContent = spec.tagline;
  submitBtn.textContent = spec.submit;
  switchText.textContent = spec.switchText;
  switchBtn.textContent = spec.switchLabel;
  passwordInput.autocomplete = spec.passwordAutocomplete;
  nameField.hidden = mode !== 'signup';
  clearError();
  focusFirstField();
}

function focusFirstField() {
  const first = mode === 'signup' ? nameInput : emailInput;
  first.focus({ preventScroll: true });
}

function clearError() {
  errorEl.hidden = true;
  errorEl.textContent = '';
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.hidden = !message;
}

/** The API answers with {detail}; FastAPI may also send a list of validation errors. */
function detailOf(payload, fallback) {
  if (!payload || typeof payload !== 'object') return fallback;
  const detail = payload.detail;
  if (typeof detail === 'string' && detail.trim()) return detail;
  if (Array.isArray(detail) && detail.length) {
    const parts = detail.map((item) => (item && typeof item.msg === 'string' ? item.msg : '')).filter(Boolean);
    if (parts.length) return parts.join(' · ');
  }
  if (typeof payload.message === 'string' && payload.message.trim()) return payload.message;
  return fallback;
}

async function checkExistingSession() {
  try {
    const response = await fetch('/api/auth/me', { cache: 'no-store', headers: { Accept: 'application/json' } });
    if (response.ok) location.replace('/');
  } catch (err) {
    // Offline or the route is not up yet: stay on the form.
  }
}

async function submit(event) {
  event.preventDefault();
  clearError();
  const spec = MODES[mode];
  const email = emailInput.value.trim();
  const password = passwordInput.value;
  const name = nameInput.value.trim();

  if (!email || !password) {
    showError('Enter your email and password.');
    (email ? passwordInput : emailInput).focus({ preventScroll: true });
    return;
  }

  const body = { email, password };
  if (mode === 'signup' && name) body.name = name;

  submitBtn.disabled = true;
  try {
    const response = await fetch(spec.endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(body),
    });
    if (response.ok) {
      location.assign('/');
      return;
    }
    let payload = null;
    try {
      payload = await response.json();
    } catch (err) {
      payload = null;
    }
    showError(detailOf(payload, response.status === 401 ? 'Those credentials did not work.' : `Sign in failed (HTTP ${response.status}).`));
    passwordInput.select();
  } catch (err) {
    showError('Could not reach the server. Try again.');
  } finally {
    submitBtn.disabled = false;
  }
}

switchBtn.addEventListener('click', () => applyMode(mode === 'signin' ? 'signup' : 'signin'));
form.addEventListener('submit', submit);
for (const input of [emailInput, passwordInput, nameInput]) {
  input.addEventListener('input', clearError);
}

applyMode('signin');
checkExistingSession();
