"""The studio mounted at /studio: Try it and Submit a run (Gradio, custom HTML).

v1's state machine and handlers are reused unchanged: ``pramaan.app._initial_custom_state`` builds the state and
``pramaan.app._handle_custom_event`` handles every action. This module only renders the v2 markup
(``studio_render.render_studio``) and keeps the plumbing v1 uses: one ``gr.HTML`` whose client script collects
``[data-field]`` values and calls ``trigger('click', values)``, one ``gr.State``, and a ``gr.LoginButton``.
"""
from __future__ import annotations

import copy

import gradio as gr

from pramaan import app as v1

from .studio_render import render_studio

STUDIO_HEAD = '<link rel="icon" href="/static/v2/favicon.svg" type="image/svg+xml">'

# Community form values kept between renders. The provider key is deliberately absent: v1 reads it from the
# transient event only, and so does this module.
_COMMUNITY_FIELDS = {"community_model": "model", "community_task_limit": "task_limit",
                     "community_charge_consent": "charge", "community_full_run_cost_consent": "full",
                     "community_archive_consent": "archive_consent"}


def studio_theme() -> gr.themes.Base:
    """Dark surfaces matching the v2 tokens, so nothing flashes light before the page paints."""
    theme = gr.themes.Base(primary_hue=gr.themes.colors.amber, neutral_hue=gr.themes.colors.stone,
                           font=["Anek Latin", "Segoe UI", "system-ui", "sans-serif"],
                           font_mono=["JetBrains Mono", "ui-monospace", "monospace"])
    surfaces = dict(body_background_fill="#15110F", body_text_color="#F4EFE6", body_text_color_subdued="#B8ADA1",
                    background_fill_primary="#15110F", background_fill_secondary="#1D1815",
                    block_background_fill="#1D1815", border_color_primary="#4A3F38",
                    input_background_fill="#1D1815", input_border_color="#4A3F38")
    return theme.set(**surfaces, **{f"{key}_dark": value for key, value in surfaces.items()})


def _event_value(event, name):
    try:
        return getattr(event, name)
    except AttributeError:
        return None


def on_event(event: gr.EventData, state: dict, oauth_profile: gr.OAuthProfile | None,
             oauth_token: gr.OAuthToken | None, request: gr.Request):
    """Run v1's handler unchanged and render the v2 views from the state it returns."""
    _v1_html, new_state = v1._handle_custom_event(event, state, oauth_profile, oauth_token, request)
    new_state = dict(new_state)
    action = _event_value(event, "action")
    if isinstance(action, str) and action:
        new_state["v2_last_action"] = action
    username = getattr(oauth_profile, "username", None) if oauth_profile is not None else None
    if isinstance(username, str) and username:
        new_state["v2_hf_user"] = username
    else:
        new_state.pop("v2_hf_user", None)
    community = dict(new_state.get("v2_community") or {})
    for field, key in _COMMUNITY_FIELDS.items():
        value = _event_value(event, field)
        if value is not None:
            community[key] = value if not isinstance(value, str) else value[:256]
    new_state["v2_community"] = community
    return render_studio(new_state), new_state


STUDIO_JS = r"""
const studio = () => element.querySelector('[data-studio]');
const ui = {tab: 'checks', preview: 'repeat'};
const pending = {
  generate: ['make', 'Asking Beam for a pattern. This can take a minute; the checks appear here when it is done.'],
  example: ['make', 'Loading the example and its checks.'],
  state: ['make', 'Loading the traditions and worked examples for this state.'],
  tradition: ['make', 'Loading the worked examples for this tradition.'],
  recheck: ['recheck', 'Rechecking your edit.'],
};
function view() { return (window.location.hash || '').split('?')[0] === '#/submit' ? 'submit' : 'try'; }
function legacyRoute() {
  const raw = window.location.hash || '';
  const [path, query = ''] = raw.slice(1).split('?');
  const params = new URLSearchParams(query);
  if (path === '/overview' || path === '/states') { window.location.replace('/'); return true; }
  if (path === '/build' || path === '/rl') { window.location.replace('/build'); return true; }
  if (path === '/leaderboard') {
    const task = params.get('task') || '';
    if (/^[a-z0-9][a-z0-9._-]{0,150}$/.test(task)) window.location.replace('/replays/' + encodeURIComponent(task));
    else window.location.replace(params.has('state') ? '/replays' : '/leaderboard');
    return true;
  }
  return false;
}
function applyView() {
  const root = studio();
  if (!root) return;
  const current = view();
  root.dataset.view = current;
  root.querySelectorAll('[data-studio-view]').forEach(node => { node.hidden = node.dataset.studioView !== current; });
  document.title = current === 'submit' ? 'Submit a run · Pramaan' : 'Try it · Pramaan';
}
function values(action) {
  const out = {};
  element.querySelectorAll('[data-field]').forEach(node => {
    if (node.type === 'file') return;
    if (node.dataset.field === 'community_api_key' && !['community_run', 'community_publish'].includes(action)) return;
    out[node.dataset.field] = node.type === 'checkbox' ? node.checked : node.value;
  });
  out.action = action;
  out.active_tab = view() === 'submit' ? 'leaderboard' : 'studio';
  return out;
}
function say(selector, text) {
  const node = element.querySelector(selector);
  if (node) node.textContent = text;
}
function clearKey() {
  // The provider key leaves the page only with the action that needs it; never keep it in the field.
  const key = element.querySelector('[data-field="community_api_key"]');
  if (key) key.value = '';
}
function send(action) {
  if (pending[action]) say(`[data-status="${pending[action][0]}"]`, pending[action][1]);
  trigger('click', values(action));
}
function loginText() { return document.querySelector('#pramaan-v2-login')?.textContent || ''; }
function signedIn() { return /\bsign out\b/i.test(loginText()); }
function syncSignIn() {
  const root = studio();
  if (!root || !signedIn() || root.dataset.signedIn === 'true' || window.__pramaanV2SignInSync) return;
  window.__pramaanV2SignInSync = true;
  trigger('click', values('refresh_sign_in'));
}
function setTab(key, focus) {
  ui.tab = key;
  element.querySelectorAll('[role="tab"][data-tab]').forEach(tab => {
    const on = tab.dataset.tab === key;
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on ? 0 : -1;
    tab.style.borderBottomColor = on ? '#F4B63F' : 'transparent';
    tab.style.color = on ? '#F4EFE6' : '#B8ADA1';
    if (on && focus) tab.focus();
  });
  element.querySelectorAll('[data-tabpanel]').forEach(panel => { panel.hidden = panel.dataset.tabpanel !== key; });
}
function setPreview(key) {
  ui.preview = key;
  element.querySelectorAll('[data-preview-choice]').forEach(button => {
    const on = button.dataset.previewChoice === key;
    button.setAttribute('aria-pressed', String(on));
    button.style.background = on ? '#F4EFE6' : 'transparent';
    button.style.color = on ? '#15110F' : '#F4EFE6';
    button.style.borderColor = on ? '#F4EFE6' : '#4A3F38';
  });
  element.querySelectorAll('.st-preview').forEach(node => { node.dataset.preview = key; });
}
function syncControls() {
  element.querySelectorAll('[data-field]').forEach(node => {
    if (node.type === 'file' || node.dataset.field === 'community_api_key') return;
    if (node.type === 'checkbox') node.checked = node.defaultChecked;
    else if (node.tagName === 'SELECT') {
      const chosen = Array.from(node.options).find(option => option.defaultSelected);
      if (chosen) node.value = chosen.value;
    } else node.value = node.defaultValue;
  });
}
async function submitArchive() {
  const status = element.querySelector('[data-community-archive-status]');
  const set = text => { if (status) status.textContent = text; };
  if (!signedIn()) { set('Sign in with Hugging Face to publish: use the sign-in button above.'); return; }
  if (!element.querySelector('[data-field="community_archive_consent"]')?.checked) {
    set('Check the publication consent box before submitting. No archive was uploaded or published.'); return;
  }
  const file = element.querySelector('[data-field="community_archive_file"]')?.files?.[0];
  if (!file || !file.name.toLowerCase().endsWith('.zip')) {
    set('Choose a ZIP archive before submitting. No archive was uploaded or published.'); return;
  }
  if (file.size > 20000000) { set('The ZIP archive is larger than 20 MB. Nothing was uploaded.'); return; }
  set('Uploading the ZIP to the Space for validation.');
  try {
    const uploaded = await upload(file);
    if (!uploaded?.path) throw new Error('upload failed');
    const out = values('community_archive_submit');
    out.community_archive_path = uploaded.path;
    trigger('click', out);
  } catch (error) {
    const reason = typeof error?.message === 'string' ? error.message.trim().slice(0, 180) : '';
    set(reason ? `Archive upload failed: ${reason}` : 'Archive upload failed. Check the file and try again.');
  }
}
element.addEventListener('click', event => {
  const menu = event.target.closest('.v2-menu');
  if (menu) {
    const head = menu.closest('.v2-header');
    const open = head.classList.toggle('open');
    menu.setAttribute('aria-expanded', open ? 'true' : 'false');
    return;
  }
  const route = event.target.closest('a[href="/submit"], a[href="/try"]');
  if (route) {
    event.preventDefault();
    const target = route.getAttribute('href') === '/submit' ? '#/submit' : '#/';
    history.pushState(null, '', window.location.pathname + window.location.search + target);
    applyView();
    window.scrollTo(0, 0);
    return;
  }
  const tab = event.target.closest('[role="tab"][data-tab]');
  if (tab) { setTab(tab.dataset.tab, false); return; }
  const preview = event.target.closest('[data-preview-choice]');
  if (preview) { setPreview(preview.dataset.previewChoice); return; }
  const copyButton = event.target.closest('[data-copy-certificate]');
  if (copyButton) {
    const bytes = Uint8Array.from(atob(copyButton.dataset.certificateB64 || ''), ch => ch.charCodeAt(0));
    navigator.clipboard.writeText(new TextDecoder().decode(bytes))
      .then(() => say('[data-status="copy"]', 'Certificate JSON copied.'))
      .catch(() => say('[data-status="copy"]', 'Copy is unavailable in this browser.'));
    return;
  }
  const button = event.target.closest('[data-action]');
  if (!button || button.disabled) return;
  event.preventDefault();
  const action = button.dataset.action;
  if (button.hasAttribute('data-bundle-request')) {
    window.__pramaanV2Bundle = true;
    say('[data-status="copy"]', 'Preparing the run bundle.');
  }
  if (action === 'community_archive_submit') { submitArchive(); return; }
  if (action === 'community_run') {
    const key = element.querySelector('[data-field="community_api_key"]');
    if (!element.querySelector('[data-field="community_charge_consent"]')?.checked || !key?.value.trim()) {
      say('[data-community-status]', 'Enter your provider key and confirm possible provider charges before starting. No request was sent.');
      return;
    }
    say('[data-community-status]', 'Run in progress. Briefs run one after another and the summary appears here when the run ends. Keep this tab open.');
    trigger('click', values(action));
    clearKey();
    return;
  }
  if (action === 'community_publish') {
    send(action);
    clearKey();
    return;
  }
  if (action === 'community_estimate') say('[data-community-status]', 'Estimating from the official token usage. No provider call is made.');
  send(action);
});
element.addEventListener('keydown', event => {
  const tab = event.target.closest('[role="tab"][data-tab]');
  if (!tab || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
  const tabs = Array.from(element.querySelectorAll('[role="tab"][data-tab]'));
  const i = tabs.indexOf(tab);
  const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 :
    (i + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
  event.preventDefault();
  setTab(tabs[next].dataset.tab, true);
});
element.addEventListener('change', event => {
  const field = event.target.dataset?.field;
  if (field === 'state' || field === 'tradition') send(field);
});
watch('value', () => {
  syncControls();
  applyView();
  setTab(ui.tab, false);
  setPreview(ui.preview);
  if (window.__pramaanV2Bundle) {
    window.__pramaanV2Bundle = false;
    element.querySelector('a[data-bundle-download]')?.click();
  }
});
window.addEventListener('hashchange', () => { if (!legacyRoute()) applyView(); });
window.addEventListener('popstate', applyView);
if (!legacyRoute()) applyView();
// Gradio sets the sign-in button's label from a load event and may replace its node, so poll briefly.
let signInChecks = 0;
const signInTimer = window.setInterval(() => {
  syncSignIn();
  if (++signInChecks >= 30 || window.__pramaanV2SignInSync) window.clearInterval(signInTimer);
}, 500);
"""


def build_studio() -> gr.Blocks:
    initial = v1._initial_custom_state()
    with gr.Blocks(title="Try it · Pramaan", fill_width=True, delete_cache=(3600, 3600)) as demo:
        with gr.Row(elem_id="pramaan-v2-login-row"):
            gr.LoginButton(value="Sign in to publish", logout_value="Sign out ({})", variant="huggingface",
                           size="sm", elem_id="pramaan-v2-login")
        page = gr.HTML(render_studio(copy.deepcopy(initial)), js_on_load=STUDIO_JS, elem_id="pramaan-studio",
                       show_label=False, container=False, padding=False, apply_default_css=False)
        state = gr.State(initial)
        # "undocumented" keeps the endpoint out of the API page and out of Gradio's run history, which would
        # otherwise record the inputs of public endpoints in the browser.
        page.click(on_event, inputs=state, outputs=[page, state], show_progress="hidden",
                   concurrency_limit=1, concurrency_id="beam", trigger_mode="multiple",
                   api_visibility="undocumented")
    demo.queue(default_concurrency_limit=1, max_size=32)
    return demo
