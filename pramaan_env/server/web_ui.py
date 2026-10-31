"""Custom HTML OpenEnv playground for the Pramaan proof environment."""

from __future__ import annotations

import html
import base64
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import gradio as gr
from PIL import Image

from pramaan.dsl import parse_pattern
from pramaan.render import render
from pramaan.paths import resolve_project_root


ROOT = resolve_project_root(__file__)
PASSING_TASK_ID = "as_gamosa-red-on-white-border-and-motif-band-easy-2"
PASSING_EXAMPLE = ROOT / "examples" / "beam_g4_as_gamosa_grid.pattern.json"
FAILING_EXAMPLE = "{ this is intentionally not valid PatternDSL JSON"

_PYTHON_SNIPPET = '''import requests

base = "https://kshitijthakkar-pramaan-proof-of-art.hf.space"
requests.post(f"{base}/web/reset", json={"task_id": "as_gamosa-red-on-white-border-and-motif-band-easy-2"}).raise_for_status()
result = requests.post(f"{base}/web/step", json={"action": {"pattern_json": open("design.pattern.json").read()}}).json()
print("reward:", result["reward"], "done:", result["done"])
'''
WEB_TITLE = "Pramaan: Proof of Art · RL environment"

_MARKUP = '''
<style>
:root{color-scheme:light;--pramaan-indigo:#172f4a;--pramaan-ink:#26384b;--pramaan-madder:#a84436;--pramaan-teal:#176a66;--pramaan-paper:#f7f3ec;--pramaan-card:#fffdf9;--pramaan-card-raised:#fff;--pramaan-card-subtle:#f3eee6;--pramaan-line:#c9bba8;--pramaan-muted:#596775;--pramaan-on-accent:#fff;--pramaan-pass:#126b50;--pramaan-pass-bg:#e5f1ec;--pramaan-fail:#9b2948;--pramaan-fail-bg:#f7e9eb;--pramaan-pending:#67430f;--pramaan-pending-bg:#f4eedf;--pramaan-checker:#ece7de;--pramaan-shadow:#172f4a}
html.dark,body.dark,.dark{color-scheme:dark;--pramaan-indigo:#e8eef5;--pramaan-ink:#e1e8ef;--pramaan-madder:#f09583;--pramaan-teal:#75d0c5;--pramaan-paper:#151b22;--pramaan-card:#202a34;--pramaan-card-raised:#293542;--pramaan-card-subtle:#26313b;--pramaan-line:#647383;--pramaan-muted:#c0cad4;--pramaan-on-accent:#10272a;--pramaan-pass:#83dfb6;--pramaan-pass-bg:#173c32;--pramaan-fail:#ff9bad;--pramaan-fail-bg:#492531;--pramaan-pending:#f2d18d;--pramaan-pending-bg:#443820;--pramaan-checker:#36424e;--pramaan-shadow:#000}
html.light,body.light{color-scheme:light}
@media(prefers-color-scheme:dark){html:not(.light):not(.dark),body:not(.light):not(.dark){color-scheme:dark;--pramaan-indigo:#e8eef5;--pramaan-ink:#e1e8ef;--pramaan-madder:#f09583;--pramaan-teal:#75d0c5;--pramaan-paper:#151b22;--pramaan-card:#202a34;--pramaan-card-raised:#293542;--pramaan-card-subtle:#26313b;--pramaan-line:#647383;--pramaan-muted:#c0cad4;--pramaan-on-accent:#10272a;--pramaan-pass:#83dfb6;--pramaan-pass-bg:#173c32;--pramaan-fail:#ff9bad;--pramaan-fail-bg:#492531;--pramaan-pending:#f2d18d;--pramaan-pending-bg:#443820;--pramaan-checker:#36424e;--pramaan-shadow:#000}}
html,body{width:100%;min-width:0}body,.gradio-container{width:100%;min-width:0;max-width:100vw!important;margin:0!important;background:var(--pramaan-paper);color:var(--pramaan-ink)}
.poa-shell{max-width:1160px;margin:0 auto;padding:clamp(12px,3vw,30px);font-family:Inter,system-ui,sans-serif;color:var(--pramaan-ink)}
.poa-head{display:flex;justify-content:space-between;align-items:center;gap:18px;border-bottom:2px solid var(--pramaan-line);padding:8px 0 18px}
.poa-brand{display:flex;align-items:center;gap:12px}.poa-logo{width:48px;height:48px;object-fit:contain;border-radius:10px;background:var(--pramaan-card);border:1px solid var(--pramaan-line)}
.poa-kicker{color:var(--pramaan-teal);font:700 12px/1.3 ui-monospace,monospace;letter-spacing:.12em;text-transform:uppercase}
.poa-title{color:var(--pramaan-indigo);font:700 clamp(32px,5vw,54px)/1.02 Georgia,serif;margin:10px 0}
.poa-subtitle{color:var(--pramaan-muted);font-size:16px;line-height:1.55;margin:0;max-width:730px}
.poa-head a,.poa-links a{color:var(--pramaan-ink)!important;background:var(--pramaan-card);font-weight:750;text-decoration:none;border:1px solid var(--pramaan-line);border-radius:10px;padding:10px 13px;white-space:nowrap}
.poa-links{display:flex;flex-wrap:wrap;gap:9px;margin-top:12px}.poa-links a:hover,.poa-head a:hover{border-color:var(--pramaan-teal);text-decoration:underline}
.poa-legend{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin:16px 0}
.poa-legend>div,.poa-panel{border:1px solid var(--pramaan-line);border-radius:14px;background:var(--pramaan-card);padding:14px}
.poa-legend b{display:block;margin-bottom:5px}.poa-red{color:var(--pramaan-fail)}.poa-gold{color:var(--pramaan-pending)}.poa-green{color:var(--pramaan-pass)}
.poa-task{border-left:4px solid var(--pramaan-teal);margin:12px 0 18px;line-height:1.55}
.poa-work{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(300px,.9fr);gap:14px;align-items:start}
.poa-label{display:block;font-weight:750;margin:0 0 8px}.poa-pattern{box-sizing:border-box;width:100%;min-height:330px;resize:vertical;border:1px solid var(--pramaan-line);border-radius:12px;background:var(--pramaan-card-raised);color:var(--pramaan-ink);padding:14px;font:13px/1.5 ui-monospace,SFMono-Regular,Consolas,monospace}
.poa-actions{display:flex;flex-wrap:wrap;gap:9px;margin:12px 0}.poa-btn{min-height:44px;padding:10px 15px;border:2px solid var(--pramaan-indigo);border-radius:9px;background:var(--pramaan-indigo);color:var(--pramaan-on-accent);font-weight:750;cursor:pointer}.poa-btn:hover{transform:translateY(-1px);box-shadow:3px 3px 0 var(--pramaan-teal)}.poa-btn.secondary{background:var(--pramaan-card);color:var(--pramaan-indigo)}.poa-btn.gold{background:var(--pramaan-teal);color:var(--pramaan-on-accent);border-color:var(--pramaan-teal)}
.poa-result{min-height:250px}.poa-result pre{max-height:520px;overflow:auto;background:var(--pramaan-card-subtle);border:1px solid var(--pramaan-line);border-radius:10px;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.45 ui-monospace,monospace}
.poa-status{min-height:32px;line-height:1.5}.poa-preview{display:block;max-width:min(100%,360px);aspect-ratio:1;border:1px solid var(--pramaan-line);border-radius:12px;background:var(--pramaan-checker);image-rendering:pixelated;margin:8px auto}.poa-checks{padding-left:20px}.poa-checks li{margin:6px 0}.poa-checks .pass{color:var(--pramaan-pass)}.poa-checks .fail{color:var(--pramaan-fail)}
.poa-details{margin-top:12px;color:var(--pramaan-muted)}.poa-details summary{cursor:pointer;font-weight:700;color:var(--pramaan-indigo)}
.poa-code{position:relative;background:var(--pramaan-indigo);color:var(--pramaan-on-accent)!important;border-radius:12px;padding:16px;overflow:auto}.poa-code pre,.poa-code code{margin:0;padding:34px 0 0;white-space:pre;line-height:1.5;background:transparent!important;color:var(--pramaan-on-accent)!important}.poa-copy{position:absolute;top:10px;right:10px;min-height:40px;border:1px solid var(--pramaan-card-subtle);border-radius:8px;background:var(--pramaan-card);color:var(--pramaan-ink)!important;padding:7px 12px;font-weight:750;cursor:pointer}
@media(max-width:720px){.poa-shell{padding:12px}.poa-head{align-items:flex-start;flex-direction:column}.poa-legend{grid-template-columns:1fr}.poa-work{grid-template-columns:1fr}.poa-pattern{min-height:260px}.poa-btn{flex:1 1 160px}}
</style>
<main class="poa-shell" id="pramaan-rl">
  <header class="poa-head"><div><div class="poa-brand"><img class="poa-logo" src="/favicon.ico" alt="Pramaan original mark"><div><div class="poa-kicker">Proof-carrying pattern environment</div><h1 class="poa-title">Art you can prove.</h1></div></div>
  <p class="poa-subtitle">Give the environment a pattern. It returns a reward from 0 to 1 and the exact check that failed.</p>
  <nav class="poa-links" aria-label="Pramaan links"><a href="/studio/">Open the Studio</a><a href="https://huggingface.co/collections/kshitijthakkar/pramaan-proof-of-art" target="_blank" rel="noopener">Open the Collection</a></nav></div>
  <a href="/studio/" aria-label="Open Pramaan Studio">Studio home ↑</a></header>
  <section class="poa-legend" aria-label="Reward guide">
    <div><b class="poa-red">0 · Rejected</b>Parse failure, truncated input, or an invalid verifier report.</div>
    <div><b class="poa-gold">0 &lt; reward &lt; 1 · Partial</b>A valid design passes some hard checks; the observation lists failures.</div>
    <div><b class="poa-green">1.0 · Proven</b>Every hard check passes; the environment returns a certificate.</div>
  </section>
  <section class="poa-panel poa-task"><div class="poa-kicker">Current challenge</div><div id="poa-brief" aria-live="polite">Loading a replay-verified example…</div></section>
  <section class="poa-work">
    <div class="poa-panel"><label class="poa-label" for="poa-pattern">PatternDSL JSON</label>
      <textarea id="poa-pattern" class="poa-pattern" spellcheck="false" aria-label="PatternDSL JSON"></textarea>
      <div class="poa-actions"><button id="poa-pass" class="poa-btn" type="button">Load example</button>
        <button id="poa-fail" class="poa-btn secondary" type="button">Load a failing example</button>
        <button id="poa-reset" class="poa-btn secondary" type="button">Reset</button>
        <button id="poa-run" class="poa-btn gold" type="button">Run one verifier step</button></div>
    </div>
    <div class="poa-panel poa-result"><div class="poa-kicker">Verifier result</div>
      <div id="poa-status" class="poa-status" role="status" aria-live="polite">Connecting to the OpenEnv checker…</div>
      <img id="poa-preview" class="poa-preview" alt="Rendered pattern preview" hidden>
      <h3>Checks</h3><ul id="poa-checks" class="poa-checks"><li>Waiting for a verifier step.</li></ul>
      <details class="poa-details"><summary>Raw observation JSON</summary><pre id="poa-observation">Waiting for the first observation.</pre></details></div>
  </section>
  <section class="poa-panel" style="margin-top:14px"><label class="poa-label">Call Pramaan from Python</label>
    <div class="poa-code"><button id="poa-copy" class="poa-copy" type="button">Copy</button><pre id="poa-python-example">__PYTHON_SNIPPET__</pre></div></section>
</main>
'''

_JS = r'''(async () => {
  const root = element.querySelector("#pramaan-rl");
  if (!root || root.dataset.ready === "true") return;
  root.dataset.ready = "true";
  const passing = __PASSING_PATTERN__;
  const failing = __FAILING_PATTERN__;
  const taskId = __PASSING_TASK_ID__;
  const input = root.querySelector("#poa-pattern");
  const brief = root.querySelector("#poa-brief");
  const status = root.querySelector("#poa-status");
  const output = root.querySelector("#poa-observation");
  const preview = root.querySelector("#poa-preview");
  const checks = root.querySelector("#poa-checks");
  const show = (result) => {
    const obs = result?.observation || {};
    brief.textContent = [obs.state, obs.tradition_id, obs.brief].filter(Boolean).join(" · ");
    output.textContent = JSON.stringify(result, null, 2);
    checks.replaceChildren();
    for (const check of (obs.checks || [])) {
      const item = document.createElement("li");
      const passed = check.status === "PASS";
      item.className = passed ? "pass" : "fail";
      item.textContent = `${passed ? "✓" : "✕"} ${check.id || "check"}: ${check.message || check.status || "result unavailable"}`;
      checks.append(item);
    }
    if (!checks.childElementCount) {
      const item = document.createElement("li");
      item.textContent = "Checks appear after you run a step.";
      checks.append(item);
    }
    const reward = result?.reward;
    if (reward === 1) status.textContent = `Verified · reward ${reward.toFixed(1)} · certificate issued · attempt ${obs.attempt}/3`;
    else if (reward === 0) status.textContent = `Not verified · reward 0 · inspect the parse error or counterexample · attempt ${obs.attempt}/3`;
    else if (typeof reward === "number") status.textContent = `Partial reward ${reward.toFixed(3)} · revise the failed checks · attempt ${obs.attempt}/3`;
    else status.textContent = `Episode ready · ${obs.task_id || "task loaded"} · submit a design for a verifier result`;
  };
  const reset = async () => {
    status.textContent = "Resetting the episode…";
    const result = await server.reset_episode(taskId);
    show(result);
    return result;
  };
  root.querySelector("#poa-pass").addEventListener("click", async () => {
    input.value = passing;
    status.textContent = "Passing example loaded. Reset, then run one verifier step.";
  });
  root.querySelector("#poa-fail").addEventListener("click", async () => {
    input.value = failing;
    try { await reset(); status.textContent = "Failing example loaded. Run one step to see the parse failure and reward 0."; }
    catch (error) { status.textContent = `Reset failed: ${error}`; }
  });
  root.querySelector("#poa-reset").addEventListener("click", async () => {
    input.value = passing;
    try { await reset(); } catch (error) { status.textContent = `Reset failed: ${error}`; }
  });
  root.querySelector("#poa-run").addEventListener("click", async () => {
    status.textContent = "Checking the design…";
    try {
      const result = await server.run_step(input.value);
      show(result);
      const image = await server.render_pattern(input.value);
      if (image) { preview.src = image; preview.hidden = false; }
      else { preview.removeAttribute("src"); preview.hidden = true; }
    } catch (error) { status.textContent = `Step failed: ${error}`; }
  });
  root.querySelector("#poa-copy").addEventListener("click", async (event) => {
    try { await navigator.clipboard.writeText(root.querySelector("#poa-python-example").innerText); event.currentTarget.textContent = "Copied"; }
    catch (_) { event.currentTarget.textContent = "Select and copy"; }
  });
  input.value = passing;
  try { await reset(); } catch (error) { status.textContent = `OpenEnv connection failed: ${error}`; }
})();'''


def _markup() -> str:
    return _MARKUP.replace("__PYTHON_SNIPPET__", html.escape(_PYTHON_SNIPPET))


def _javascript(passing_pattern: str) -> str:
    return (_JS.replace("__PASSING_PATTERN__", json.dumps(passing_pattern))
            .replace("__FAILING_PATTERN__", json.dumps(FAILING_EXAMPLE))
            .replace("__PASSING_TASK_ID__", json.dumps(PASSING_TASK_ID)))


def build_pramaan_web(
    web_manager: Any,
    action_fields: list[dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """Build a custom HTML playground backed by OpenEnv's session manager."""
    del action_fields, metadata, is_chat_env, quick_start_md
    passing_pattern = PASSING_EXAMPLE.read_text(encoding="utf-8")

    async def reset_episode(task_id: str = PASSING_TASK_ID) -> dict[str, Any]:
        return await web_manager.reset_environment({"task_id": task_id})

    async def run_step(pattern_json: str) -> dict[str, Any]:
        return await web_manager.step_environment({"pattern_json": pattern_json or ""})

    def render_pattern(pattern_json: str) -> str:
        """Render submitted DSL as a small PNG data URL; never execute model output."""
        if not isinstance(pattern_json, str) or len(pattern_json.encode("utf-8")) > 20_000:
            return ""
        try:
            pattern = parse_pattern(pattern_json)
            _masks, grid = render(pattern)
            palette = [tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
                       for color in pattern.palette]
            pixels = [palette[int(index)] for index in grid.flat]
            image = Image.new("RGB", (64, 64))
            image.putdata(pixels)
            image = image.resize((384, 384), Image.Resampling.NEAREST)
            buffer = BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        except (ValueError, TypeError, IndexError):
            return ""

    with gr.Blocks(title=title) as blocks:
        gr.HTML(
            value=_markup(),
            js_on_load=_javascript(passing_pattern),
            server_functions=[reset_episode, run_step, render_pattern],
        )
    return blocks
