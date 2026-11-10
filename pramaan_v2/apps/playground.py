"""The playground mounted at /web by OpenEnv (spec 8.9 and 10, ``Playground.dc.html``).

The page is one Gradio HTML component. Its script calls the OpenEnv routes ``/web/reset`` and
``/web/step`` exactly as any client would, and asks the server function ``view`` to draw what came
back. The environment contract is untouched.
"""
from __future__ import annotations

import json
from typing import Any

import gradio as gr

from ..site.shell import asset_url, json_for_script
from .playground_render import MAX_ACTION_CHARS, SUITES, brief_options, render_page, render_view

PLAYGROUND_TITLE = "Playground · Pramaan"

PLAYGROUND_HEAD = (
    f'<link rel="icon" href="{asset_url("favicon.svg")}" type="image/svg+xml">'
    f'<link rel="preload" href="{asset_url("fonts/anek-latin-latin-wdth-normal.woff2")}" as="font" type="font/woff2" '
    "crossorigin>"
    f'<link rel="stylesheet" href="{asset_url("v2.css")}">'
    '<meta name="theme-color" content="#15110F">'
)

PLAYGROUND_JS = r'''(async () => {
  const root = element.querySelector("[data-pg]");
  if (!root || root.dataset.ready === "true") return;
  root.dataset.ready = "true";
  const BRIEFS = __BRIEFS__;
  const $ = (selector) => root.querySelector(selector);
  const suiteSelect = $("#pg-suite"), briefSelect = $("#pg-brief"), resetButton = $("#pg-reset");
  const status = $("#pg-status"), area = $("#pg-pattern"), runButton = $("#pg-run"), hint = $("#pg-hint");
  const S = { suite: suiteSelect.value, reset: null, steps: [], seed: null, examples: [], busy: false };

  // Header menu below 720 px (the shared script does this on the other pages).
  const header = root.querySelector(".v2-header"), menu = header && header.querySelector(".v2-menu");
  if (menu) menu.addEventListener("click", () => {
    const open = header.classList.toggle("open");
    menu.setAttribute("aria-expanded", open ? "true" : "false");
  });

  const route = (name) => new URL(name, window.location.href.split("#")[0].split("?")[0]).toString();
  const post = async (name, body) => {
    const response = await fetch(route(name), { method: "POST", headers: { "Content-Type": "application/json" },
                                                body: JSON.stringify(body) });
    let data = null;
    try { data = await response.json(); } catch (_) { data = null; }
    if (!response.ok) {
      const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : response.status;
      throw new Error(String(detail));
    }
    return data;
  };

  const fillBriefs = (suite, selected) => {
    briefSelect.replaceChildren();
    for (const [value, label] of BRIEFS[suite].concat([["seed", "Random brief from a seed"]])) {
      const option = document.createElement("option");
      option.value = value; option.textContent = label;
      if (value === selected) option.selected = true;
      briefSelect.append(option);
    }
  };

  // The editor grows with its pattern, as on the board, up to a cap.
  const fit = () => {
    area.style.height = "auto";
    area.style.height = Math.min(Math.max(area.scrollHeight + 2, 560), 1400) + "px";
  };

  const press = (index) => {
    root.querySelectorAll("[data-pg-example]").forEach((chip) => {
      chip.setAttribute("aria-pressed", String(Number(chip.dataset.pgExample) === index));
    });
  };

  const apply = (view, fresh) => {
    $("#pg-reset-panel").innerHTML = view.reset_panel;
    $("#pg-obs-title").textContent = view.title;
    $("#pg-obs").innerHTML = view.observation;
    $("#pg-slots").innerHTML = view.slots;
    $("#pg-saved").innerHTML = view.saved;
    $("#pg-http code").textContent = view.http;
    $("#pg-client code").textContent = view.client;
    status.textContent = view.status;
    runButton.disabled = Boolean(view.run_disabled);
    hint.textContent = view.hint;
    if (fresh) {
      S.examples = view.examples;
      $("#pg-chips").innerHTML = view.chips;
      area.value = S.examples.length ? S.examples[0].pattern : "";
      press(0);
      fit();
    }
  };

  const draw = async (fresh) => {
    const view = await server.view(JSON.stringify({ suite: S.suite, reset: S.reset, steps: S.steps, seed: S.seed }));
    apply(view, fresh);
    return view;
  };

  const reset = async () => {
    if (S.busy) return;
    S.busy = true;
    status.textContent = "Resetting the episode…";
    const body = { suite: suiteSelect.value };
    if (briefSelect.value === "seed") {
      S.seed = Math.floor(Math.random() * 1000000);
      body.seed = S.seed;
    } else {
      S.seed = null;
      body.task_id = briefSelect.value;
    }
    try {
      S.reset = await post("reset", body);
      S.suite = suiteSelect.value;
      S.steps = [];
      const view = await draw(true);
      const query = new URLSearchParams(window.location.search);
      query.set("task_id", view.task_id);
      query.delete("suite");
      window.history.replaceState(null, "", window.location.pathname + "?" + query.toString());
    } catch (error) {
      status.textContent = "Reset did not go through (" + error.message + "). Try again.";
    } finally {
      S.busy = false;
    }
  };

  const step = async () => {
    if (S.busy || runButton.disabled) return;
    if (!S.reset) { await reset(); }
    S.busy = true;
    runButton.disabled = true;
    status.textContent = "Checking the pattern…";
    const pattern = area.value;
    try {
      const result = await post("step", { action: { pattern_json: pattern } });
      S.steps.push({ pattern_json: pattern, result });
      const view = await draw(false);
      if (view.done) resetButton.focus();
    } catch (error) {
      runButton.disabled = false;
      status.textContent = "That step did not go through (" + error.message + "). Reset the episode and try again.";
    } finally {
      S.busy = false;
    }
  };

  suiteSelect.addEventListener("change", () => {
    fillBriefs(suiteSelect.value, BRIEFS[suiteSelect.value][0][0]);
    status.textContent = "New suite selected. Reset the episode to start a brief from it.";
  });
  briefSelect.addEventListener("change", () => {
    status.textContent = "New brief selected. Reset the episode to start it.";
  });
  resetButton.addEventListener("click", reset);
  runButton.addEventListener("click", step);
  $("#pg-chips").addEventListener("click", (event) => {
    const chip = event.target.closest("[data-pg-example]");
    if (!chip) return;
    const index = Number(chip.dataset.pgExample);
    if (!S.examples[index]) return;
    area.value = S.examples[index].pattern;
    press(index);
    fit();
  });
  area.addEventListener("input", () => { press(-1); fit(); });

  // Deep link: /web/?task_id=... (and optionally &suite=...) preselects a brief.
  const params = new URLSearchParams(window.location.search);
  const wanted = params.get("task_id");
  if (wanted) {
    const suite = Object.keys(BRIEFS).find((name) => BRIEFS[name].some(([value]) => value === wanted));
    if (suite) {
      suiteSelect.value = suite;
      fillBriefs(suite, wanted);
    } else {
      status.textContent = "That brief is not in either suite. Showing the default brief.";
    }
  } else if (params.get("suite") && BRIEFS[params.get("suite")]) {
    suiteSelect.value = params.get("suite");
    fillBriefs(suiteSelect.value, BRIEFS[suiteSelect.value][0][0]);
  }
  try { S.examples = (await server.view(JSON.stringify({ suite: suiteSelect.value, steps: [] }))).examples; } catch (_) {}
  await reset();
})();'''


def _script() -> str:
    briefs = {suite: brief_options(suite) for suite, _ in SUITES}
    return PLAYGROUND_JS.replace("__BRIEFS__", json_for_script(briefs, ensure_ascii=False))


MAX_STATE_CHARS = 4 * MAX_ACTION_CHARS + 64_000


def view(state_json: str) -> dict[str, Any]:
    """Draw the page parts for an episode state sent by the page script."""
    if not isinstance(state_json, str) or len(state_json) > MAX_STATE_CHARS:
        return render_view({})
    try:
        state = json.loads(state_json)
    except (TypeError, json.JSONDecodeError):
        state = {}
    return render_view(state if isinstance(state, dict) else {})


def build_playground(
    web_manager: Any,
    action_fields: list[dict[str, Any]],
    metadata: Any,
    is_chat_env: bool,
    title: str,
    quick_start_md: str,
) -> gr.Blocks:
    """The v2 playground. ``web_manager`` drives the episode through the ``/web/reset`` and ``/web/step``
    routes OpenEnv registers beside this mount, so the page needs no handle on it."""
    del web_manager, action_fields, metadata, is_chat_env, quick_start_md
    with gr.Blocks(title=title or PLAYGROUND_TITLE, fill_width=True) as blocks:
        gr.HTML(value=render_page(), head=PLAYGROUND_HEAD, js_on_load=_script(), server_functions=[view],
                apply_default_css=False, container=False, padding=False, elem_classes=["pg-mount"])
    return blocks


__all__ = ["PLAYGROUND_TITLE", "build_playground"]
