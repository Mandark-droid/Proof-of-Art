import ast
import re
import threading
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

from scripts.ui_contrast_audit import audit_visible_text


ROOT = Path(__file__).resolve().parents[1]


def _run_with_edge_or_skip(audit):
    """Bound driver startup and Edge launch; a missing local browser is not a UI failure."""
    launched = threading.Event()
    finished = threading.Event()
    result = {}

    def run():
        manager = playwright.sync_playwright()
        p = browser = None
        try:
            p = manager.start()
            browser = p.chromium.launch(channel="msedge", headless=True, timeout=30_000)
        except Exception as exc:
            try:
                p.stop() if p is not None else manager.stop()
            except Exception:
                pass
            result["launch_error"] = exc
            launched.set()
            return
        launched.set()
        try:
            audit(p, browser)
        except BaseException as exc:
            result["audit_error"] = exc
        finally:
            try:
                browser.close()
            finally:
                try:
                    p.stop()
                finally:
                    finished.set()

    worker = threading.Thread(target=run, name="pramaan-rendered-contrast", daemon=True)
    worker.start()
    if not launched.wait(timeout=30):
        pytest.skip("browser launch unavailable in this " + "environment (30-second timeout)")
    if "launch_error" in result:
        pytest.skip(
            "browser launch unavailable in this "
            f"environment ({type(result['launch_error']).__name__})"
        )
    if not finished.wait(timeout=90):
        pytest.fail("rendered contrast audit exceeded 90 seconds after browser launch")
    if "audit_error" in result:
        raise result["audit_error"]


def _load_pramaan_css() -> str:
    tree = ast.parse((ROOT / "pramaan" / "app.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "PRAMAAN_CSS" for target in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("PRAMAAN_CSS assignment was not found")


@pytest.mark.browser
def test_rendered_pramaan_theme_markup_meets_wcag_aa_in_light_and_dark():
    css = _load_pramaan_css()
    replay = (ROOT / "assets" / "proof-replay.html").read_text(encoding="utf-8")
    controls = (
        '<div class="prose"><nav class="pramaan-ui-tabs">'
        '<a class="pramaan-ui-tab" href="#one">Inactive tab</a>'
        '<a class="pramaan-ui-tab" aria-selected="true" href="#two">Active tab</a>'
        '</nav><nav class="tab-nav"><button class="selected"><span>Native selected tab</span>'
        '</button></nav><nav class="pramaan-ui-subtabs"><button aria-pressed="true">'
        '<span>Pattern source</span></button></nav><a class="pramaan-ui-button" '
        'href="#download">Download run bundle</a></div>'
        '<section class="pramaan-ui-card"><h2>Verifier evidence</h2>'
        '<p class="pramaan-muted">The checker reports the failed rule.</p>'
        '<label for="pattern">Pattern source</label><input id="pattern" value="example">'
        '<button class="pramaan-ui-button secondary">Audit and contribute archive</button>'
        '<button class="pramaan-ui-button" disabled>Run and publish</button></section>'
        '<p id="contrast-copy">Inherited dark prose copy</p>'
    )
    markup = (
        "<!doctype html><html><head><meta charset='utf-8'><style>" + css
        + "</style><style>.prose :where(p:not([class]),strong:not([class]),div:not([class]),"
        + "summary:not([class]),small:not([class])) { color: #f5f5f4 !important; } "
        + ".prose a { color: #287d78 !important; } "
        + ".prose a.pramaan-ui-button { background: #172f4a !important; } "
        + ".tab-nav button.selected, .pramaan-ui-subtabs button[aria-pressed='true'] "
        + "{ color: #172f4a !important; }</style>"
        + "</head><body><main class='gradio-container'><div class='pramaan-ui'>"
        + controls + replay + "</div></main></body></html>"
    )

    def audit_with_browser(p, browser):
        page = browser.new_page(viewport={"width": 1440, "height": 900}, color_scheme="light")
        page.set_content(markup, wait_until="domcontentloaded", timeout=30000)
        for theme in ("light", "dark", "system-dark"):
            page.emulate_media(color_scheme="light" if theme == "light" else "dark")
            page.evaluate("""theme => {
              document.documentElement.classList.toggle('light', theme === 'light');
              document.body.classList.toggle('dark', theme === 'dark');
            }""", theme)
            audit = audit_visible_text(page)
            assert audit["text_items"] > 10
            assert not audit["horizontal_overflow"]
            tab_colors = page.locator(".pramaan-ui-tab").evaluate_all(
                "nodes => nodes.map(node => ({color: getComputedStyle(node).color, "
                "background: getComputedStyle(node).backgroundColor}))"
            )
            assert tab_colors[0]["color"] != tab_colors[0]["background"], tab_colors
            assert tab_colors[1]["color"] != tab_colors[1]["background"], tab_colors
            native_controls = page.locator(
                ".tab-nav button.selected, .pramaan-ui-subtabs button[aria-pressed='true'], "
                "a.pramaan-ui-button"
            ).evaluate_all(
                "nodes => nodes.map(node => ({text: node.textContent.trim(), "
                "color: getComputedStyle(node).color, background: getComputedStyle(node).backgroundColor}))"
            )
            for control in native_controls:
                assert _contrast_rgb(control["color"], control["background"]) >= 4.5, (
                    theme, control
                )
            copy_color = page.locator("#contrast-copy").evaluate("el => getComputedStyle(el).color")
            expected_copy = "rgb(38, 56, 75)" if theme == "light" else "rgb(225, 232, 239)"
            assert copy_color == expected_copy, (theme, copy_color, expected_copy)
            assert audit["failures"] == [], f"{theme} theme WCAG AA failures: " + "; ".join(
                f"{row['ratio']}:1 (needs {row['required']}:1) · {row['text']} · {row['className']}"
                for row in audit["failures"]
            )
            expected_surface = "rgb(247, 243, 236)" if theme == "light" else "rgb(21, 27, 34)"
            surface = page.locator(".gradio-container").evaluate("el => getComputedStyle(el).backgroundColor")
            assert surface == expected_surface, f"{theme} theme did not activate its page surface: {surface}"
        page.close()

    _run_with_edge_or_skip(audit_with_browser)


@pytest.mark.browser
def test_home_branding_fits_phone_width_in_light_and_dark():
    from pramaan.app import _initial_custom_state, _render_custom_app

    css = _load_pramaan_css()
    rendered = _render_custom_app(_initial_custom_state())
    markup = (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width,initial-scale=1'><style>" + css
        + "</style></head><body><main class='gradio-container'><div class='pramaan-ui'>"
        + rendered + "</div></main></body></html>"
    )

    def audit_with_browser(p, browser):
        page = browser.new_page(viewport={"width": 390, "height": 844}, color_scheme="light")
        page.set_content(markup, wait_until="domcontentloaded", timeout=30000)
        for theme in ("light", "dark"):
            page.emulate_media(color_scheme=theme)
            page.evaluate(
                "theme => { document.documentElement.className = theme; "
                "document.body.className = theme; }",
                theme,
            )
            intro = page.locator(".pramaan-home-intro")
            headline = intro.locator("h1")
            subtitle = intro.locator("p")
            assert headline.count() == 1
            assert headline.inner_text() == "AI drew it. Math checked it."
            assert subtitle.inner_text().startswith("Proof of art that you can verify.")
            assert headline.is_visible() and subtitle.is_visible()
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert headline.evaluate("el => el.getBoundingClientRect().right <= innerWidth")
            assert subtitle.evaluate("el => el.getBoundingClientRect().right <= innerWidth")
        page.close()

    _run_with_edge_or_skip(audit_with_browser)


@pytest.mark.browser
def test_primary_tabs_fit_phone_and_pass_computed_contrast_in_both_themes():
    from pramaan.app import _initial_custom_state, _render_custom_app

    css = _load_pramaan_css()
    rendered = _render_custom_app(_initial_custom_state())
    markup = (
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
        "content='width=device-width,initial-scale=1'><style>" + css
        + "</style></head><body><main class='gradio-container'><div class='pramaan-ui'>"
        + rendered + "</div></main></body></html>"
    )
    tabs = {
        "benchmark": "Home",
        "leaderboard": "Models",
        "studio": "Try it",
        "rl": "Build with it",
    }

    def audit_with_browser(p, browser):
        page = browser.new_page(viewport={"width": 390, "height": 844}, color_scheme="light")
        page.set_content(markup, wait_until="domcontentloaded", timeout=30000)
        for theme in ("light", "dark"):
            page.emulate_media(color_scheme=theme)
            page.evaluate(
                "theme => { document.documentElement.className = theme; "
                "document.body.className = theme; }",
                theme,
            )
            for key, label in tabs.items():
                page.evaluate("""key => {
                  document.querySelectorAll('[data-tab-panel]').forEach(panel => {
                    panel.hidden = panel.dataset.tabPanel !== key;
                  });
                  document.querySelectorAll('[data-tab-button]').forEach(button => {
                    button.setAttribute('aria-selected', String(button.dataset.tabButton === key));
                  });
                }""", key)
                panel = page.locator(f'[data-tab-panel="{key}"]')
                assert panel.is_visible(), f"{label} panel did not become visible"
                audit = audit_visible_text(page)
                assert audit["text_items"] > 10, f"{theme} {label} panel has too little visible text"
                overflow = panel.evaluate("""panel => {
                  const viewport = document.documentElement.clientWidth;
                  const panelRect = panel.getBoundingClientRect();
                  const visible = element => {
                    const rect = element.getBoundingClientRect();
                    if (rect.width < 1 || rect.height < 1) return false;
                    for (let node = element; node && node.nodeType === 1; node = node.parentElement) {
                      const style = getComputedStyle(node);
                      if (style.display === 'none' || style.visibility === 'hidden' || node.hidden) return false;
                    }
                    return true;
                  };
                  const escapes = [];
                  for (const element of [panel, ...panel.querySelectorAll('*')]) {
                    if (!visible(element)) continue;
                    const rect = element.getBoundingClientRect();
                    if (rect.left >= -1 && rect.right <= viewport + 1) continue;
                    let clippedByScroller = false;
                    for (let parent = element.parentElement; parent && panel.contains(parent); parent = parent.parentElement) {
                      const style = getComputedStyle(parent);
                      if (style.overflowX === 'auto' || style.overflowX === 'scroll') {
                        clippedByScroller = true;
                        break;
                      }
                    }
                    if (!clippedByScroller) escapes.push({
                      tag: element.tagName.toLowerCase(),
                      className: String(element.className || '').slice(0, 80),
                      left: Math.round(rect.left), right: Math.round(rect.right)
                    });
                  }
                  return {
                    viewport,
                    panel: {left: Math.round(panelRect.left), right: Math.round(panelRect.right)},
                    escapes
                  };
                }""")
                assert overflow["panel"]["left"] >= -1 and overflow["panel"]["right"] <= overflow["viewport"] + 1, (
                    f"{theme} {label} panel exceeds 390px: {overflow}"
                )
                assert overflow["escapes"] == [], (
                    f"{theme} {label} has elements outside the viewport and outside a scroll wrapper: "
                    f"{overflow['escapes']}"
                )
                assert audit["failures"] == [], f"{theme} {label} WCAG AA failures: " + "; ".join(
                    f"{row['ratio']}:1 (needs {row['required']}:1) · {row['text']} · {row['className']}"
                    for row in audit["failures"]
                )
        page.close()

    _run_with_edge_or_skip(audit_with_browser)


@pytest.mark.browser
def test_rendered_rl_web_markup_meets_wcag_aa_in_light_and_dark():
    from pramaan_env.server.web_ui import _markup

    def audit_with_browser(p, browser):
        page = browser.new_page(viewport={"width": 1440, "height": 900}, color_scheme="light")
        page.set_content("<!doctype html><html><head><meta charset='utf-8'></head><body>" + _markup()
                         + "</body></html>", wait_until="domcontentloaded", timeout=30000)
        for theme in ("light", "dark"):
            page.emulate_media(color_scheme=theme)
            page.evaluate("theme => { document.documentElement.className = theme; document.body.className = theme; }", theme)
            audit = audit_visible_text(page)
            assert not audit["horizontal_overflow"]
            assert audit["failures"] == [], f"{theme} /web/ WCAG AA failures: " + "; ".join(
                f"{row['ratio']}:1 (needs {row['required']}:1) · {row['text']} · {row['className']}"
                for row in audit["failures"]
            )
        page.close()

    _run_with_edge_or_skip(audit_with_browser)


def _theme_luminance(hex_color: str) -> float:
    rgb = [int(hex_color[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4 for value in rgb]
    return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2]


def _theme_contrast(foreground: str, background: str) -> float:
    light, dark = sorted((_theme_luminance(foreground), _theme_luminance(background)), reverse=True)
    return (light + .05) / (dark + .05)


def _theme_tokens(selector: str) -> dict[str, str]:
    css = _load_pramaan_css()
    start = css.index(selector)
    opening = css.index("{", start)
    closing = css.index("}", opening)
    return dict(re.findall(r"(--[\w-]+)\s*:\s*(#[\da-fA-F]{6})", css[opening + 1:closing]))


def _contrast_rgb(foreground: str, background: str) -> float:
    def luminance(color: str) -> float:
        channels = [int(value) / 255 for value in re.findall(r"\d+", color)[:3]]
        linear = [value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4 for value in channels]
        return .2126 * linear[0] + .7152 * linear[1] + .0722 * linear[2]

    light, dark = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (light + .05) / (dark + .05)


def test_theme_defines_every_used_token_and_explicit_text_contrast():
    css = _load_pramaan_css()
    used = set(re.findall(r"var\((--[\w-]+)", css))
    defined = set(re.findall(r"(--[\w-]+)\s*:", css))
    assert not (used - defined), f"Undefined theme tokens: {sorted(used - defined)}"

    def tokens_for(selector: str) -> dict[str, str]:
        start = css.index(selector)
        opening = css.index("{", start)
        closing = css.index("}", opening)
        return dict(re.findall(r"(--[\w-]+)\s*:\s*(#[\da-fA-F]{6})", css[opening + 1:closing]))

    themes = {
        "light": tokens_for(":root"),
        "dark": tokens_for("html.dark, body.dark, .dark"),
        "system dark": tokens_for("html:not(.light):not(.dark), body:not(.light):not(.dark)"),
    }
    for theme, colors in themes.items():
        for token in ("text", "muted"):
            for surface in ("card", "paper", "subtle"):
                fg, bg = colors.get(f"--pramaan-{token}"), colors.get(f"--pramaan-card{'' if surface == 'card' else '-subtle'}")
                if surface == "paper":
                    bg = colors.get("--pramaan-paper")
                assert fg and bg and _theme_contrast(fg, bg) >= 4.5, (theme, token, surface, fg, bg)


def test_selected_home_grammar_count_keeps_contrast_over_selected_chip():
    css = _load_pramaan_css()
    selector = (
        '.gradio-container .pramaan-ui .pramaan-wall-grammar-chip'
        '[aria-pressed="true"] small { color: var(--pramaan-on-accent) !important; }'
    )
    assert selector in css

    for theme, selector_name in (
        ("light", ":root"),
        ("dark", "html.dark, body.dark, .dark"),
        ("system dark", "html:not(.light):not(.dark), body:not(.light):not(.dark)"),
    ):
        colors = _theme_tokens(selector_name)
        foreground = colors.get("--pramaan-on-accent")
        background = colors.get("--pramaan-indigo")
        assert foreground and background
        assert _theme_contrast(foreground, background) >= 4.5, (
            theme, foreground, background,
        )


def test_models_charts_and_matrix_state_colours_meet_wcag_contrast_in_both_themes():
    themes = {
        "light": _theme_tokens(":root"),
        "dark": _theme_tokens("html.dark, body.dark, .dark"),
    }
    text_pairs = (
        ("indigo", "card-raised", "chart headings and model names"),
        ("indigo", "card-subtle", "composition and matrix headings"),
        ("ink", "card-raised", "model labels and point labels"),
        ("ink", "card-subtle", "composition labels"),
        ("muted", "card-raised", "chart notes and text alternatives"),
        ("muted", "card-subtle", "chart captions and axes"),
        ("muted", "paper", "page-level result labels"),
        ("paper", "indigo", "selected control label"),
        ("pass", "pass-bg", "passed matrix state"),
        ("pending", "pending-bg", "repaired matrix state"),
        ("fail", "fail-bg", "failed matrix state"),
        ("muted", "card-subtle", "not-run matrix state"),
    )
    graphic_pairs = (
        ("chart-1", "card-raised", "cost-chart points"),
        ("chart-3", "card-subtle", "unreadable composition and failure bars"),
        ("pass", "card-subtle", "passed composition segment"),
        ("fail", "card-subtle", "rule-failure composition segment"),
        ("madder", "card-raised", "cost frontier line"),
        ("muted", "card-raised", "cost-chart axes and label leaders"),
        ("pass", "pass-bg", "passed matrix cell mark"),
        ("pending", "pending-bg", "repaired matrix cell mark"),
        ("fail", "fail-bg", "failed matrix cell mark and near-invisible hatch"),
        ("muted", "card-subtle", "not-run matrix cell mark"),
        ("indigo", "card-raised", "chart keyboard focus ring"),
    )
    for theme, colors in themes.items():
        for foreground, background, context in text_pairs:
            ratio = _theme_contrast(colors[f"--pramaan-{foreground}"], colors[f"--pramaan-{background}"])
            assert ratio >= 4.5, f"{theme}: {context} contrast is {ratio:.2f}:1"
        for foreground, background, context in graphic_pairs:
            ratio = _theme_contrast(colors[f"--pramaan-{foreground}"], colors[f"--pramaan-{background}"])
            assert ratio >= 3, f"{theme}: {context} contrast is {ratio:.2f}:1"


def test_every_models_chart_has_a_text_alternative_and_visible_keyboard_focus():
    import re
    from pramaan.app import _benchmark_summary, _load_gallery_arm_documents, _load_gallery_index, _home_suite_tasks
    from pramaan.ui.leaderboard import render_model_report

    tasks = _home_suite_tasks()
    backgrounds = {task["id"]: task["palette"][0] for task in tasks if task.get("palette")}
    rendered = render_model_report(
        _benchmark_summary(), gallery_documents=_load_gallery_arm_documents(_load_gallery_index()),
        suite_tasks=tasks, task_backgrounds=backgrounds,
    )
    charts = {
        "model-report-chart": ("pramaan-model-chart", "Text alternative:"),
        "model-cost-chart": ("pramaan-model-cost", "Text alternative:"),
        "model-results-matrix": ("pramaan-results-matrix", "Text alternative:"),
        "first-try-failures-chart": ("pramaan-first-failures", "Text alternative:"),
    }
    css = _load_pramaan_css()
    for testid, (class_name, alternative) in charts.items():
        opening = re.search(rf'<section[^>]*data-testid="{testid}"[^>]*>', rendered)
        assert opening and 'tabindex="0"' in opening.group(0), testid
        section = rendered[opening.end():].split("</section>", 1)[0]
        assert alternative in section, testid
        assert f".{class_name}:focus-visible" in css, testid
    for testid in ("model-cost-chart",):
        assert re.search(rf'<g class="point" tabindex="0"[^>]*data-testid="{testid.replace("chart", "point")}[^>]*>', rendered)
    assert ".pramaan-cost-plot .point:focus-visible circle" in css
    assert ".pramaan-latency-plot" not in css


def test_gradio_native_components_use_the_pramaan_contrast_palette():
    from pramaan.app import create_brand_theme

    theme = create_brand_theme()
    expected = {
        "body_background_fill": "#f7f3ec",
        "body_background_fill_dark": "#151b22",
        "body_text_color": "#172f4a",
        "body_text_color_dark": "#e8eef5",
        "body_text_color_subdued": "#455463",
        "body_text_color_subdued_dark": "#cbd6e0",
        "block_background_fill": "#fffdf9",
        "block_background_fill_dark": "#202a34",
        "input_background_fill": "#ffffff",
        "input_background_fill_dark": "#293542",
    }
    for token, color in expected.items():
        assert getattr(theme, token) == color, (token, getattr(theme, token), color)

    for fg, bg in (
        (theme.body_text_color, theme.body_background_fill),
        (theme.body_text_color_subdued, theme.block_background_fill),
        (theme.body_text_color_dark, theme.body_background_fill_dark),
        (theme.body_text_color_subdued_dark, theme.block_background_fill_dark),
        (theme.body_text_color_dark, theme.input_background_fill_dark),
    ):
        assert _theme_contrast(fg, bg) >= 4.5, (fg, bg)


def test_pramaan_tabs_override_gradio_important_prose_link_color():
    css = _load_pramaan_css()
    assert ".gradio-container .pramaan-ui .pramaan-ui-tabs > a.pramaan-ui-tab" in css
    assert "color: var(--pramaan-indigo) !important" in css
    assert "color: var(--pramaan-on-accent) !important" in css

    for foreground, background in (
        ("#172f4a", "#fffdf9"), ("#ffffff", "#176a66"),
        ("#e8eef5", "#202a34"), ("#10272a", "#75d0c5"),
    ):
        assert _theme_contrast(foreground, background) >= 4.5, (foreground, background)


def test_pramaan_native_selected_controls_and_download_link_override_gradio_colors():
    css = _load_pramaan_css()
    for required in (
        ".gradio-container .tab-nav button.selected",
        ".gradio-container .pramaan-ui .pramaan-ui-subtabs > button[aria-pressed=\"true\"]",
        ".gradio-container .pramaan-ui a.pramaan-ui-button",
    ):
        assert required in css
    assert css.count("color: var(--pramaan-on-accent) !important") >= 2
    assert "color: var(--pramaan-indigo) !important" in css

    for foreground, background in (
        ("#ffffff", "#176a66"),
        ("#10272a", "#75d0c5"),
        ("#172f4a", "#fffdf9"),
        ("#e8eef5", "#202a34"),
    ):
        assert _theme_contrast(foreground, background) >= 4.5, (foreground, background)


def test_selected_wall_model_status_uses_full_contrast_at_twelve_pixels():
    css = _load_pramaan_css()
    assert ".pramaan-wall-model-chip small" in css
    assert "font-size: 12px; opacity: 1" in css
    assert '.gradio-container .pramaan-ui .pramaan-wall-model-chip[aria-checked="true"] small' in css
    assert "color: var(--pramaan-on-accent) !important; font-size: 12px !important; opacity: 1 !important" in css
    for foreground, background in (
        ("#ffffff", "#176a66"),
        ("#10272a", "#75d0c5"),
    ):
        assert _theme_contrast(foreground, background) >= 4.5, (foreground, background)
