"""Verify Hugging Face's embedded Studio and the public Studio/RL routes.

Run after deploying the Space image:
    uv run python scripts/smoke_embed.py --output-dir artifacts/embed-smoke

For a local server, pass --base-url http://localhost:8000 --embed-url ...
and install Playwright plus a browser (the Hugging Face-page check requires
network access to both huggingface.co and the Space).
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from urllib.parse import urlsplit


DEFAULT_BASE_URL = "https://kshitijthakkar-pramaan-proof-of-art.hf.space"
DEFAULT_EMBED_URL = "https://huggingface.co/spaces/kshitijthakkar/pramaan-proof-of-art"
EXPECTED_REDIRECTS = {"/": "/studio/", "/studio": "/studio/", "/web": "/web/"}


def is_known_hf_avatar_404(status: int, url: str) -> bool:
    parsed = urlsplit(url)
    return status == 404 and parsed.netloc.lower() == "huggingface.co" and parsed.path == "/api/organizations/kshitijthakkar/avatar"


def attach_error_audit(page, errors: list[str]) -> None:
    def on_console(message):
        # Browser console omits the failed resource URL for generic 404 messages;
        # the response hook below records the URL and filters only the known HF avatar.
        if message.type == "error" and not message.text.startswith("Failed to load resource:"):
            errors.append(message.text)

    def on_response(response):
        if response.status >= 400 and not is_known_hf_avatar_404(response.status, response.url):
            errors.append(f"HTTP {response.status} {response.url}")

    page.on("console", on_console)
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("response", on_response)


def validate_home_wall_rendering(*, state_count: int, art_count: int, loaded_art_count: int,
                                 placeholder_count: int, beam_selected: bool,
                                 non_ring_mark_count: int) -> None:
    if state_count != 36:
        raise AssertionError(f"Home Wall must render 36 state tiles; found {state_count}")
    if art_count < 36 or loaded_art_count < 36 or placeholder_count:
        raise AssertionError(
            "Home Wall reference art is missing or not loaded: "
            f"{art_count} art elements, {loaded_art_count} loaded, {placeholder_count} placeholders"
        )
    if not beam_selected:
        raise AssertionError("Home Wall Beam model must be selected for the result-mark smoke")
    if non_ring_mark_count < 1:
        raise AssertionError("Home Wall Beam results must include at least one non-ring status mark")


def validate_visitor_panels(*, proof_step_count: int, repaired_tile_loaded: bool,
                            model_keys: list[str], fable_cap_visible: bool,
                            models_signin_bar_present: bool,
                            try_state: str, try_example: str, rl_snippet: str) -> None:
    if proof_step_count != 7:
        raise AssertionError(f"Home Proof Replay must expose seven replay/proof steps; found {proof_step_count}")
    if not repaired_tile_loaded:
        raise AssertionError("Home Proof Replay repaired tile is missing or not loaded")
    if len(model_keys) != 11 or len(set(model_keys)) != 11:
        raise AssertionError(f"Models must show eleven unique official model rows; found {model_keys!r}")
    if not fable_cap_visible:
        raise AssertionError("Models must show Claude Fable 5.1 with 72/72 coverage and its authorized USD 22 cap")
    if models_signin_bar_present:
        raise AssertionError("Models tab must not show the Sign in to publish bar")
    if try_state.casefold() != "gujarat" or not try_example.strip():
        raise AssertionError("Try it must open with its Gujarat state and a selected built-in example")
    if "PramaanEnv" not in rl_snippet or "env.reset" not in rl_snippet or "env.step" not in rl_snippet:
        raise AssertionError("Build with it must show the runnable PramaanEnv reset/step snippet")


def validate_kinnauri_replay(text: str) -> None:
    required = ("Himachal Pradesh", "Kinnauri", "16, 21", "47, 21")
    missing = [value for value in required if value not in text]
    if missing:
        raise AssertionError(f"Home Proof Replay is missing saved Kinnauri evidence: {missing!r}")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, new_url):
        return None


def check_http_redirects(base_url: str) -> None:
    opener = build_opener(_NoRedirect)
    base_host = urlsplit(base_url).netloc.lower()
    for path, expected in EXPECTED_REDIRECTS.items():
        request = Request(base_url.rstrip("/") + path, method="GET")
        try:
            response = opener.open(request, timeout=15)
            status, location = response.status, response.headers.get("Location", "")
        except HTTPError as error:
            status, location = error.code, error.headers.get("Location", "")
        assert status == 307, f"{path}: expected 307, got {status}"
        target = urlsplit(location)
        assert target.path == expected, f"{path}: expected redirect path {expected!r}, got {location!r}"
        assert not target.netloc or target.netloc.lower() == base_host, f"{path}: cross-origin redirect {location!r}"
        print(f"HTTP {path} -> {status} {location}")


async def check_browser(base_url: str, embed_url: str, output_dir: Path, browser_name: str) -> None:
    try:
        from playwright.async_api import async_playwright
    except ImportError as error:  # pragma: no cover - depends on optional browser tooling
        raise SystemExit("Install Playwright before running the browser smoke: uv pip install playwright") from error

    output_dir.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as playwright:
        browser_type = getattr(playwright, browser_name)
        chrome_candidates = (
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "Google/Chrome/Application/chrome.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "Google/Chrome/Application/chrome.exe",
        )
        if browser_name == "chromium" and sys.platform == "win32" and any(path.is_file() for path in chrome_candidates):
            browser = await browser_type.launch(channel="chrome", headless=True)
        else:
            browser = await browser_type.launch(headless=True)
        errors: list[str] = []

        context = await browser.new_context(viewport={"width": 1440, "height": 900})
        page = await context.new_page()
        attach_error_audit(page, errors)
        await page.goto(embed_url, wait_until="domcontentloaded", timeout=60_000)
        await page.wait_for_timeout(1_000)
        found = False
        deadline = asyncio.get_running_loop().time() + 45
        while asyncio.get_running_loop().time() < deadline:
            for frame in page.frames[1:]:
                try:
                    body = await frame.locator("body").inner_text(timeout=1_000)
                except Exception:
                    continue
                if "Proof of Art" in body:
                    found = True
                    break
            if found:
                break
            await page.wait_for_timeout(500)
        assert found, "The Hugging Face Space page did not render 'Proof of Art' inside an iframe within 45 seconds"
        assert not errors, "Browser errors: " + " | ".join(errors)
        await page.screenshot(path=str(output_dir / "hf-embed-desktop.png"), full_page=True)
        print("Hugging Face page iframe rendered Proof of Art; no Mixed Content errors")
        await context.close()

        for width, height, label in ((1440, 900, "desktop"), (390, 844, "phone")):
            for route, name in (("/studio/#/overview", "studio"), ("/web/", "rl")):
                context = await browser.new_context(viewport={"width": width, "height": height})
                route_page = await context.new_page()
                route_errors: list[str] = []
                attach_error_audit(route_page, route_errors)
                await route_page.goto(base_url.rstrip("/") + route, wait_until="domcontentloaded", timeout=60_000)
                studio_route = route.startswith("/studio/")
                expected_text = "Proof of Art" if studio_route else "AI drew it. Math checked it."
                body = route_page.locator("body")
                deadline = asyncio.get_running_loop().time() + 45
                text = ""
                while asyncio.get_running_loop().time() < deadline:
                    text = await body.inner_text()
                    if expected_text in text and len(text.strip()) > 100:
                        break
                    await route_page.wait_for_timeout(500)
                assert expected_text in text, f"{route} did not render {expected_text!r} at {width}px"
                assert len(text.strip()) > 100, f"{route} returned an effectively blank page at {width}px"
                assert not route_errors, f"{route} console errors at {width}px: " + " | ".join(route_errors)
                if route == "/web/":
                    for selector in ("#poa-reward-meter", "#poa-attempt", "#poa-done", "#poa-fail", "#poa-unreadable"):
                        assert await route_page.locator(selector).count() == 1, f"/web/ missing result control {selector}"
                    assert "0.2 to under 0.8" in await body.inner_text(), "/web/ reward bands are missing the partial-reward range"
                    await route_page.wait_for_function(
                        "document.querySelector('#poa-brief')?.innerText.toLowerCase().includes('kinnauri')", timeout=45_000
                    )
                    assert "kinnauri" in (await route_page.locator("#poa-brief").inner_text()).casefold(), "/web/ did not load the Kinnauri task"
                    await route_page.locator("#poa-fail").click()
                    await route_page.locator("#poa-run").click()
                    await route_page.wait_for_function(
                        "document.querySelector('#poa-status')?.innerText.includes('0.725')", timeout=20_000
                    )
                    failed_result = await route_page.locator("#poa-observation").text_content()
                    assert "symmetry_d2" in failed_result and "16" in failed_result and "47" in failed_result, \
                        "/web/ saved failure does not expose the Kinnauri symmetry witness"
                    await route_page.wait_for_function(
                        "(() => { const image = document.querySelector('#poa-preview'); return image && !image.hidden && image.complete && image.naturalWidth === 384; })()",
                        timeout=15_000,
                    )
                    await route_page.locator("#poa-reset").click()
                    await route_page.wait_for_function(
                        "document.querySelector('#poa-status')?.innerText.includes('Episode ready')", timeout=20_000
                    )
                    await route_page.locator("#poa-run").click()
                    await route_page.wait_for_function(
                        "document.querySelector('#poa-status')?.innerText.includes('reward 1.0')", timeout=20_000
                    )
                    await route_page.locator("#poa-reset").click()
                    await route_page.wait_for_function(
                        "document.querySelector('#poa-status')?.innerText.includes('Episode ready')", timeout=20_000
                    )
                    await route_page.locator("#poa-unreadable").click()
                    await route_page.locator("#poa-run").click()
                    await route_page.wait_for_function(
                        "document.querySelector('#poa-status')?.innerText.includes('Reward 0')", timeout=20_000
                    )
                if studio_route:
                    home_panel = route_page.locator('[data-tab-panel="benchmark"]')
                    replay = home_panel.locator('[data-testid="proof-replay"]')
                    validate_kinnauri_replay(await replay.inner_text())
                    proof_step_count = await replay.locator(".pramaan-proof-step[data-proof-step]").count()
                    repaired_tile = replay.locator('[data-proof-result] img.pramaan-proof-tile')
                    repaired_tile_loaded = await repaired_tile.count() == 1 and await repaired_tile.evaluate(
                        "image => image.complete && image.naturalWidth > 0"
                    )

                    await route_page.evaluate("window.location.hash = '#/leaderboard'")
                    models_panel = route_page.locator('[data-tab-panel="leaderboard"]')
                    await models_panel.wait_for(state="visible", timeout=5_000)
                    model_rows = models_panel.locator('[data-testid^="model-report-row-"]')
                    model_keys = await model_rows.evaluate_all(
                        "rows => rows.map(row => row.dataset.modelKey || '')"
                    )
                    fable_row = models_panel.locator('[data-testid="model-report-row-claude-fable-5-1"]')
                    fable_text = await fable_row.inner_text() if await fable_row.count() == 1 else ""
                    fable_cap_visible = "72 of 72 evaluated" in fable_text and "USD 22 cap" in fable_text
                    models_text = await models_panel.inner_text()
                    models_signin_bar_present = "Sign in to publish" in models_text

                    await route_page.evaluate("window.location.hash = '#/studio'")
                    try_panel = route_page.locator('[data-tab-panel="studio"]')
                    await try_panel.wait_for(state="visible", timeout=5_000)
                    try_state = await try_panel.locator('[data-field="state"]').input_value()
                    try_example = await try_panel.locator('[data-field="example"]').input_value()

                    await route_page.evaluate("window.location.hash = '#/build'")
                    build_panel = route_page.locator('[data-tab-panel="rl"]')
                    await build_panel.wait_for(state="visible", timeout=5_000)
                    # The contract and download cards also contain code blocks;
                    # the first block is the client reset/step example.
                    rl_snippet = await build_panel.locator("pre code").first.inner_text()
                    validate_visitor_panels(
                        proof_step_count=proof_step_count,
                        repaired_tile_loaded=repaired_tile_loaded,
                        model_keys=model_keys,
                        fable_cap_visible=fable_cap_visible,
                        models_signin_bar_present=models_signin_bar_present,
                        try_state=try_state,
                        try_example=try_example,
                        rl_snippet=rl_snippet,
                    )

                    await route_page.evaluate("window.location.hash = '#/overview'")
                    await route_page.locator('[data-tab-panel="benchmark"]').wait_for(state="visible", timeout=5_000)
                    tiles = route_page.locator(".pramaan-wall-state")
                    state_count = await tiles.count()
                    for index in range(state_count):
                        await tiles.nth(index).scroll_into_view_if_needed()
                    art_count = await route_page.locator(".pramaan-wall-state img.pramaan-wall-art").count()
                    placeholder_count = await route_page.locator(
                        ".pramaan-wall-state .pramaan-wall-empty-art"
                    ).count()
                    if art_count >= 36:
                        await route_page.wait_for_function(
                            """() => {
                              const images = [...document.querySelectorAll('.pramaan-wall-state img.pramaan-wall-art')];
                              return images.length >= 36 && images.every(image => image.complete && image.naturalWidth > 0);
                            }""",
                            timeout=15_000,
                        )
                    loaded_art_count = await route_page.locator(
                        ".pramaan-wall-state img.pramaan-wall-art"
                    ).evaluate_all("images => images.filter(image => image.complete && image.naturalWidth > 0).length")
                    beam = route_page.locator('[data-wall-model-choice="beam-hf"]')
                    beam_selected = await beam.count() == 1 and await beam.get_attribute("aria-checked") == "true"
                    non_ring_mark_count = await route_page.locator(
                        '.pramaan-wall-state [data-wall-mark][data-status="first"], '
                        '.pramaan-wall-state [data-wall-mark][data-status="repair"], '
                        '.pramaan-wall-state [data-wall-mark][data-status="fail"]'
                    ).count()
                    validate_home_wall_rendering(
                        state_count=state_count,
                        art_count=art_count,
                        loaded_art_count=loaded_art_count,
                        placeholder_count=placeholder_count,
                        beam_selected=beam_selected,
                        non_ring_mark_count=non_ring_mark_count,
                    )
                await route_page.screenshot(path=str(output_dir / f"{name}-{label}.png"), full_page=True)
                print(f"{route} rendered at {width}x{height}; body text {len(text)} chars; console clean")
                await context.close()
        await browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--embed-url", default=DEFAULT_EMBED_URL)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/embed-smoke"))
    parser.add_argument("--browser", choices=("chromium", "firefox", "webkit"), default="chromium")
    parser.add_argument("--http-only", action="store_true", help="Only check the pure-HTTP redirect contract")
    args = parser.parse_args()
    check_http_redirects(args.base_url)
    if not args.http_only:
        asyncio.run(check_browser(args.base_url, args.embed_url, args.output_dir, args.browser))


if __name__ == "__main__":
    main()
