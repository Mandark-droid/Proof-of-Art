"""WCAG AA text contrast audit for local or hosted Pramaan UI builds.

Usage: python scripts/ui_contrast_audit.py <base_url> [output_dir]
The base URL is the app root (for example http://localhost:7860).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlencode

from playwright.sync_api import sync_playwright


ROUTES = {
    "overview": "#/overview",
    "leaderboard": "#/leaderboard",
    "studio": "#/studio",
    "gallery": "#/states",
    "traditions": "#/states?view=traditions",
    "rl": "#/rl",
    "web": "/web/",
}
VIEWPORTS = {"desktop": (1440, 900), "mobile": (390, 844)}

CONTRAST_SCRIPT = r"""
() => {
  const parse = (value) => {
    let match = value.match(/rgba?\(([^)]+)\)/i);
    if (match) {
      const values = match[1].split(/[ ,/]+/).filter(Boolean).map(Number);
      return [values[0], values[1], values[2], values.length > 3 ? values[3] : 1];
    }
    match = value.match(/color\(srgb\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s*\/\s*([\d.]+))?\s*\)/i);
    if (match) return [Number(match[1])*255, Number(match[2])*255, Number(match[3])*255, match[4] === undefined ? 1 : Number(match[4])];
    return null;
  };
  const over = (top, bottom) => {
    const a = top[3];
    return [top[0]*a + bottom[0]*(1-a), top[1]*a + bottom[1]*(1-a), top[2]*a + bottom[2]*(1-a), 1];
  };
  const luminance = (c) => {
    const linear = (v) => { v /= 255; return v <= 0.04045 ? v/12.92 : Math.pow((v+0.055)/1.055, 2.4); };
    return 0.2126*linear(c[0]) + 0.7152*linear(c[1]) + 0.0722*linear(c[2]);
  };
  const ratio = (a,b) => { const x=luminance(a), y=luminance(b); return (Math.max(x,y)+0.05)/(Math.min(x,y)+0.05); };
  const visible = (el) => {
    const r=el.getBoundingClientRect();
    if (r.width<2 || r.height<2) return false;
    for (let n=el; n&&n.nodeType===1; n=n.parentElement) {
      const cs=getComputedStyle(n);
      if (cs.display==='none'||cs.visibility==='hidden'||n.hidden) return false;
    }
    return true;
  };
  const effectiveBackground = (el) => {
    const chain=[]; let image=false;
    for (let n=el;n&&n.nodeType===1;n=n.parentElement) {
      const cs=getComputedStyle(n), c=parse(cs.backgroundColor);
      if (cs.backgroundImage&&cs.backgroundImage!=='none') image=true;
      if (c&&c[3]>0) chain.push(c);
      if (c&&c[3]>=1) break;
    }
    let bg=[255,255,255,1];
    for (let i=chain.length-1;i>=0;i--) bg=over(chain[i],bg);
    return {bg,image};
  };
  const rows=[], seen=new Set(), walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
  let node;
  while ((node=walker.nextNode())) {
    const text=node.nodeValue.replace(/\s+/g,' ').trim();
    if (text.length<2) continue;
    const el=node.parentElement;
    if (!el||['SCRIPT','STYLE','NOSCRIPT'].includes(el.tagName)||!visible(el)) continue;
    const cs=getComputedStyle(el); let fg=parse(cs.color); if (!fg) continue;
    let opacity=1;
    for (let n=el;n&&n.nodeType===1;n=n.parentElement) opacity*=parseFloat(getComputedStyle(n).opacity||'1');
    fg=[fg[0],fg[1],fg[2],fg[3]*opacity];
    const {bg,image}=effectiveBackground(el), effectiveFg=over(fg,bg), value=ratio(effectiveFg,bg);
    const size=parseFloat(cs.fontSize), bold=parseInt(cs.fontWeight,10)>=700;
    const need=(size>=24||(size>=18.66&&bold))?3:4.5;
    const key=el.tagName+'|'+String(el.className||'').slice(0,70)+'|'+cs.color+'|'+JSON.stringify(bg.map(Math.round))+'|'+text.slice(0,24);
    if (seen.has(key)) continue; seen.add(key);
    rows.push({tag:el.tagName.toLowerCase(),className:String(el.className||'').slice(0,70),text:text.slice(0,80),ratio:Math.round(value*100)/100,required:need,foreground:effectiveFg.map(Math.round),background:bg.map(Math.round),imageBackground:image,fail:value<need});
  }
  return {rows,htmlClass:document.documentElement.className,bodyClass:document.body.className,
    viewportWidth:document.documentElement.clientWidth,documentWidth:document.documentElement.scrollWidth};
}
"""


def audit_visible_text(page) -> dict:
    """Inspect visible rendered text and return measured contrast failures."""
    data = page.evaluate(CONTRAST_SCRIPT)
    failures = [row for row in data["rows"] if row["fail"]]
    return {**data, "text_items": len(data["rows"]), "failures": failures,
            "horizontal_overflow": data["documentWidth"] > data["viewportWidth"]}


def run_audit(base_url: str, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    base = base_url.rstrip("/")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, timeout=30_000)
        for theme in ("light", "dark"):
            for viewport_name, (width, height) in VIEWPORTS.items():
                context = browser.new_context(color_scheme=theme, viewport={"width": width, "height": height})
                page = context.new_page()
                query = urlencode({"__theme": theme})
                page.goto(f"{base}/?{query}", wait_until="domcontentloaded", timeout=120000)
                page.wait_for_selector("#pramaan-custom-app", timeout=90000)
                page.wait_for_timeout(1200)
                for route_name, route in ROUTES.items():
                    if route.startswith("/"):
                        page.goto(f"{base}{route}?{query}", wait_until="domcontentloaded", timeout=120000)
                        page.wait_for_selector("#pramaan-rl", timeout=90000)
                    else:
                        page.goto(f"{base}/?{query}", wait_until="domcontentloaded", timeout=120000)
                        page.wait_for_selector("#pramaan-custom-app", timeout=90000)
                        page.evaluate("route => { window.location.hash = route; }", route)
                    page.wait_for_timeout(450)
                    data = audit_visible_text(page)
                    key = f"{theme}-{viewport_name}-{route_name}"
                    report[key] = {
                        "theme": theme,
                        "viewport": {"width": width, "height": height},
                        "text_items": data["text_items"],
                        "failures": len(data["failures"]),
                        "horizontal_overflow": data["horizontal_overflow"],
                        "document_width": data["documentWidth"],
                        "worst": sorted(data["failures"], key=lambda row: row["ratio"])[:20],
                        "html_class": data["htmlClass"],
                        "body_class": data["bodyClass"],
                    }
                    page.screenshot(path=str(output_dir / f"{key}.png"), full_page=False)
                    print(f"{key:30} items={data['text_items']:4} FAIL={len(data['failures']):3} "
                          f"OVERFLOW={data['horizontal_overflow']}", flush=True)
                context.close()
        browser.close()
    (output_dir / "contrast_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="app root URL, such as http://localhost:7860")
    parser.add_argument("output_dir", nargs="?", type=Path, default=Path("artifacts/ui-contrast-audit"))
    args = parser.parse_args()
    report = run_audit(args.base_url, args.output_dir)
    failures = sum(row["failures"] for row in report.values())
    overflows = sum(row["horizontal_overflow"] for row in report.values())
    print(f"Total rendered text contrast failures: {failures}", flush=True)
    print(f"Routes with horizontal overflow: {overflows}", flush=True)
    return 1 if failures or overflows else 0


if __name__ == "__main__":
    raise SystemExit(main())
