"""Accessible, data-backed proof card for replay and Studio results."""

from __future__ import annotations

import base64
import html
import json
import re

from .check_labels import check_label, plain_check_label


_PNG_DATA_URI = re.compile(r"data:image/png;base64,[A-Za-z0-9+/]+={0,2}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _counterexample_text(checks: list[dict]) -> str:
    for check in checks:
        witness = check.get("counterexample")
        if not isinstance(witness, dict):
            continue
        original, transformed = witness.get("original"), witness.get("transformed")
        if (isinstance(original, list) and len(original) == 2
                and isinstance(transformed, list) and len(transformed) == 2
                and all(type(value) is int and 0 <= value < 64 for value in original + transformed)):
            relation = "its mirror" if witness.get("transform") == "F_vertical" else "its transformed pixel"
            return (f"Pixel ({original[0]}, {original[1]}) and {relation} "
                    f"({transformed[0]}, {transformed[1]}) differ")
        pixel = witness.get("pixel")
        if isinstance(pixel, list) and len(pixel) == 2 and all(type(value) is int for value in pixel):
            return f"Pixel ({pixel[0]}, {pixel[1]}) breaks this rule"
    return "The checker found a geometric mismatch."


def render_proof_card(
    certificate: dict,
    *,
    image_uri: str = "",
    state: str = "",
    tradition: str = "",
    reward: int | float | None = None,
    download_html: str = "",
    replay_href: str | None = None,
) -> str:
    """Render a passing or failing verifier certificate without overstating it."""
    if not isinstance(certificate, dict):
        return '<section class="pramaan-proof-card" data-proof-card role="status">No checker certificate is available.</section>'
    checks = [check for check in certificate.get("checks", []) if isinstance(check, dict)]
    passed = (certificate.get("all_hard_constraints_pass") is True
              and bool(checks) and all(check.get("status") == "PASS" for check in checks))
    exclusions = certificate.get("exclusions", [])
    if not isinstance(exclusions, list):
        exclusions = []
    exclusions_html = "".join(
        f"<li>{html.escape(item)}</li>" for item in exclusions if isinstance(item, str)
    )
    labels = []
    visible_checks = checks if passed else [check for check in checks if check.get("status") == "FAIL"]
    for check in visible_checks:
        status = "pass" if check.get("status") == "PASS" else "fail"
        marker = "✓" if status == "pass" else "×"
        label = plain_check_label(check) if passed else check_label(check.get("id"))
        labels.append(
            f'<li class="pramaan-proof-card-check is-{status}">'
            f'<span aria-hidden="true">{marker}</span>{html.escape(label)}</li>'
        )
    image = (f'<img class="pramaan-proof-card-image" src="{image_uri}" '
             f'alt="Verified pattern tile for {html.escape(state or "this design", quote=True)}" '
             'loading="lazy" decoding="async">') if _PNG_DATA_URI.fullmatch(image_uri or "") else ""
    context = " · ".join(value for value in (state, tradition) if isinstance(value, str) and value.strip())
    context_html = f'<p class="pramaan-proof-card-context">{html.escape(context)}</p>' if context else ""
    exclusions_block = (
        '<div class="pramaan-proof-card-exclusions"><strong>What this certificate excludes</strong>'
        f'<ul>{exclusions_html}</ul></div>'
        '<p class="pramaan-proof-card-note">Anyone can recompute this. It is not a digital signature.</p>'
    )
    action = (f'<a class="pramaan-ui-button secondary" href="{html.escape(replay_href, quote=True)}">'
              'Replay saved proof on Home</a>' if replay_href else
              '<button type="button" class="pramaan-ui-button secondary" data-proof-replay>Replay proof</button>')

    if passed:
        total = len(checks)
        title = f"{total} of {total} checks passed"
        hash_rows = []
        for field, label in (("dsl_sha256", "Design SHA-256"), ("mask_sha256", "Pixels SHA-256")):
            digest = certificate.get(field)
            if isinstance(digest, str) and _SHA256.fullmatch(digest):
                hash_rows.append(
                    f'<div><dt>{label}</dt><dd><code>{digest[:12]}…</code></dd></div>'
                )
        reward_html = (f'<p class="pramaan-proof-card-reward">Reward {html.escape(str(reward))}</p>'
                       if isinstance(reward, (int, float)) and not isinstance(reward, bool) else "")
        certificate_json = json.dumps(certificate, ensure_ascii=False, sort_keys=True, indent=2)
        certificate_b64 = base64.b64encode(certificate_json.encode("utf-8")).decode("ascii")
        validator = html.escape(str(certificate.get("validator_version", "not recorded")))
        details = (
            f'<dl class="pramaan-proof-card-hashes">{"".join(hash_rows)}'
            f'<div><dt>Validator version</dt><dd><code>{validator}</code></dd></div></dl>'
            f'<button type="button" class="pramaan-ui-button secondary" data-copy-certificate '
            f'data-certificate-b64="{certificate_b64}">Copy certificate JSON</button>'
            '<span class="pramaan-proof-card-copy-status" role="status" aria-live="polite"></span>'
        )
        content = (
            f'<p class="pramaan-proof-card-result">{title}</p>{reward_html}'
            f'<ul class="pramaan-proof-card-checks" aria-label="Passed checks">{"".join(labels)}</ul>'
            f'{details}{download_html}{action}'
        )
        heading = "Proof of Art"
        state_class = "is-proven"
    else:
        failed_count = len(visible_checks)
        content = (
            f'<p class="pramaan-proof-card-result">{failed_count} failed check(s)</p>'
            f'<ul class="pramaan-proof-card-checks" aria-label="Failed checks">{"".join(labels)}</ul>'
            f'<p class="pramaan-proof-card-witness">{html.escape(_counterexample_text(visible_checks))}</p>'
        )
        heading = "Not proven"
        state_class = "is-unproven"

    return (
        f'<section class="pramaan-proof-card {state_class}" data-proof-card aria-label="{heading}">'
        f'<div class="pramaan-proof-card-heading">{image}<div><p class="pramaan-ui-eyebrow">Proof of Art</p>'
        f'<h3>{heading}</h3>{context_html}</div></div>'
        f'{content}{exclusions_block}</section>'
    )
