"""Re-verify and render the saved first-screen Beam proof replay."""

import re
from pathlib import Path

from pramaan.app import PROOF_REPLAY_HTML_PATH, _render_proof_replay_html_from_data


def main() -> None:
    html = _render_proof_replay_html_from_data()
    if 'data-testid="proof-replay"' not in html or "Mask SHA-256" not in html:
        raise SystemExit("The saved trajectory did not produce a complete proof replay.")
    result_step = re.search(
        r'(<div class="pramaan-proof-step" data-proof-step="\d+" data-proof-result) hidden(>.*?</div>)',
        html,
        re.S,
    )
    if not result_step:
        raise SystemExit("The saved trajectory has no replayable passing result.")
    html = html[:result_step.start()] + result_step.group(1) + result_step.group(2) + html[result_step.end():]

    def set_step_visibility(match: re.Match[str]) -> str:
        tag = match.group(1)
        if "data-proof-result" in tag:
            tag = tag.replace(" hidden", "")
        elif " hidden" not in tag:
            tag += " hidden"
        return tag + ">"

    html = re.sub(r'(<div class="pramaan-proof-step" data-proof-step="\d+"[^>]*)>', set_step_visibility, html)
    result_body = re.search(
        r'<div class="pramaan-proof-step" data-proof-step="\d+" data-proof-result[^>]*>(.*?)</div>',
        html,
        re.S,
    )
    passing_image = re.search(
        r'<img class="pramaan-proof-tile" src="([^"]+)" alt="([^"]+)" width="384" height="384" decoding="async">',
        result_body.group(1) if result_body else "",
    )
    if not passing_image:
        raise SystemExit("The passing result has no renderable tile image.")
    image_src, image_alt = passing_image.groups()
    html = re.sub(
        r'(<img class="pramaan-proof-tile" data-proof-active-tile src=")[^"]+(" alt=")[^"]+(" width="384" height="384" decoding="async">)',
        lambda match: match.group(1) + image_src + match.group(2) + image_alt + match.group(3),
        html,
        count=1,
    )
    context = re.search(r'<p class="pramaan-proof-context">(.*?)</p>', html)
    state = context.group(1).split(" · ", 1)[0] if context else ""
    html = re.sub(
        r'(<figcaption data-proof-figure-caption>).*?(</figcaption>)',
        lambda match: match.group(1) + "Beam repair · all checks pass · " + state + match.group(2),
        html,
        count=1,
    )
    html = re.sub(
        r'(<span class="pramaan-proof-progress" data-proof-progress role="status">).*?(</span>)',
        r'\1Verified result · replay ready\2',
        html,
        count=1,
    )
    PROOF_REPLAY_HTML_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROOF_REPLAY_HTML_PATH.write_text(html + "\n", encoding="utf-8")
    print(f"Replayed and rendered the proof hero to {Path(PROOF_REPLAY_HTML_PATH).as_posix()}")


if __name__ == "__main__":
    main()
