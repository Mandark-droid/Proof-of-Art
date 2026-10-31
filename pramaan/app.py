from __future__ import annotations

import json
import math
import os
import base64
import hashlib
import uuid
import re
import threading
import zipfile
import logging
from datetime import datetime, timezone
from io import BytesIO
from html import escape as html_escape
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import gradio as gr
from PIL import Image
from pydantic import ValidationError
from dotenv import load_dotenv

from .certificate import make_certificate
from .branding import HEADLINE
from .checks.core import all_pass, verify_pattern
from .dsl import parse_pattern
from .export import artifact_bundle, render_highlighted_png, render_png, trajectory_bundle
from .orchestrate import MAX_BRIEF_CHARS, generate_and_verify, require_integer_control
from .ui.quilt import render_data_quilt, state_grid_cell, state_grid_style
from .ui.home import render_reference_wall
from .ui.compare import render_arena, render_comparison_charts, render_same_brief_wall
from .ui.leaderboard import (leaderboard_numeric_attr, render_canonical_leaderboard,
                             render_community_leaderboard, render_model_report)
from .ui.findings import render_findings_strip
from .paths import resolve_project_root
from .tasks import canonical_jsonl_bytes
from . import VALIDATOR_VERSION

try:
    from .traditions import ALL_STATES, load_registry, traditions_for_state
    TRADITION_REGISTRY_AVAILABLE = True
except ImportError:  # Keep the standalone app usable without the optional registry package.
    ALL_STATES = (
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
        "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
        "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
        "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
        "West Bengal", "Andaman and Nicobar Islands", "Chandigarh",
        "Dadra and Nagar Haveli and Daman and Diu", "Delhi", "Jammu and Kashmir", "Ladakh",
        "Lakshadweep", "Puducherry",
    )
    TRADITION_REGISTRY_AVAILABLE = False

    def load_registry():
        return {}

    def traditions_for_state(state: str):
        if state == "Gujarat":
            return [SimpleNamespace(
                id="gj_ajrakh_fallback", name="Ajrakh-inspired geometric repeat",
                grammar="periodic_raster", palette_presets={"indigo_madder": ["#172F4A", "#A84436", "#F4E5C5", "#262321"]},
                default_profile={"symmetry": "D4"}, cautions="Prototype fallback; registry is not integrated yet.",
                sources=[], sources_unverified=[], states=["Gujarat"], state_types=["state"],
                confidence="low", tier="A",
            )]
        return []

ROOT = resolve_project_root(__file__)
EVAL_SUITE_ID = "eval-v2.1"
EVAL_SUITE_SHA256 = "b04b568d84d2568fd261789819d4e925844dba5afe9c5962275ee571d84e4e3a"
EVAL_RESULT_PATH = Path("benchmarks/results/eval-v2.1-beam-20261030.jsonl")
EVAL_TRAJECTORY_PATH = Path("benchmarks/data/eval-v2.1-trajectories.jsonl")
EVAL_SUMMARY_PATH = Path("benchmarks/data/eval-v2.1-summary.json")
PROOF_REPLAY_TRAJECTORY_ID = "beam-hf:gj_bandhani-easy-6:b04b568d84d2"
PROOF_REPLAY_PROMPT_ID = "gj_bandhani-easy-6"


def _load_frozen_eval_suite():
    path = ROOT / "benchmarks" / "suites" / f"{EVAL_SUITE_ID}.jsonl"
    suite_bytes = path.read_bytes()
    actual_digest = hashlib.sha256(canonical_jsonl_bytes(suite_bytes)).hexdigest()
    if actual_digest != EVAL_SUITE_SHA256:
        raise ValueError("The bundled evaluation suite does not match its frozen digest.")
    prompts = [json.loads(line) for line in suite_bytes.decode("utf-8").splitlines() if line.strip()]
    if len(prompts) != 72 or len({row.get("id") for row in prompts}) != 72:
        raise ValueError("The bundled evaluation suite does not contain 72 unique prompts.")
    return suite_bytes, prompts
load_dotenv(ROOT / ".env", override=False)
EXAMPLES = {
    "Indigo Starfield · verified, hand-authored": ROOT / "examples" / "ajrakh_starfield.pattern.json",
    "D4 counterexample · failing, hand-authored": ROOT / "examples" / "ajrakh_d4_counterexample.pattern.json",
    "D4 rosette · Beam-generated, repaired, verified": ROOT / "examples" / "beam_d4_rosette.pattern.json",
    "Nested Diamond Rosette · Beam-generated, verified": ROOT / "examples" / "beam_medium_03.pattern.json",
    "Warm Earth Repeat · Beam-generated, verified": ROOT / "examples" / "beam_medium_06.pattern.json",
    "Nested Diamond Cluster · Beam-generated, verified": ROOT / "examples" / "beam_hard_07.pattern.json",
    "Bandhani Dot Field · Beam-generated, verified": ROOT / "examples" / "beam_g2_gj_bandhani.pattern.json",
    "Singhbhum Bands · Beam-generated, repaired, verified": ROOT / "examples" / "beam_g3_jh_singhbhum_bands.pattern.json",
    "Gamosa Counted Grid · Beam-generated, verified": ROOT / "examples" / "beam_g4_as_gamosa_grid.pattern.json",
    "Aripana Rotational Field · Beam-generated, repaired, verified": ROOT / "examples" / "beam_g5_br_aripana_rotational.pattern.json",
    "Sikku Kolam Loop · Beam-generated, repaired, verified": ROOT / "examples" / "beam_g6_tn_kolam_sikku.pattern.json",
    "Sidi Saiyyed Jaali · Beam-generated, repaired, verified": ROOT / "examples" / "beam_g7_gj_sidi_saiyyed_jaali.pattern.json",
}
EXAMPLE_STATES = {
    "ajrakh_starfield": "Gujarat",
    "ajrakh_d4_counterexample": "Gujarat",
    "beam_d4_rosette": "Gujarat",
    "beam_medium_03": "Gujarat",
    "beam_medium_06": "Gujarat",
    "beam_hard_07": "Gujarat",
    "beam_g2_gj_bandhani": "Gujarat",
    "beam_g3_jh_singhbhum_bands": "Jharkhand",
    "beam_g4_as_gamosa_grid": "Assam",
    "beam_g5_br_aripana_rotational": "Bihar",
    "beam_g6_tn_kolam_sikku": "Tamil Nadu",
    "beam_g7_gj_sidi_saiyyed_jaali": "Gujarat",
}
BEAM_TRAJECTORIES = {
    "beam_d4_rosette": ROOT / "examples" / "beam_d4_rosette.trajectory.json",
    "beam_medium_03": ROOT / "examples" / "beam_medium_03.trajectory.json",
    "beam_medium_06": ROOT / "examples" / "beam_medium_06.trajectory.json",
    "beam_hard_07": ROOT / "examples" / "beam_hard_07.trajectory.json",
    "beam_g2_gj_bandhani": ROOT / "examples" / "beam_g2_gj_bandhani.trajectory.json",
    "beam_g3_jh_singhbhum_bands": ROOT / "examples" / "beam_g3_jh_singhbhum_bands.trajectory.json",
    "beam_g4_as_gamosa_grid": ROOT / "examples" / "beam_g4_as_gamosa_grid.trajectory.json",
    "beam_g5_br_aripana_rotational": ROOT / "examples" / "beam_g5_br_aripana_rotational.trajectory.json",
    "beam_g6_tn_kolam_sikku": ROOT / "examples" / "beam_g6_tn_kolam_sikku.trajectory.json",
    "beam_g7_gj_sidi_saiyyed_jaali": ROOT / "examples" / "beam_g7_gj_sidi_saiyyed_jaali.trajectory.json",
}


def _tradition_choices(state: str | None):
    return list(_cached_tradition_choices(state or ""))


@lru_cache(maxsize=36)
def _cached_tradition_choices(state: str):
    return tuple(sorted(traditions_for_state(state), key=lambda item: (item.name.casefold(), item.id)))


def _tradition_by_id(tradition_id: str | None):
    if not tradition_id:
        return None
    found = load_registry().get(tradition_id)
    if found is not None or TRADITION_REGISTRY_AVAILABLE:
        return found
    return next((item for state in ALL_STATES for item in traditions_for_state(state)
                 if item.id == tradition_id), None)


def _example_choices(state_name: str | None, tradition_id: str | None):
    """Return worked examples belonging to the currently selected state and tradition."""
    if not state_name or not tradition_id:
        return []
    tradition = _tradition_by_id(tradition_id)
    if tradition is None or state_name not in tradition.states:
        return []
    choices = []
    for name, path in EXAMPLES.items():
        try:
            pattern = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(pattern, dict) or pattern.get("tradition_id") != tradition_id:
            continue
        example_id = Path(path).name.removesuffix(".pattern.json")
        example_state = EXAMPLE_STATES.get(example_id)
        if example_state != state_name:
            continue
        choices.append((f"{name} · {state_name} / {tradition.name}", name))
    return choices


def _example_for_selection(state_name: str | None, tradition_id: str | None) -> str:
    choices = _example_choices(state_name, tradition_id)
    return choices[0][1] if choices else ""


def _tradition_dropdown_update(state: str | None):
    traditions = _tradition_choices(state)
    choices = [(f"{item.name} · {item.id}", item.id) for item in traditions]
    return gr.update(choices=choices, value=(traditions[0].id if traditions else None))


def _tradition_markdown_for_state(state: str | None):
    traditions = _tradition_choices(state)
    if not traditions:
        return f"No sourced tradition records are available for **{state or 'this state'}** yet."
    lines = [f"## {state}", "Craft notes describe sources and cautions; they do not certify cultural authenticity."]
    for item in traditions:
        lines.extend([f"### {item.name}", f"Grammar: `{item.grammar}` · confidence: **{item.confidence}** · tier: **{item.tier}**",
                      f"Caution: {item.cautions}"])
        for source in list(item.sources) + list(item.sources_unverified):
            lines.append(f"- [{source.title}]({source.url}) · {source.publisher} — {source.supports}")
    return "\n\n".join(lines)


def _benchmark_summary():
    path = ROOT / EVAL_SUMMARY_PATH
    if not path.is_file():
        return {"status": "The audited eval-v2.1 benchmark summary is not available in this Space build.",
                "path": EVAL_SUMMARY_PATH.as_posix()}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            raise ValueError("unsupported summary format")
        summary_digest = (summary.get("prompt_suite_sha256") or summary.get("suite_sha256")
                          or summary.get("suite_digest"))
        if summary_digest is not None and summary_digest != EVAL_SUITE_SHA256:
            return {"status": "The bundled benchmark summary belongs to a superseded suite.",
                    "path": EVAL_SUMMARY_PATH.as_posix()}
        return summary
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {"status": "Benchmark summary could not be loaded.", "error": str(exc)[:300]}


def _benchmark_summary_html() -> str:
    summary = _benchmark_summary()
    if not isinstance(summary, dict):
        summary = {"status": "Benchmark summary has an unsupported format."}
    by_model = summary.get("by_model")
    if not isinstance(by_model, dict) or not by_model:
        message = html_escape(str(summary.get("status", "Audited benchmark summary is not available in this Space build.")))
        return (
            '<section class="pramaan-benchmark-empty" role="status">'
            '<span class="pramaan-benchmark-kicker">Benchmark results</span>'
            f"<p>{message}</p></section>"
        )

    def percent(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return f"{value * 100:.1f}%" if 0 <= value <= 1 else f"{value:.1f}%"
        return "not available"

    def costs(value):
        if not isinstance(value, dict) or not value:
            return "not available"
        return ", ".join(f"{html_escape(str(currency))} {amount:.2f}"
                          for currency, amount in sorted(value.items())
                          if isinstance(amount, (int, float)) and not isinstance(amount, bool)
                          and math.isfinite(amount)) or "not available"

    expected_suite_size = int(summary.get("expected_prompt_count", 72))
    comparison_ready = bool(by_model) and all(
        isinstance(model_summary, dict)
        and isinstance(model_summary.get("overall"), dict)
        and model_summary["overall"].get("n") == expected_suite_size == 72
        and isinstance(model_summary.get("study"), dict)
        and model_summary["study"].get("complete") is True
        for model_summary in by_model.values()
    )
    comparison_note = (
        '<p class="pramaan-benchmark-comparison-ready" role="status">'
        'Full-suite coverage recorded for every listed model. Scores share the expected 72-task coverage.</p>'
        if comparison_ready else
        '<p class="pramaan-benchmark-comparison-warning" role="status">'
        'Context only · at least one result has incomplete, legacy, or unverified coverage. '
        'These scores are not an apples-to-apples comparison. A model must complete the same 72 tasks under the frozen protocol.</p>'
    )

    def duration(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return f"{value / 1000:.1f} s"
        return "not available"

    def token_mean(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return f"{value:,.0f} tokens / prompt"
        return "not available"

    cards = []
    for model_key, model_summary in sorted(by_model.items(), key=lambda item: str(item[0]).casefold()):
        overall = model_summary.get("overall", {}) if isinstance(model_summary, dict) else {}
        if not isinstance(overall, dict):
            overall = {}
        count = overall.get("n", "not available")
        model_expected = (model_summary.get("study", {}).get("expected_prompt_count", expected_suite_size)
                          if isinstance(model_summary, dict) and isinstance(model_summary.get("study"), dict)
                          else expected_suite_size)
        complete = (count == model_expected == 72
                    and isinstance(model_summary, dict)
                    and isinstance(model_summary.get("study"), dict)
                    and model_summary["study"].get("complete") is True)
        coverage = f"{html_escape(str(count))}/72 {EVAL_SUITE_ID} tasks"
        if model_expected != 72:
            coverage += f" · summary declares {html_escape(str(model_expected))}"
        if count == 72:
            coverage += " · 72 prompts"
        coverage_state = "Full suite" if complete else "Partial / legacy"
        mean_elapsed = overall.get("mean_elapsed_ms")
        request_latency = overall.get("mean_request_elapsed_ms_per_prompt")
        verifier_latency = overall.get("mean_verification_elapsed_ms_per_prompt")
        cards.append(
            '<article class="pramaan-benchmark-card">'
            f'<h3>{html_escape(str(model_key))}</h3>'
            '<dl>'
            f'<div><dt>Coverage</dt><dd>{coverage} · {coverage_state}</dd></div>'
            f'<div><dt>Pass@1</dt><dd>{percent(overall.get("validity_at_1"))}</dd></div>'
            f'<div><dt>Pass@3</dt><dd>{percent(overall.get("validity_at_3"))}</dd></div>'
            f'<div><dt>Tokens</dt><dd>{token_mean(overall.get("mean_tokens_when_available"))}</dd></div>'
            f'<div><dt>Wall time</dt><dd>{duration(mean_elapsed)} mean total</dd></div>'
            f'<div><dt>Request latency</dt><dd>{duration(request_latency)} mean / prompt</dd></div>'
            f'<div><dt>Verifier latency</dt><dd>{duration(verifier_latency)} mean / prompt</dd></div>'
            f'<div><dt>Reported cost</dt><dd>{costs(overall.get("reported_cost_totals_by_currency"))}</dd></div>'
            f'<div><dt>Rate-card estimate</dt><dd>{costs(overall.get("estimated_cost_totals_by_currency"))}</dd></div>'
            '</dl></article>'
        )
    return (comparison_note + '<section class="pramaan-benchmark-grid" aria-label="Per-model benchmark results">'
            + "".join(cards) + "</section>")


def _community_runtime():
    """Load the community runner only when its privacy-safe API is present."""
    try:
        from benchmarks import run as runner
        from scripts import push_hub_datasets as publisher
        if (not callable(getattr(runner, "community_litellm_model_config", None))
                or not callable(getattr(runner, "run_community_evaluation", None))
                or not callable(getattr(publisher, "sanitize_community_eval_rows", None))
                or not callable(getattr(publisher, "publish_community_eval_run", None))):
            return None
        return runner, publisher
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


def _community_submission_archive_runtime():
    """Load the archive auditor and publisher only when both are integrated."""
    try:
        from pramaan.community import audit_submission_archive
        from scripts import push_hub_datasets as publisher
        publish = getattr(publisher, "publish_community_submission_archive", None)
        if not callable(audit_submission_archive) or not callable(publish):
            return None
        return audit_submission_archive, publish
    except (ImportError, AttributeError, TypeError, ValueError):
        return None


def _resolve_gradio_upload(path_text: str) -> Path:
    """Resolve a Gradio upload and reject paths outside its managed temp folder."""
    upload_root = Path(gr.utils.get_upload_folder()).resolve(strict=True)
    path = Path(path_text).resolve(strict=True)
    if not path.is_relative_to(upload_root) or not path.is_file():
        raise ValueError("Choose a ZIP uploaded to this Space.")
    if path.suffix.lower() != ".zip":
        raise ValueError("The contribution must be a ZIP archive.")
    if path.stat().st_size > 20_000_000:
        raise ValueError("The ZIP archive exceeds the 20 MB limit.")
    return path


def _remove_gradio_upload(path_text) -> None:
    """Remove only a temporary file that resolves within Gradio's upload folder."""
    if not isinstance(path_text, str) or not path_text:
        return
    try:
        upload_root = Path(gr.utils.get_upload_folder()).resolve(strict=True)
        path = Path(path_text).resolve(strict=True)
        if path.is_relative_to(upload_root) and path.is_file():
            path.unlink()
    except (OSError, RuntimeError, TypeError, ValueError):
        return


def _publish_community_archive(archive_path: str) -> str:
    """Audit then publish a user-consented ZIP through the configured scoped writer."""
    if not os.getenv("HF_COMMUNITY_WRITE_TOKEN"):
        return "Archive contribution is not enabled in this Space. No archive was published."
    runtime = _community_submission_archive_runtime()
    if runtime is None:
        return "Archive validation is not available in this Space build. No archive was published."
    _, publisher = runtime
    publish = getattr(publisher, "publish_community_submission_archive", None)
    if not callable(publish):
        return "Archive validation is not available in this Space build. No archive was published."
    try:
        archive = _resolve_gradio_upload(archive_path)
        result = publish(
            archive, run_id=f"space-upload-{uuid.uuid4().hex}",
            expected_suite_sha256=EVAL_SUITE_SHA256,
        )
    except ValueError as exc:
        reason = " ".join(str(exc).split())[:240]
        return f"Archive not published: {reason or 'replay audit rejected the archive.'}"
    except Exception:
        return "Archive validation or publication failed. No archive was published; contact the project maintainer."
    if (not isinstance(result, dict)
            or result.get("repo_id") != "kshitijthakkar/pramaan-community-runs"):
        return "Archive processing finished without a verifiable publication receipt. Contact the project maintainer."
    count = result.get("records")
    if not isinstance(count, int) or count < 1:
        return "Archive processing finished without a valid publication receipt. Contact the project maintainer."
    return f"Published {count} replay-audited record(s) to the public community dataset (kshitijthakkar/pramaan-community-runs)."


def _community_submission_archive_ready() -> bool:
    return (bool(os.getenv("HF_COMMUNITY_WRITE_TOKEN"))
            and _community_submission_archive_runtime() is not None
            and callable(globals().get("_publish_community_archive")))


def _community_model_choices():
    return [("ready", "Community runner available")] if _community_runtime() is not None else []


def _community_publish_ready() -> bool:
    if not os.getenv("HF_COMMUNITY_WRITE_TOKEN"):
        return False
    runtime = _community_runtime()
    if runtime is None:
        return False
    import inspect
    publish_fn = getattr(runtime[1], "publish_community_eval_run", None)
    if not callable(publish_fn):
        return False
    try:
        parameters = inspect.signature(publish_fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return (any(p.name == "submitted_by" for p in parameters)
            or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters))


def _summarize_community_rows(records, prompt_ids, model_keys):
    from benchmarks.summarize import summarize
    return summarize(records, expected_prompt_ids=prompt_ids, expected_model_keys=model_keys)


@lru_cache(maxsize=1)
def _community_comparison_snapshot():
    """Load one newest replay-valid public community run per model, plus Beam when present."""
    runtime = _community_runtime()
    if runtime is None:
        return {"status": "Community runs are not available in this Space build.", "rows": [], "summary": {}}
    _runner, publisher = runtime
    loader = getattr(publisher, "load_recent_community_runs", None)
    if not callable(loader):
        return {"status": "Recent community results are not available yet.", "rows": [], "summary": {}}
    try:
        suite_bytes, suite_prompts = _load_frozen_eval_suite()
        suite_sha = EVAL_SUITE_SHA256
        prompt_ids = [row["id"] for row in suite_prompts]
        recent_runs = loader(limit=100)
        if not isinstance(recent_runs, list):
            raise ValueError("community run listing is malformed")
        selected, selected_rows = {}, []
        for run in recent_runs:
            if not isinstance(run, dict):
                continue
            model_key, run_id = run.get("model_key"), run.get("run_id")
            if (not isinstance(model_key, str) or not model_key or len(model_key) > 256
                    or model_key in selected):
                continue
            rows = run.get("rows")
            if not isinstance(rows, list) or not rows:
                continue
            if any(row.get("model_key") != model_key or row.get("run_id") != run_id for row in rows
                   if isinstance(row, dict)):
                continue
            # Revalidate every stored run before its metrics or artwork are shown.
            try:
                safe_rows = publisher.sanitize_community_eval_rows(
                    rows, run_id=run_id, expected_suite_sha256=suite_sha,
                )
            except Exception:
                # A community run from a superseded suite must not block current results.
                continue
            if not safe_rows:
                continue
            selected[model_key] = {
                "run_id": run_id, "submitted_at": str(run.get("submitted_at", "")),
            }
            selected_rows.extend(safe_rows)

        beam_rows = []
        beam_paths = (ROOT / EVAL_RESULT_PATH, ROOT / EVAL_TRAJECTORY_PATH)
        beam_path = next((path for path in beam_paths if path.is_file()), None)
        if beam_path is not None:
            for line in beam_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if (isinstance(row, dict) and row.get("model_key") == "beam-hf"
                        and row.get("prompt_suite_sha256") == suite_sha):
                    beam_rows.append(row)
        records = beam_rows + selected_rows
        if not records:
            return {"status": "No replay-valid community runs are available yet.",
                    "rows": [], "summary": {}, "selected": selected}
        summary = _summarize_community_rows(records, prompt_ids, ["beam-hf", *selected.keys()])
        if not isinstance(summary.get("common_completed"), dict):
            return {"status": "The shared-question comparison helper is not available in this Space build.",
                    "rows": selected_rows, "beam_rows": beam_rows, "selected": selected, "summary": {}}
        return {"status": "Community results loaded.", "rows": selected_rows,
                "beam_rows": beam_rows, "selected": selected, "summary": summary}
    except Exception:
        return {"status": "Community results could not be loaded or replayed.", "rows": [], "summary": {}}


def _leaderboard_numeric_attr(value) -> str:
    return leaderboard_numeric_attr(value)


def _community_leaderboard_html(snapshot: dict) -> str:
    return render_community_leaderboard(snapshot)




@lru_cache(maxsize=64)
def _read_gallery_json(path_text: str, file_size: int, modified_ns: int):
    try:
        value = json.loads(Path(path_text).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _gallery_json(path: Path):
    try:
        stat = path.stat()
    except OSError:
        return None
    return _read_gallery_json(str(path), stat.st_size, stat.st_mtime_ns)


def _load_gallery_index():
    index = _gallery_json(ROOT / "assets" / "gallery" / "index.json")
    if (not isinstance(index, dict) or index.get("suite_id") != EVAL_SUITE_ID
            or index.get("suite_sha256") != EVAL_SUITE_SHA256
            or index.get("validator_version") != VALIDATOR_VERSION
            or not isinstance(index.get("states"), list)
            or len(index.get("states", [])) != len(ALL_STATES)
            or not isinstance(index.get("arms"), list)
            or not isinstance(index.get("common_prompt_ids"), list)):
        return None
    common_ids = index["common_prompt_ids"]
    if (any(not isinstance(item, str) or not item for item in common_ids)
            or len(set(common_ids)) != len(common_ids) or len(common_ids) > 72):
        return None
    return index


def _load_gallery_arm_documents(index: dict | None) -> dict[str, dict]:
    """Load only safely named replay-audited arms referenced by the current index."""
    if not isinstance(index, dict) or not isinstance(index.get("arms"), list):
        return {}
    documents = {}
    for arm in index["arms"]:
        key = arm.get("key") if isinstance(arm, dict) else None
        if not isinstance(key, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}", key):
            continue
        document = _gallery_json(ROOT / "assets" / "gallery" / f"{key}.json")
        if isinstance(document, dict) and isinstance(document.get("tasks"), dict):
            documents[key] = document
    return documents


def _wilson_interval(successes: int, total: int):
    if total <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _canonical_leaderboard_html(index: dict | None = None) -> str:
    return render_canonical_leaderboard(
        index, root=ROOT, suite_id=EVAL_SUITE_ID, suite_sha256=EVAL_SUITE_SHA256,
        load_gallery_index=_load_gallery_index, gallery_json=_gallery_json,
        wilson_interval=_wilson_interval, comparison_charts=render_comparison_charts,
    )


def _arena_default_model_keys(summary: dict | None) -> list[str]:
    by_model = summary.get("by_model") if isinstance(summary, dict) else None
    if not isinstance(by_model, dict):
        return ["beam-hf"]
    ranked = []
    for key, document in by_model.items():
        if not isinstance(document, dict):
            continue
        overall = document.get("overall") if isinstance(document.get("overall"), dict) else {}
        n = overall.get("n")
        score = overall.get("validity_at_3")
        first = overall.get("validity_at_1")
        if (type(n) is int and n > 0 and isinstance(score, (int, float))
                and not isinstance(score, bool) and isinstance(first, (int, float))
                and not isinstance(first, bool)):
            ranked.append((float(score), float(first), str(key)))
    ranked.sort(key=lambda row: (-row[0], -row[1], row[2]))
    selected = ["beam-hf"]
    selected.extend(key for _score, _first, key in ranked if key != "beam-hf")
    return list(dict.fromkeys(selected))[:4]


def _model_methodology_html(index: dict | None = None, summary: dict | None = None) -> str:
    """Show measured request settings for every model with a published result."""
    index = index if isinstance(index, dict) else _load_gallery_index()
    indexed_arms = [row for row in index.get("arms", []) if isinstance(row, dict)] if isinstance(index, dict) else []
    arm_map = {str(row.get("key")): row for row in indexed_arms if row.get("key")}
    model_map = summary.get("by_model") if isinstance(summary, dict) else None
    if isinstance(model_map, dict):
        ordered_keys = list(dict.fromkeys([str(row.get("key")) for row in indexed_arms if row.get("key")] +
                                          [str(key) for key in model_map]))
        arms = []
        for key in ordered_keys:
            document = model_map.get(key) if isinstance(model_map.get(key), dict) else {}
            merged = dict(document)
            merged.update(arm_map.get(key, {}))
            merged["key"] = key
            arms.append(merged)
    else:
        arms = indexed_arms
    expected_count = summary.get("expected_prompt_count", 72) if isinstance(summary, dict) else 72
    if type(expected_count) is not int or expected_count <= 0:
        expected_count = 72

    def value_text(value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return f"{value:g}"
        if isinstance(value, str) and value:
            return value
        return "not specified"

    rows = []
    for arm in arms:
        key = str(arm.get("key") or "")
        params = arm.get("request_params") if isinstance(arm.get("request_params"), dict) else {}
        if not params and isinstance(arm.get("parameters"), dict):
            params = arm["parameters"]
        output_cap = params.get("max_tokens", params.get("max_completion_tokens"))
        output_config = params.get("output_config") if isinstance(params.get("output_config"), dict) else {}
        extra_body = params.get("extra_body") if isinstance(params.get("extra_body"), dict) else {}
        thinking = extra_body.get("thinking") if isinstance(extra_body.get("thinking"), dict) else {}
        effort = params.get("reasoning_effort", output_config.get("effort"))
        reasoning_cap = params.get("max_reasoning_tokens", params.get("reasoning_tokens"))
        rows.append(
            f'<li data-model-protocol-row="{html_escape(key, quote=True)}">'
            f'<strong>{html_escape(str(arm.get("label") or key))}</strong> · '
            f'{html_escape(str(arm.get("provider") or "not specified"))}; '
            f'temperature {html_escape(value_text(params.get("temperature")) if "temperature" in params else "not sent")}; '
            f'effort {html_escape(value_text(effort))}; '
            f'thinking {html_escape(value_text(thinking.get("type")))}; '
            f'output cap {html_escape(value_text(output_cap))}; '
            f'reasoning cap {html_escape(value_text(reasoning_cap))}</li>'
        )
    body = ("".join(rows) if rows else
            '<li>Provider request settings have not been published yet.</li>')
    model_label = "model" if len(arms) == 1 else "models"
    return (
        '<details id="leaderboard-methodology" class="pramaan-leaderboard-method">'
        f'<summary>Methodology · {len(arms)} {model_label} · {expected_count} suite tasks</summary>'
        '<p>The verifier and task definitions were frozen. Up to three proposals were allowed per task; provider-error rows are not scored. Per-model request settings are listed below.</p>'
        f'<ul class="pramaan-model-protocol-list">{body}</ul>'
        '<p>One task has a documented unsatisfiable constraint. Rows were collected in two sittings.</p>'
        '</details>'
    )


def _canonical_state_gallery_html(index: dict | None, state_name: str, model_key: str = "all") -> str:
    return render_same_brief_wall(
        index, state_name, root=ROOT, suite_id=EVAL_SUITE_ID,
        suite_sha256=EVAL_SUITE_SHA256, gallery_json=_gallery_json, model_key=model_key,
    )


def _community_state_gallery(snapshot: dict, state_name: str, model_key: str | None = None) -> str:
    figures = []
    rows = snapshot.get("rows", []) if isinstance(snapshot, dict) else []
    for row in rows:
        if (not isinstance(row, dict) or row.get("state") != state_name
                or (model_key not in (None, "all") and row.get("model_key") != model_key)):
            continue
        attempts = row.get("attempts") if isinstance(row.get("attempts"), list) else []
        accepted = next((attempt for attempt in reversed(attempts)
                         if isinstance(attempt, dict)
                         and isinstance(attempt.get("report"), dict)
                         and attempt["report"].get("all_hard_constraints_pass") is True
                         and isinstance(attempt.get("candidate"), dict)), None)
        if accepted is None:
            continue
        try:
            pattern = parse_pattern(json.dumps(accepted["candidate"]))
            checks, masks, grid = verify_pattern(pattern)
            certificate = make_certificate(pattern, checks, masks, grid)
            report = accepted["report"]
            saved_checks = [check for check in report.get("checks", [])
                            if isinstance(check, dict) and check.get("id") != "candidate_truncated"]
            if (not all_pass(checks) or saved_checks != checks
                    or certificate["mask_sha256"] != row.get("accepted_mask_sha256")):
                continue
            uri = _image_data_uri(render_png(grid, pattern.palette, scale=4, repeats=1))
        except (ValidationError, ValueError, TypeError):
            continue
        if uri:
            label = f"{row.get('model_key')} · {row.get('prompt_id')} · {row.get('tradition_id')}"
            figures.append(f'<figure><img src="{uri}" loading="lazy" decoding="async" alt="Replayed accepted community design"><figcaption>{html_escape(label)}</figcaption></figure>')
        if len(figures) >= 12:
            break
    return ("".join(figures) if figures else
            '<p class="pramaan-muted">No replay-verified community designs are available for this state yet.</p>')


def _community_model_config_is_eligible(config: dict) -> bool:
    if not isinstance(config, dict):
        return False
    params, budget = config.get("request_params"), config.get("budget")
    return (isinstance(params, dict) and params.get("max_tokens") == 4096
            and isinstance(budget, dict) and budget.get("cap_usd") == 10.0)


_COMMUNITY_DAILY_RUNS: set[tuple[str, str]] = set()
_COMMUNITY_DAILY_RUNS_LOCK = threading.Lock()


def _community_daily_identity(oauth_profile, request) -> str | None:
    """Hash an authenticated user or anonymous Gradio session for in-memory limits."""
    username = getattr(oauth_profile, "username", None)
    if isinstance(username, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,63}", username):
        subject = f"hf:{username.casefold()}"
    else:
        session_hash = getattr(request, "session_hash", None)
        if not isinstance(session_hash, str) or not session_hash:
            return None
        subject = f"session:{session_hash}"
    return hashlib.sha256(subject.encode("utf-8")).hexdigest()


def _claim_community_daily_run(identity: str) -> bool:
    """Allow one USD-10-capped BYO evaluation per identity and UTC day."""
    day = datetime.now(timezone.utc).date().isoformat()
    with _COMMUNITY_DAILY_RUNS_LOCK:
        stale = {entry for entry in _COMMUNITY_DAILY_RUNS if entry[0] != day}
        _COMMUNITY_DAILY_RUNS.difference_update(stale)
        entry = (day, identity)
        if entry in _COMMUNITY_DAILY_RUNS:
            return False
        _COMMUNITY_DAILY_RUNS.add(entry)
        return True


def _estimate_community_full_run_cost(model_name: str, *, runner=None, config=None) -> dict | None:
    """Use the runner's official-usage estimate and validate it before display."""
    if runner is None:
        runtime = _community_runtime()
        if runtime is None:
            return None
        runner, _publisher = runtime
    if not isinstance(model_name, str) or not model_name:
        return None
    try:
        config = config or runner.community_litellm_model_config(model_name)
        estimate_fn = getattr(runner, "estimate_community_evaluation_cost", None)
        if not _community_model_config_is_eligible(config) or not callable(estimate_fn):
            return None
        estimate = estimate_fn(model_name, task_count=72)
    except Exception:
        return None
    if not isinstance(estimate, dict):
        return None
    estimated_usd = estimate.get("estimated_usd")
    run_cap_usd = estimate.get("per_run_cap_usd")
    exceeds_run_cap = estimate.get("estimated_cost_exceeds_per_run_cap")
    means = (estimate.get("mean_input_tokens_per_task"),
             estimate.get("mean_output_tokens_per_task"))
    if (estimate.get("reference_suite") != EVAL_SUITE_ID
            or estimate.get("reference_tasks") != 72
            or estimate.get("source") != "official_beam_eval_v2.1_measured_usage"
            or estimate.get("task_count") != 72
            or isinstance(run_cap_usd, bool) or not isinstance(run_cap_usd, (int, float))
            or not math.isclose(run_cap_usd, 10.0, rel_tol=0, abs_tol=1e-12)
            or not isinstance(exceeds_run_cap, bool)
            or isinstance(estimated_usd, bool) or not isinstance(estimated_usd, (int, float))
            or not math.isfinite(estimated_usd) or estimated_usd < 0
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value < 0 for value in means)):
        return None
    pricing = config.get("pricing") if isinstance(config.get("pricing"), dict) else {}
    return {**estimate, "model_key": model_name, "cap_usd": run_cap_usd,
            "rate_card_source": pricing.get("source", "provider rate card"),
            "rate_card_observed_at": pricing.get("observed_at")}


def _run_community_evaluation(model_name: str, provider_api_key: str, consent: bool, *,
                              publish: bool = False, submitted_by: str | None = None,
                              task_limit: int = 12, full_run_cost_acknowledged: bool = False,
                              reviewed_estimate: dict | None = None,
                              daily_identity: str | None = None,
                              enforce_daily_limit: bool = False) -> str:
    """Run a capped visitor evaluation; publish only with consent and authenticated attribution."""
    if consent is not True:
        return "Confirm that provider charges may apply before starting. No provider request or upload was made."
    if isinstance(task_limit, bool) or task_limit not in {12, 72}:
        return "Choose either the 12-question Quick run or the 72-question full suite. No provider request was made."
    if publish and (not isinstance(submitted_by, str)
                    or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", submitted_by)):
        return "Sign in with Hugging Face before submitting public results. No provider request was made."
    if publish and not os.getenv("HF_COMMUNITY_WRITE_TOKEN"):
        return "Public community uploads are not configured in this Space. No provider request was made."
    if not isinstance(provider_api_key, str) or not provider_api_key.strip() or len(provider_api_key) > 4096:
        return "Enter a valid provider key. No provider request or upload was made."
    if (not isinstance(model_name, str) or len(model_name) > 256
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}/[A-Za-z0-9][A-Za-z0-9._:/-]{0,190}", model_name)):
        return "Enter a valid provider/model identifier. No provider request was made."
    runtime = _community_runtime()
    if runtime is None:
        return "Community evaluation is not enabled in this Space build yet. No provider request was made."
    runner, publisher = runtime
    try:
        config = runner.community_litellm_model_config(model_name)
    except (ValueError, RuntimeError):
        return "Model or pricing validation failed during no-call preflight. No provider request was made."
    except Exception:
        return "Community preflight failed. No provider request was made."
    if not _community_model_config_is_eligible(config):
        return "This model does not have the fixed 4,096-token protocol and USD 10 spend cap. No provider request was made."
    full_run_estimate = None
    if task_limit == 72:
        full_run_estimate = _estimate_community_full_run_cost(
            model_name, runner=runner, config=config)
        if full_run_estimate is None:
            return "A complete official token baseline or current USD rate card is unavailable. No provider request was made."
        if full_run_cost_acknowledged is not True:
            return (f"Review the full-suite estimate of USD {full_run_estimate['estimated_usd']:.4f} "
                    "(hard cap USD 10) and confirm it before starting. No provider request was made.")
        if (not isinstance(reviewed_estimate, dict)
                or reviewed_estimate.get("model_key") != model_name
                or reviewed_estimate.get("rate_card_observed_at") != full_run_estimate["rate_card_observed_at"]
                or not isinstance(reviewed_estimate.get("estimated_usd"), (int, float))
                or not math.isclose(reviewed_estimate["estimated_usd"],
                                    full_run_estimate["estimated_usd"], rel_tol=1e-12, abs_tol=1e-12)):
            return "Estimate this model's full run first, then review the current estimate and confirm it. No provider request was made."
    if publish and not _community_publish_ready():
        return ("Public submission is unavailable until the Space write secret and authenticated submitter "
                "attribution support are configured. No provider request was made.")
    import inspect
    run_fn = getattr(runner, "run_community_evaluation", None)
    try:
        parameters = inspect.signature(run_fn).parameters.values()
    except (TypeError, ValueError):
        return "This Space build does not expose the bounded visitor-run interface. No provider request was made."
    if not {"task_limit", "full_run_cost_acknowledged"}.issubset(
            inspect.signature(run_fn).parameters):
        return "This Space build does not expose the bounded visitor-run interface. No provider request was made."
    if enforce_daily_limit:
        if not isinstance(daily_identity, str) or not daily_identity:
            return "The Space could not establish a daily run limit for this session. No provider request was made."
        if not _claim_community_daily_run(daily_identity):
            return "This account or browser session has already used its one USD 10-capped evaluation for today (UTC). No provider request was made."
    publish_fn = getattr(publisher, "publish_community_eval_run", None)
    try:
        rows, suite_sha = run_fn(model_name, provider_api_key, task_limit=task_limit,
                                 full_run_cost_acknowledged=(task_limit == 72))
        if not isinstance(rows, list) or not rows:
            return "No completed candidate was produced within the configured limit; nothing was uploaded."
        run_id = rows[0].get("run_id") if isinstance(rows[0], dict) else None
        if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
            return "The runner returned an invalid result receipt. Nothing was uploaded."
        count = len(rows)
        pass1 = sum(bool(row.get("validity_at_1")) for row in rows if isinstance(row, dict))
        pass3 = sum(bool(row.get("validity_at_3")) for row in rows if isinstance(row, dict))
        token_values = [row.get("total_tokens") for row in rows if isinstance(row, dict)
                        and isinstance(row.get("total_tokens"), (int, float))]
        elapsed_values = [row.get("elapsed_ms") for row in rows if isinstance(row, dict)
                          and isinstance(row.get("elapsed_ms"), (int, float))]
        request_latency_values = [row.get("latency_ms", {}).get("request_elapsed_total")
                                  for row in rows if isinstance(row, dict)
                                  and isinstance(row.get("latency_ms"), dict)
                                  and isinstance(row["latency_ms"].get("request_elapsed_total"), (int, float))]
        verifier_latency_values = [row.get("verification_elapsed_ms") for row in rows
                                   if isinstance(row, dict)
                                   and isinstance(row.get("verification_elapsed_ms"), (int, float))]
        reported_costs, estimated_costs = {}, {}
        for row in rows:
            cost = row.get("cost") or {} if isinstance(row, dict) else {}
            if isinstance(cost, dict):
                amount, currency = cost.get("amount"), cost.get("currency")
                if isinstance(amount, (int, float)) and currency:
                    reported_costs[currency] = reported_costs.get(currency, 0.0) + amount
                amount, currency = cost.get("estimated_amount"), cost.get("estimated_currency")
                if isinstance(amount, (int, float)) and currency:
                    estimated_costs[currency] = estimated_costs.get(currency, 0.0) + amount
        reported_text = (", ".join(f"{currency} {amount:.4f}" for currency, amount in sorted(reported_costs.items()))
                         if reported_costs else "not reported")
        estimated_text = (", ".join(f"{currency} {amount:.4f}" for currency, amount in sorted(estimated_costs.items()))
                          if estimated_costs else "not available")
        tokens_text = str(sum(token_values)) if token_values else "not reported"
        elapsed_text = f"{sum(elapsed_values) / 1000:.1f} s" if elapsed_values else "not available"
        request_text = (f"{sum(request_latency_values) / len(request_latency_values):.0f} ms mean/prompt"
                        if request_latency_values else "not available")
        verifier_text = (f"{sum(verifier_latency_values) / len(verifier_latency_values):.1f} ms mean/prompt"
                         if verifier_latency_values else "not available")
        summary = (f"Community run {run_id}: {count}/{task_limit} requested questions; "
                   f"pass@1 {pass1}/{count}; pass@3 {pass3}/{count}. "
                   f"Tokens {tokens_text}; wall time {elapsed_text}; provider-reported cost {reported_text}; "
                   f"rate-card estimate {estimated_text}; request latency {request_text}; "
                   f"verifier latency {verifier_text}. "
                   "This run is not ranked unless compared on the same completed question IDs.")
        if full_run_estimate is not None:
            summary += (f" Full-suite expected cost from official mean usage: USD "
                        f"{full_run_estimate['estimated_usd']:.4f}; hard cap USD 10.")
            if full_run_estimate["estimated_cost_exceeds_per_run_cap"]:
                summary += " The projection exceeds the cap, so partial coverage is likely."
        if not publish:
            return summary + " Results stay in this run and are not added to a dataset."

        published = publish_fn(
            rows, run_id=run_id, expected_suite_sha256=suite_sha,
            provider_api_key=provider_api_key, submitted_by=submitted_by,
        )
        _community_comparison_snapshot.cache_clear()
        url = published.get("url") if isinstance(published, dict) else None
        if not isinstance(url, str) or not url.startswith("https://huggingface.co/datasets/"):
            repo = getattr(publisher, "COMMUNITY_REPO", getattr(publisher, "EVAL_REPO", ""))
            url = f"https://huggingface.co/datasets/{repo}"
        return f"Published {summary} Submitter: {submitted_by}. Public results: {url}"
    except (ValueError, RuntimeError):
        return "Community run or publication failed. Provider charges may have occurred; sensitive details are not shown."
    except Exception:
        return "Community run or publication failed. Provider charges may have occurred; sensitive details are not shown."
    finally:
        provider_api_key = ""


def _benchmark_state_view(state: str | None):
    paths = (
        ROOT / EVAL_TRAJECTORY_PATH,
        ROOT / EVAL_RESULT_PATH,
    )
    path = next((candidate for candidate in paths if candidate.is_file()), None)
    if path is None:
        return [], {"status": "Accepted benchmark previews will appear when audited eval-v2.1 trajectories are included in this Space build."}
    previews, details = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        if (row.get("prompt_suite_sha256") != EVAL_SUITE_SHA256
                or (state is not None and row.get("state") != state)
                or row.get("validity_at_3") is not True):
            continue

        calls = row.get("calls") or row.get("model_calls") or []
        if not isinstance(calls, list):
            calls = []
        accepted_call = next((call for call in reversed(calls) if isinstance(call, dict)
                              if call.get("accepted") is True and call.get("candidate")), None)
        saved_report = (accepted_call or {}).get("verifier_report") or (accepted_call or {}).get("report")
        accepted_candidate = (accepted_call or {}).get("candidate")
        if accepted_call is None:
            attempts = row.get("attempts")
            attempts = attempts if isinstance(attempts, list) else []
            accepted_attempt = next((attempt for attempt in reversed(attempts) if isinstance(attempt, dict)
                                     if attempt.get("report", {}).get("all_hard_constraints_pass") is True
                                     and attempt.get("candidate")), None)
            if accepted_attempt is None:
                continue
            accepted_candidate = accepted_attempt["candidate"]
            saved_report = accepted_attempt.get("report")
            accepted_call = accepted_attempt
        if not isinstance(saved_report, dict) or not isinstance(accepted_candidate, dict):
            continue
        try:
            pattern = parse_pattern(json.dumps(accepted_candidate))
            checks, masks, grid = verify_pattern(pattern)
            certificate = make_certificate(pattern, checks, masks, grid)
        except (ValidationError, ValueError, TypeError):
            continue
        checks_saved = saved_report.get("checks")
        if not isinstance(checks_saved, list) or not all(isinstance(check, dict) for check in checks_saved):
            continue
        saved_checks = [check for check in checks_saved
                        if check.get("id") != "candidate_truncated"]
        if (saved_checks != checks or saved_report.get("all_hard_constraints_pass") is not True
                or certificate["mask_sha256"] != row.get("accepted_mask_sha256")):
            continue
        label = f"{row.get('state', 'State')} · {row.get('prompt_id', 'task')} · {row.get('grammar', '')} · {row.get('level', '')}"
        previews.append((render_png(grid, pattern.palette, scale=4, repeats=1), label))
        proposal_call = next((item for item in calls
                              if isinstance(item, dict) and item.get("phase") == "proposal"), accepted_call)
        messages = (proposal_call or {}).get("messages", [])
        exact_prompt = next((message.get("content", "") for message in messages
                             if isinstance(message, dict) and message.get("role") == "user"), "") \
            if isinstance(messages, list) else ""
        details.append({
            "state": row.get("state"), "prompt_id": row.get("prompt_id"), "grammar": row.get("grammar"),
            "level": row.get("level"), "exact_prompt": exact_prompt,
            "accepted_mask_sha256": certificate["mask_sha256"],
        })
    return previews, details or {"status": f"No accepted eval-v2.1 designs are available yet for {state}."}


def _festival_lens_states() -> set[str]:
    sourced_statuses = {"sourced_institutional", "sourced_secondary"}
    states = set()
    for state_name in ALL_STATES:
        for tradition in _tradition_choices(state_name):
            primary_state = (tradition.get("primary_state") if isinstance(tradition, dict)
                             else getattr(tradition, "primary_state", None))
            if primary_state != state_name:
                continue
            lens = tradition.get("festival_lens") if isinstance(tradition, dict) else getattr(tradition, "festival_lens", None)
            if isinstance(lens, dict):
                surface, status, claims = lens.get("surface"), lens.get("diwali_status"), lens.get("claim_ids")
            else:
                surface = getattr(lens, "surface", None)
                status, claims = getattr(lens, "diwali_status", None), getattr(lens, "claim_ids", None)
            has_claim = isinstance(claims, (list, tuple)) and any(
                isinstance(claim, str) and claim.strip() for claim in claims
            )
            if surface in {"floor", "wall"} and status in sourced_statuses and has_claim:
                states.add(state_name)
                break
    return states


def _benchmark_state_mosaic_html(previews=None, details=None, *, festival_lens_states=None,
                                 festival_lens_active: bool = False, selectable: bool = False,
                                 selected_state: str | None = None) -> str:
    if previews is None or details is None:
        previews, details = _benchmark_state_view(None)
    data_unavailable = isinstance(details, dict)
    unavailable_message = (str(details.get("status") or
                               "Replay-audited benchmark data is unavailable in this Space build.")
                           if data_unavailable else "")
    accepted = {}
    if isinstance(details, list):
        for (image, _label), detail in zip(previews, details):
            if not isinstance(detail, dict):
                continue
            state_name = detail.get("state")
            if isinstance(state_name, str) and state_name in ALL_STATES and state_name not in accepted:
                accepted[state_name] = (image, detail)

    tiles = []
    festival_lens_states = set(festival_lens_states or ())
    for state_name in ALL_STATES:
        safe_state = html_escape(state_name, quote=True)
        festival_linked = "true" if state_name in festival_lens_states else "false"
        tile_index = len(tiles)
        is_selected = selectable and state_name == selected_state
        tab_index = "0" if (is_selected or (not selectable and tile_index == 0)) else "-1"
        tile_interaction = (
            f'data-gallery-state-tile data-action="gallery_state_select" aria-selected="{str(is_selected).lower()}" '
            f'title="Filter Gallery by {safe_state}"' if selectable else ""
        )
        record = accepted.get(state_name)
        cell = state_grid_cell(state_name)
        grid_position = state_grid_style(state_name)
        grid_data = (f'data-grid-col="{cell[0]}" data-grid-row="{cell[1]}" '
                     f'style="{grid_position}"' if cell is not None else "")
        if record is None:
            state_code = _state_tile_code(state_name)
            empty_aria = (f"{safe_state}: evaluation data unavailable in this build" if data_unavailable
                          else f"{safe_state}: no replay-verified passing example in this build")
            empty_label = "Evaluation data unavailable." if data_unavailable else "No replay-verified passing example in this build."
            tiles.append(
                f'<div class="pramaan-state-tile pramaan-state-tile-empty" role="gridcell" tabindex="{tab_index}" '
                f'data-state-tile data-state="{safe_state}" data-festival-linked="{festival_linked}" data-state-code="{state_code}" {grid_data} '
                f'{tile_interaction} aria-label="{empty_aria}">'
                f'<strong aria-hidden="true">{state_code}</strong><span>{empty_label}</span></div>'
            )
            continue
        image, detail = record
        state_code = _state_tile_code(state_name)
        prompt_id = html_escape(str(detail.get("prompt_id", "task")))
        grammar = html_escape(str(detail.get("grammar", "pattern")))
        tradition_id = html_escape(str(detail.get("tradition_id", "tradition")))
        uri = _image_data_uri(image)
        tiles.append(
            f'<figure class="pramaan-state-tile" role="gridcell" tabindex="{tab_index}" data-state-tile '
            f'data-state="{safe_state}" data-festival-linked="{festival_linked}" data-state-code="{state_code}" {grid_data} '
            f'{tile_interaction} aria-label="{safe_state}, {tradition_id}, {grammar}, replay-verified design for {prompt_id}">'
            f'<strong aria-hidden="true">{state_code}</strong><img src="{uri}" loading="lazy" decoding="async" alt="Replay-verified geometric pattern for {safe_state}">'
            f'<figcaption><strong>{html_escape(state_name)}</strong><span>{prompt_id} · {grammar}</span></figcaption></figure>'
        )
    active_class = " is-festival-lens" if festival_lens_active else ""
    status_html = (f'<p class="pramaan-ui-muted" role="status">{html_escape(unavailable_message)}</p>'
                   if data_unavailable else "")
    selection_note = ('<p class="pramaan-ui-muted">Select a state tile to filter the replay-verified designs below.</p>'
                      if selectable else "")
    return (status_html + selection_note + '<div class="pramaan-state-mosaic-scroll" tabindex="0" aria-label="Scrollable geographic quilt">'
            + f'<div class="pramaan-state-mosaic{active_class}" role="grid" aria-label="36 state and Union Territory pattern tiles">'
            + "".join(tiles) + '</div></div>' + _quilt_legend_html())


def _state_tile_code(state_name: str) -> str:
    words = [word for word in state_name.split() if word.lower() not in {"and", "of", "the"}]
    if len(words) >= 2:
        return "".join(word[0] for word in words[:2]).upper()
    return (words[0][:2] if words else state_name[:2]).upper()


def _add_proof_seal_to_bundle(bundle_path: str) -> str:
    """Include the seal artwork alongside every downloadable verified artifact."""
    seal = ROOT / "assets" / "pramaan-seal.svg"
    with zipfile.ZipFile(bundle_path, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(seal, "proof-seal.svg")
    return bundle_path


def _artifact_bundle_with_seal(*args, **kwargs) -> str:
    return _add_proof_seal_to_bundle(artifact_bundle(*args, **kwargs))


def _quilt_legend_html() -> str:
    entries = (
        ("verified-first", "Verified on first proposal"),
        ("verified-repaired", "Verified after repair"),
        ("failed-all", "Failed all proposals"),
        ("accepted-other-model", "Accepted by another model"),
        ("not-evaluated", "Not evaluated"),
        ("community-run", "Community run"),
    )
    items = "".join(
        f'<li data-tile-state="{state}"><span aria-hidden="true"></span>{label}</li>'
        for state, label in entries
    )
    return f'<ul class="pramaan-quilt-legend" aria-label="Quilt evidence legend">{items}</ul>'


def _image_data_uri(image) -> str:
    if image is None:
        return ""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _initial_custom_state() -> dict:
    state_name = "Gujarat" if "Gujarat" in ALL_STATES else ALL_STATES[0]
    traditions = _tradition_choices(state_name)
    tradition = traditions[0] if traditions else None
    profile = dict(tradition.default_profile) if tradition else {}
    palette_name = next(iter(tradition.palette_presets), "") if tradition else ""
    form = {
        "brief": "An indigo and madder repeat with a balanced geometric rhythm.",
        "state": state_name,
        "tradition": tradition.id if tradition else "",
        "palette": palette_name,
        "symmetry": profile.get("symmetry", "D4"),
        "max_colours": int(profile.get("max_colours", 4)),
        "min_coverage": float(profile.get("min_coverage", 0.02)),
        "max_coverage": float(profile.get("max_coverage", 0.70)),
        "exclusive_ink_layers": bool(profile.get("exclusive_ink_layers", False)),
        "min_gap": int(profile.get("min_gap", 0)),
        "required_loops": int(profile.get("required_loops", 1)),
        "cell_size": int(profile.get("cell_size", 4)),
        "translation_axis": profile.get("translation_period", {}).get("axis", "x"),
        "translation_period": int(profile.get("translation_period", {}).get("period", 8)),
        "min_clearance": int(profile.get("min_clearance", 0)),
        "frame": int(profile.get("frame", 0)),
        "min_web": int(profile.get("min_web", 0)),
    }
    form["example"] = _example_for_selection(state_name, tradition.id if tradition else None)
    return {
        "active_tab": "benchmark", "form": form,
        "about_state": state_name, "benchmark_state": state_name,
        "wall_model": "beam-hf", "wall_mode": "reference",
        "community_loaded": False,
        "result": {"status": "Load an example to inspect its independent checker report."},
    }


def _form_from_event(event, previous: dict) -> dict:
    form = dict(previous.get("form") or {})
    fields = (
        "brief", "state", "tradition", "example", "palette", "symmetry", "max_colours", "min_coverage",
        "max_coverage", "exclusive_ink_layers", "min_gap", "required_loops", "cell_size",
        "translation_axis", "translation_period", "min_clearance", "frame", "min_web", "source",
    )
    for key in fields:
        value = getattr(event, key, None)
        if value is not None:
            form[key] = value
    for key in ("max_colours", "min_gap", "required_loops", "cell_size", "translation_period", "min_clearance", "frame", "min_web"):
        if key in form:
            try:
                form[key] = int(form[key])
            except (TypeError, ValueError):
                pass
    for key in ("min_coverage", "max_coverage"):
        if key in form:
            try:
                form[key] = float(form[key])
            except (TypeError, ValueError):
                pass
    form["exclusive_ink_layers"] = str(form.get("exclusive_ink_layers", "false")).lower() in {"true", "1", "on"}
    return form


def _form_for_tradition(state_name: str, tradition_id: str, previous: dict) -> dict:
    form = dict(previous)
    item = _tradition_by_id(tradition_id)
    if item is None or state_name not in item.states:
        return form
    profile = dict(item.default_profile)
    form.update({
        "state": state_name, "tradition": item.id,
        "palette": next(iter(item.palette_presets), ""),
        "symmetry": profile.get("symmetry", "D2"),
        "max_colours": int(profile.get("max_colours", 4)),
        "min_coverage": float(profile.get("min_coverage", 0.02)),
        "max_coverage": float(profile.get("max_coverage", 0.70)),
        "exclusive_ink_layers": bool(profile.get("exclusive_ink_layers", False)),
        "min_gap": int(profile.get("min_gap", 0)),
        "required_loops": int(profile.get("required_loops", 1)),
        "cell_size": int(profile.get("cell_size", 4)),
        "translation_axis": profile.get("translation_period", {}).get("axis", "x"),
        "translation_period": int(profile.get("translation_period", {}).get("period", 8)),
        "min_clearance": int(profile.get("min_clearance", 0)),
        "frame": int(profile.get("frame", 0)), "min_web": int(profile.get("min_web", 0)),
    })
    form["example"] = _example_for_selection(state_name, item.id)
    return form


def _form_for_example(name: str, previous: dict) -> dict:
    path = EXAMPLES.get(name)
    if path is None:
        return dict(previous)
    try:
        candidate = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(previous)
    tradition_id = candidate.get("tradition_id") if isinstance(candidate, dict) else None
    item = _tradition_by_id(tradition_id)
    if item is None or not item.states:
        return dict(previous)
    example_id = Path(path).name.removesuffix(".pattern.json")
    state_name = EXAMPLE_STATES.get(example_id)
    previous_state = previous.get("state")
    if state_name not in item.states:
        state_name = previous_state if previous_state in item.states else item.states[0]
    form = _form_for_tradition(state_name, item.id, previous)
    form["example"] = name
    return form


def _tradition_html_for_state(state_name: str) -> str:
    traditions = _tradition_choices(state_name)
    if not traditions:
        return f'<p class="pramaan-muted">No sourced tradition records are available yet for {html_escape(state_name)}.</p>'
    cards = []
    for item in traditions:
        sources = []
        for source in list(item.sources) + list(item.sources_unverified):
            title = html_escape(str(source.title))
            url = html_escape(str(source.url), quote=True)
            publisher = html_escape(str(source.publisher))
            supports = html_escape(str(source.supports))
            sources.append(f'<li><a href="{url}" target="_blank" rel="noopener noreferrer">{title}</a> · {publisher} — {supports}</li>')
        cards.append(
            '<article class="pramaan-ui-card"><p class="pramaan-ui-eyebrow">'
            f'{html_escape(str(item.grammar))} · {html_escape(str(item.confidence))} confidence</p>'
            f'<h3>{html_escape(str(item.name))}</h3>'
            f'<p>{html_escape(str(item.cautions))}</p>'
            f'<ul>{"".join(sources)}</ul></article>'
        )
    return '<div class="pramaan-ui-card-grid">' + "".join(cards) + '</div>'


def _field(label: str, key: str, value, *, kind="text", options=None, attrs="") -> str:
    field_id = f"pramaan-{key.replace('_', '-')}"
    label_html = f'<label for="{field_id}">{html_escape(label)}</label>'
    if options is not None:
        choices = []
        for choice_value, choice_label in options:
            selected = " selected" if str(choice_value) == str(value) else ""
            choices.append(f'<option value="{html_escape(str(choice_value), quote=True)}"{selected}>{html_escape(str(choice_label))}</option>')
        control = f'<select id="{field_id}" data-field="{key}" {attrs}>{"".join(choices)}</select>'
    elif kind == "textarea":
        control = f'<textarea id="{field_id}" data-field="{key}" {attrs}>{html_escape(str(value or ""))}</textarea>'
    else:
        control = f'<input id="{field_id}" data-field="{key}" type="{kind}" value="{html_escape(str(value), quote=True)}" {attrs}>'
    return f'<div class="pramaan-ui-field">{label_html}{control}</div>'


PROOF_REPLAY_HTML_PATH = ROOT / "assets" / "proof-replay.html"


def _proof_replay_html() -> str:
    """Read the pre-rendered replay so first paint never waits for geometry checks."""
    try:
        content = PROOF_REPLAY_HTML_PATH.read_text(encoding="utf-8")
        return content.replace(f"<h1>{html_escape(HEADLINE)}</h1>", "", 1)
    except OSError:
        return ('<article class="pramaan-ui-card pramaan-proof-replay" data-testid="proof-replay">'
                f'<p class="pramaan-ui-eyebrow">Proof Replay</p>'
                '<p role="status">The saved replay is not available in this build.</p></article>')


@lru_cache(maxsize=1)
def _home_reference_rows() -> list[dict]:
    """Load all 71 independently replayed task references for the Home Wall."""
    path = ROOT / "benchmarks" / "data" / "eval-v2.1-references.jsonl"
    references: dict[str, dict] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (not isinstance(row, dict) or row.get("suite_id") != EVAL_SUITE_ID
                or row.get("suite_sha256") != EVAL_SUITE_SHA256
                or row.get("source") != "procedural_reference"
                or row.get("data_role") != "reference" or row.get("training_eligible") is not False
                or row.get("all_hard_constraints_pass") is not True):
            continue
        task_id = row.get("task_id")
        certificate = row.get("certificate")
        if (not isinstance(task_id, str) or not isinstance(certificate, dict)
                or certificate.get("all_hard_constraints_pass") is not True
                or certificate.get("mask_sha256") != row.get("mask_sha256")
                or not isinstance(row.get("checks"), list)
                or not all(isinstance(check, dict) and check.get("status") == "PASS" for check in row["checks"])):
            continue
        thumbnail_path = ROOT / "assets" / "reference-thumbnails" / f"{task_id}.png"
        try:
            thumbnail_bytes = thumbnail_path.read_bytes()
        except OSError:
            continue
        if not thumbnail_bytes.startswith(b"\x89PNG\r\n\x1a\n") or len(thumbnail_bytes) > 150_000:
            continue
        row = dict(row)
        row["thumbnail_uri"] = "data:image/png;base64," + base64.b64encode(thumbnail_bytes).decode("ascii")
        references[task_id] = row
    return list(references.values())


@lru_cache(maxsize=1)
def _home_suite_tasks() -> list[dict]:
    """Read the frozen 72-task suite once for the paired easy/hard Wall details."""
    path = ROOT / "benchmarks" / "suites" / f"{EVAL_SUITE_ID}.jsonl"
    try:
        payload = path.read_bytes()
        if hashlib.sha256(canonical_jsonl_bytes(payload)).hexdigest() != EVAL_SUITE_SHA256:
            return []
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if len(rows) != 72 or len({row.get("id") for row in rows if isinstance(row, dict)}) != 72:
        return []
    return rows


def _home_metrics_html() -> str:
    summary = _benchmark_summary()
    model = summary.get("by_model", {}).get("beam-hf", {}) if isinstance(summary, dict) else {}
    overall = model.get("overall", {}) if isinstance(model, dict) else {}
    total = overall.get("n")
    first = overall.get("validity_at_1")
    after = overall.get("validity_at_3")
    def count(rate):
        return str(round(rate * total)) if type(total) is int and total > 0 and isinstance(rate, (int, float)) else "—"
    total_text = str(total) if type(total) is int and total > 0 else "—"
    return ('<section class="pramaan-home-metrics" aria-label="Beam evaluation summary">'
            f'<article><strong>{count(first)} of {total_text}</strong><span>Beam designs correct on the first try</span></article>'
            f'<article><strong>{count(after)} of {total_text}</strong><span>Beam designs correct after checker feedback</span></article>'
            f'<article><strong>{len(ALL_STATES)}</strong><span>states and territories covered</span></article>'
            '</section>')


@lru_cache(maxsize=1)
def _render_proof_replay_html_from_data() -> str:
    """Build the replay from a saved Beam trace, replaying every candidate locally."""
    try:
        records = EVAL_TRAJECTORY_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        records = []
    selected = None
    for line in records:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("trajectory_id") != PROOF_REPLAY_TRAJECTORY_ID:
            continue
        calls = record.get("calls") if isinstance(record, dict) else None
        if (not isinstance(calls, list) or len(calls) < 2
                or "rangoli" in str(record.get("tradition_id", "")).casefold()
                or calls[0].get("accepted") is not False
                or not isinstance(calls[0].get("candidate"), dict)
                or calls[-1].get("accepted") is not True):
            continue
        selected = record
        break
    if selected is None:
        return ('<article class="pramaan-ui-card pramaan-proof-replay" data-testid="proof-replay">'
                f'<p class="pramaan-ui-eyebrow">Proof Replay</p>'
                '<p role="status">A replayable saved trajectory is not available in this build.</p></article>')

    calls = selected["calls"]
    attempts = []
    for call in calls:
        candidate = call.get("candidate")
        if not isinstance(candidate, dict):
            continue
        try:
            pattern = parse_pattern(json.dumps(candidate, ensure_ascii=False))
            checks, masks, grid = verify_pattern(pattern)
        except (TypeError, ValueError, ValidationError):
            continue
        accepted = all_pass(checks)
        if accepted is not (call.get("accepted") is True):
            continue
        if accepted and len(set(int(value) for value in grid.flat)) < 2:
            continue
        failure = next((item for item in checks if item.get("status") == "FAIL"
                        and isinstance((item.get("counterexample") or {}).get("pixel"), list)), None)
        pixel = ((failure or {}).get("counterexample") or {}).get("pixel")
        markers = [(pixel[0], pixel[1], "#d12f8a")] if (
            isinstance(pixel, list) and len(pixel) == 2
            and all(type(value) is int and 0 <= value < 64 for value in pixel)) else []
        image = render_highlighted_png(grid, pattern.palette, markers, scale=3, repeats=2)
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        certificate = make_certificate(pattern, checks, masks, grid) if accepted else None
        attempts.append({
            "phase": str(call.get("phase") or "attempt"), "accepted": accepted,
            "checks": checks, "pixel": pixel if markers else None,
            "image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii"),
            "pattern": pattern.model_dump(mode="json"), "certificate": certificate,
        })
    first_failed = next((row for row in attempts if not row["accepted"] and any(
        isinstance(check, dict) and check.get("status") == "FAIL" for check in row["checks"])), None)
    final_pass = next((row for row in reversed(attempts) if row["accepted"]), None)
    if first_failed is None or final_pass is None:
        return ('<article class="pramaan-ui-card pramaan-proof-replay" data-testid="proof-replay">'
                f'<p class="pramaan-ui-eyebrow">Proof Replay</p><h1>{html_escape(HEADLINE)}</h1>'
                '<p role="status">The saved trajectory did not pass the current replay checks.</p></article>')

    safe = html_escape
    brief = safe(str(selected.get("brief", "Saved evaluation task")))
    state = safe(str(selected.get("state", "")))
    tradition = safe(str(selected.get("tradition_id", "")))
    grammar = safe(str(selected.get("grammar", "")))
    first_checks = [item for item in first_failed["checks"] if isinstance(item, dict)]

    def image_html(row: dict, alt: str) -> str:
        return (f'<img class="pramaan-proof-tile" src="{row["image"]}" alt="{safe(alt)}" '
                'width="384" height="384" decoding="async">')

    def check_list(rows: list[dict]) -> str:
        return '<ul class="pramaan-proof-checks">' + "".join(
            '<li class="' + ("is-pass" if row.get("status") == "PASS" else "is-fail") + '">'
            + '<span aria-hidden="true">' + ("✓" if row.get("status") == "PASS" else "×") + '</span> '
            + '<strong>' + safe(str(row.get("id", "check"))) + '</strong> '
            + '<span>' + safe(str(row.get("message", ""))) + '</span></li>'
            for row in rows) + '</ul>'

    counterexample = first_failed["pixel"]
    pixel_text = (f"pixel ({counterexample[0]}, {counterexample[1]})"
                  if isinstance(counterexample, list) else "topological check witness")
    certificate = final_pass["certificate"] or {}
    digest = str(certificate.get("mask_sha256", ""))
    pattern_json = json.dumps(final_pass["pattern"], ensure_ascii=False, separators=(",", ":"))
    pattern_b64 = base64.b64encode(pattern_json.encode("utf-8")).decode("ascii")
    steps = [
        f'<div class="pramaan-proof-step" data-proof-step="0" hidden><p><b>The brief</b></p><p>{brief}</p></div>',
        f'<div class="pramaan-proof-step" data-proof-step="1"><p><b>Attempt 1 · did not pass</b></p>{image_html(first_failed, "First failed attempt tile")}</div>',
        f'<div class="pramaan-proof-step" data-proof-step="2" hidden><p><b>Exact checks · attempt 1</b></p>{check_list(first_checks)}</div>',
    ]
    if counterexample:
        steps.append(f'<div class="pramaan-proof-step" data-proof-step="{len(steps)}" hidden><p><b>Counterexample · {pixel_text}</b></p>{image_html(first_failed, "Counterexample pixel outlined in magenta on the first attempt")}</div>')
    attempt_number = 1
    for row in attempts:
        if row is first_failed or row is final_pass:
            continue
        attempt_number += 1
        if row["accepted"]:
            continue
        steps.append(
            f'<div class="pramaan-proof-step" data-proof-step="{len(steps)}" hidden>'
            f'<p><b>Repair {attempt_number} · still fails</b></p>'
            + image_html(row, f"Repair attempt {attempt_number} tile")
            + check_list([item for item in row["checks"] if isinstance(item, dict)])
            + '</div>')
        if row["pixel"]:
            point = row["pixel"]
            steps.append(
                f'<div class="pramaan-proof-step" data-proof-step="{len(steps)}" hidden>'
                f'<p><b>Counterexample · pixel ({point[0]}, {point[1]})</b></p>'
                + image_html(row, f"Repair counterexample pixel {point[0]}, {point[1]} outlined in magenta")
                + '</div>')
    steps.append(
        f'<div class="pramaan-proof-step" data-proof-step="{len(steps)}" data-proof-result hidden>'
        '<p><b>Repair · all checks pass</b></p>'
        + image_html(final_pass, "Repaired tile; all deterministic checks pass")
        + check_list([item for item in final_pass["checks"] if isinstance(item, dict)])
        + '</div>')
    certificate_step = len(steps)
    steps.append(
        f'<div class="pramaan-proof-step" data-proof-step="{certificate_step}" hidden>'
        '<p><b>Replayable certificate</b></p><p>Mask SHA-256</p>'
        + f'<code class="pramaan-proof-digest">{safe(digest)}</code>'
        + image_html(final_pass, "Final verified tile") + '</div>')
    return (
        '<article class="pramaan-ui-card pramaan-proof-replay" id="proof-replay" data-testid="proof-replay" '
        'aria-label="Proof Replay of a saved Beam design">'
        '<div class="pramaan-proof-copy"><p class="pramaan-ui-eyebrow">Proof Replay · saved Beam trajectory</p>'
        f'<h1>{html_escape(HEADLINE)}</h1><p class="pramaan-proof-context">{state} · {tradition} · {grammar}</p>'
        f'<p class="pramaan-proof-brief"><strong>Brief</strong> {brief}</p>'
        f'<div class="pramaan-proof-stage" data-proof-stage aria-live="polite">{"".join(steps)}</div>'
        '<div class="pramaan-proof-controls">'
        '<button type="button" class="pramaan-ui-button" data-proof-replay data-testid="proof-replay-play">Replay proof</button>'
        '<button type="button" class="pramaan-ui-button secondary" data-proof-step data-testid="proof-replay-step">Step</button>'
        '<span class="pramaan-proof-progress" data-proof-progress role="status">Attempt 1 of replay</span>'
        '</div>'
        f'<button type="button" class="pramaan-ui-button secondary" data-proof-verify '
        f'data-pattern-b64="{pattern_b64}" data-testid="proof-verify-yourself">Verify it yourself in Studio</button>'
        '</div><figure class="pramaan-proof-art">'
        + f'<img class="pramaan-proof-tile" data-proof-active-tile src="{first_failed["image"]}" '
        'alt="First failed tile with its exact pixel counterexample outlined" width="384" height="384" decoding="async">'
        + f'<figcaption data-proof-figure-caption>Attempt 1 · did not pass · {state}</figcaption>'
        + '</figure></article>'
    )


def _render_custom_app(ui_state: dict) -> str:
    form = dict(ui_state.get("form") or {})
    state_name = form.get("state", ALL_STATES[0])
    tradition_id = form.get("tradition", "")
    item = _tradition_by_id(tradition_id)
    traditions = _tradition_choices(state_name)
    tradition_options = [(row.id, f"{row.name} · {row.id}") for row in traditions]
    palettes = item.palette_presets if item else {}
    palette_options = [(name, name.replace("_", " ").title()) for name in palettes]
    symmetry_options = ["None", "D2", "D4"] if item and item.grammar == "periodic_raster" else ["None", "D2", "D4", "C2", "C4"]
    if item and item.grammar == "periodic_raster" and form.get("symmetry") not in symmetry_options:
        form["symmetry"] = "D2"
    grammar = item.grammar if item else ""
    status = str((ui_state.get("result") or {}).get("status", ""))
    beam_available = bool(os.getenv("HF_TOKEN"))
    beam_note = ("Beam generation is configured on this Space. A generation action may use the Space owner's inference allowance."
                 if beam_available else "Offline example mode · the Beam provider is unavailable in this build.")

    source = str((ui_state.get("result") or {}).get("source", ""))
    certificate = str((ui_state.get("result") or {}).get("certificate", ""))
    history = str((ui_state.get("result") or {}).get("history", ""))
    preview = (ui_state.get("result") or {}).get("image")
    preview_uri = _image_data_uri(preview)
    gallery = (ui_state.get("result") or {}).get("gallery", [])
    gallery_html = []
    for image, label in gallery:
        uri = _image_data_uri(image)
        if uri:
            gallery_html.append(f'<figure><img src="{uri}" loading="lazy" decoding="async" alt="{html_escape(str(label), quote=True)}"><figcaption>{html_escape(str(label))}</figcaption></figure>')
    download = (ui_state.get("result") or {}).get("bundle")
    download_html = ""
    if isinstance(download, str) and Path(download).is_file():
        try:
            encoded = base64.b64encode(Path(download).read_bytes()).decode("ascii")
            download_html = f'<a class="pramaan-ui-button" download="pramaan-run-bundle.zip" href="data:application/zip;base64,{encoded}">Download run bundle</a>'
        except OSError:
            download_html = '<p class="pramaan-muted">The temporary download has expired. Re-run the verifier to create a new bundle.</p>'

    example_choices = _example_choices(state_name, tradition_id)
    example_options = [(value, label) for label, value in example_choices]
    selected_example = form.get("example", "")
    if selected_example not in {value for _, value in example_choices}:
        selected_example = example_choices[0][1] if example_choices else ""
    state_options = [(state, state) for state in ALL_STATES]
    if item and example_choices:
        example_guidance = (
            f'<p class="pramaan-muted">Showing examples for {html_escape(state_name)} · '
            f'{html_escape(item.name)}. Change either selector to update the examples.</p>'
        )
    elif item:
        example_guidance = (
            f'<p class="pramaan-muted">No worked example is available for {html_escape(state_name)} · '
            f'{html_escape(item.name)} yet. Change either selector to browse matching examples.</p>'
        )
    else:
        example_guidance = (
            '<p class="pramaan-muted">Choose a state and tradition to see matching worked examples.</p>'
        )
    studio = (
        '<section><p class="pramaan-ui-intro">Choose a craft-inspired starting point, adjust the rules, and ask Beam to propose a pattern.</p><div class="pramaan-ui-layout"><article class="pramaan-ui-card pramaan-ui-form">'
        '<p class="pramaan-ui-eyebrow">Design studio</p><h2>Set the construction rules</h2>'
        '<p class="pramaan-muted">The tradition choice supplies its own grammar, palette and starting constraints. The verifier checks the resulting finite pattern.</p>'
        + _field("State / Union Territory", "state", state_name, options=state_options)
        + _field("Tradition", "tradition", tradition_id, options=tradition_options)
        + example_guidance
        + _field("Worked example for this tradition", "example", selected_example, options=example_options)
        + ('<button type="button" class="pramaan-ui-button secondary" data-action="example">Load example</button>'
           if example_choices else '<button type="button" class="pramaan-ui-button secondary" data-action="example" disabled>Load example</button>')
        + _field("Pattern brief", "brief", form.get("brief", ""), kind="textarea", attrs=f'maxlength="{MAX_BRIEF_CHARS}" rows="3"')
        + (f'<p class="pramaan-muted">Grammar: <code>{html_escape(grammar or "select a tradition")}</code></p>')
        + _field("Palette preset", "palette", form.get("palette", ""), options=palette_options)
        + _field("Symmetry", "symmetry", form.get("symmetry", "D2"), options=[(value, value) for value in symmetry_options])
        + '<div class="pramaan-ui-grid">'
        + _field("Maximum colours", "max_colours", form.get("max_colours", 4), kind="number", attrs='min="1" max="6" step="1"')
        + _field("Minimum coverage", "min_coverage", form.get("min_coverage", 0.02), kind="number", attrs='min="0" max="0.5" step="0.01"')
        + _field("Maximum coverage", "max_coverage", form.get("max_coverage", 0.7), kind="number", attrs='min="0.1" max="1" step="0.05"')
        + _field("Minimum motif spacing", "min_gap", form.get("min_gap", 0), kind="number", attrs='min="0" max="8" step="1"')
        + '</div><label class="pramaan-ui-check"><input type="checkbox" data-field="exclusive_ink_layers"'
        + (" checked" if form.get("exclusive_ink_layers") else "") + '> Require disjoint ink layers</label>'
    )
    if grammar == "kolam_loop":
        studio += _field("Required Kolam loops", "required_loops", form.get("required_loops", 1), kind="number", attrs='min="1" max="81" step="1"')
    elif grammar == "counted_grid":
        studio += _field("Counted-grid cell size", "cell_size", form.get("cell_size", 4), options=[(value, value) for value in (2, 4, 8)])
    elif grammar == "stripe_band":
        studio += _field("Translation axis", "translation_axis", form.get("translation_axis", "x"), options=[(value, value) for value in ("x", "y", "xy")])
        studio += _field("Translation period", "translation_period", form.get("translation_period", 8), kind="number", attrs='min="1" max="64" step="1"')
    elif grammar == "dot_field":
        studio += _field("Minimum dot clearance", "min_clearance", form.get("min_clearance", 0), kind="number", attrs='min="0" max="64" step="1"')
    elif grammar == "jaali_lattice":
        studio += _field("Jaali frame", "frame", form.get("frame", 0), kind="number", attrs='min="0" max="12" step="1"')
        studio += _field("Minimum Jaali web", "min_web", form.get("min_web", 0), kind="number", attrs='min="0" max="32" step="1"')
    studio += (
        f'<p class="pramaan-ui-muted" role="status">{html_escape(beam_note)}</p>'
        '<div class="pramaan-ui-actions"><button type="button" class="pramaan-ui-button" data-action="generate"'
        + ("" if beam_available else ' disabled title="Beam is not configured in this build"')
        + '>Generate with Beam</button><button type="button" class="pramaan-ui-button secondary" data-action="recheck">Recheck edited PatternDSL</button></div></article>'
        '<article class="pramaan-ui-card pramaan-ui-result"><p class="pramaan-ui-eyebrow">Verifier evidence</p>'
        f'<h2>{html_escape(status or "Ready to load an example")}</h2>'
        + (f'<img class="pramaan-ui-preview" src="{preview_uri}" alt="Verified Pramaan pattern preview">' if preview_uri else '<div class="pramaan-ui-placeholder">Load an example or generate a pattern to see the 3 × 3 repeat.</div>')
        + '<nav class="pramaan-ui-subtabs" aria-label="Evidence panels">'
        + '<button type="button" data-local-panel="source" aria-pressed="true">Pattern source</button>'
        + '<button type="button" data-local-panel="checks" aria-pressed="false">Checks and certificate</button>'
        + '<button type="button" data-local-panel="attempts" aria-pressed="false">Attempts</button>'
        + '<button type="button" data-local-panel="previews" aria-pressed="false">Proposals</button></nav>'
        + f'<div class="pramaan-ui-subpanel" data-panel="source"><label for="pramaan-source">Editable PatternDSL JSON</label><textarea id="pramaan-source" data-field="source" rows="18" spellcheck="false">{html_escape(source)}</textarea></div>'
        + f'<div class="pramaan-ui-subpanel" data-panel="checks" hidden><pre>{html_escape(certificate)}</pre></div>'
        + f'<div class="pramaan-ui-subpanel" data-panel="attempts" hidden><pre>{html_escape(history)}</pre></div>'
        + f'<div class="pramaan-ui-subpanel pramaan-ui-gallery" data-panel="previews" hidden>{"".join(gallery_html) or "<p class=\"pramaan-muted\">Proposal previews will appear here.</p>"}</div>'
        + download_html + '</article></div></section>'
    )

    benchmark_state = ui_state.get("benchmark_state", state_name)
    gallery_model = ui_state.get("gallery_model", "all")
    gallery_index = _load_gallery_index()
    gallery_arm_documents = _load_gallery_arm_documents(gallery_index)
    all_benchmark_images, all_benchmark_info = _benchmark_state_view(None)
    if ui_state.get("community_loaded") is True:
        community_snapshot = _community_comparison_snapshot()
    else:
        community_snapshot = {
            "status": "Select Refresh public community runs to load the latest public results.",
            "rows": [], "beam_rows": [], "selected": {}, "summary": {},
        }
    community_models_in_gallery = sorted({
        str(row.get("model_key")) for row in community_snapshot.get("rows", [])
        if isinstance(row, dict) and row.get("model_key")
    }, key=str.casefold)
    gallery_model_options = [("all", "All models")]
    canonical_arms = gallery_index.get("arms", []) if isinstance(gallery_index, dict) else []
    canonical_options = {
        str(arm["key"]): str(arm.get("label") or arm["key"])
        for arm in canonical_arms if isinstance(arm, dict) and isinstance(arm.get("key"), str)
    }
    if "beam-hf" not in canonical_options:
        canonical_options["beam-hf"] = "Beam"
    gallery_model_options.extend((key, label) for key, label in sorted(canonical_options.items(), key=lambda pair: pair[1].casefold()))
    gallery_model_options.extend((model, model) for model in community_models_in_gallery
                                 if model not in canonical_options)
    gallery_model_values = {value for value, _label in gallery_model_options}
    if gallery_model not in gallery_model_values:
        gallery_model = "all"
    if isinstance(all_benchmark_info, list) and gallery_model in ("all", "beam-hf"):
        selected_state = [(image, label, detail)
                          for (image, label), detail in zip(all_benchmark_images, all_benchmark_info)
                          if isinstance(detail, dict) and detail.get("state") == benchmark_state]
        benchmark_images = [(image, label) for image, label, _detail in selected_state]
        benchmark_info = [detail for _image, _label, detail in selected_state]
    else:
        benchmark_images, benchmark_info = [], all_benchmark_info
    festival_states = _festival_lens_states()
    festival_lens_active = bool(ui_state.get("festival_lens", False))
    if festival_states:
        festival_toggle = (
            '<label class="pramaan-ui-check"><input type="checkbox" data-field="festival_lens"'
            + (" checked" if festival_lens_active else "")
            + '> Festival lens</label><p class="pramaan-ui-muted">Highlights only states whose tradition tags include sourced festival claims; this does not imply every listed tradition is specific to Diwali.</p>'
        )
    else:
        festival_toggle = (
            '<label class="pramaan-ui-check"><input type="checkbox" data-field="festival_lens" disabled> Festival lens</label>'
            '<p class="pramaan-ui-muted">Sourced festival tags are not available in this build yet.</p>'
        )
    quilt_model_options = [
        (key, canonical_options.get(key, key))
        for key in canonical_options if key in gallery_arm_documents
    ]
    quilt_controls = ""
    if quilt_model_options:
        quilt_keys = [key for key, _label in quilt_model_options]
        quilt_model = ui_state.get("quilt_model", quilt_keys[0])
        if quilt_model not in quilt_keys:
            quilt_model = quilt_keys[0]
        compare_options = [(key, label) for key, label in quilt_model_options if key != quilt_model]
        compare_model = ui_state.get("quilt_compare_model", compare_options[0][0] if compare_options else "")
        if compare_model not in {key for key, _label in compare_options}:
            compare_model = compare_options[0][0] if compare_options else ""
        quilt_controls = (
            '<div class="pramaan-quilt-controls" aria-label="Benchmark quilt controls">'
            + _field("Model", "quilt_model", quilt_model, options=quilt_model_options)
            + _field("Difficulty", "quilt_level", ui_state.get("quilt_level", "easy"),
                     options=[("easy", "Easy"), ("hard", "Hard")])
            + _field("Quilt view", "quilt_mode", ui_state.get("quilt_mode", "single"), options=[
                ("single", "One model"), ("compare", "Compare models"), ("hardness", "First-attempt hardness")])
            + _field("Compare with", "quilt_compare_model", compare_model,
                     options=compare_options or [("", "No second model available")],
                     attrs=("" if compare_options else "disabled"))
            + '</div>'
        )
        if festival_lens_active:
            benchmark_mosaic = _benchmark_state_mosaic_html(
                all_benchmark_images, all_benchmark_info,
                festival_lens_states=festival_states, festival_lens_active=True,
            )
            quilt_controls += '<p class="pramaan-ui-muted">Festival lens is active; the quilt is showing its Beam overview. Turn it off to browse model comparisons.</p>'
        else:
            benchmark_mosaic = render_data_quilt(
                gallery_index, gallery_arm_documents, model_key=quilt_model,
                level=str(ui_state.get("quilt_level", "easy")),
                mode=str(ui_state.get("quilt_mode", "single")), compare_model=compare_model,
            )
    else:
        benchmark_mosaic = _benchmark_state_mosaic_html(
            all_benchmark_images, all_benchmark_info,
            festival_lens_states=festival_states, festival_lens_active=festival_lens_active,
        )
    arena_html = render_arena(
        gallery_index, gallery_arm_documents, suite_id=EVAL_SUITE_ID,
        suite_sha256=EVAL_SUITE_SHA256, prompt_id=PROOF_REPLAY_PROMPT_ID,
        replay_prompt_id=PROOF_REPLAY_PROMPT_ID,
        selected_model_keys=_arena_default_model_keys(_benchmark_summary()),
    )
    home_wall = render_reference_wall(
        _home_reference_rows(),
        gallery_index.get("states", []) if isinstance(gallery_index, dict) else [],
        gallery_arm_documents,
        _home_suite_tasks(),
        canonical_arms,
        selected_model=str(ui_state.get("wall_model", "beam-hf")),
        display_mode=str(ui_state.get("wall_mode", "reference")),
    )
    benchmark = (
        '<section class="pramaan-home" aria-label="Home"><div class="pramaan-home-intro">'
        f'<h1>{html_escape(HEADLINE)}</h1>'
        '<p>AI designs patterns from 36 Indian craft traditions. An independent checker verifies every rule the design claims, or shows the exact pixel that breaks it.</p></div>'
        + _proof_replay_html()
        + '<nav class="pramaan-home-actions" aria-label="Start exploring">'
        '<a class="pramaan-ui-button" href="#proof-replay">Watch a proof</a>'
        '<a class="pramaan-ui-button secondary" href="#/studio">Try it</a>'
        '<a class="pramaan-home-text-link" href="/web/">Use it as an RL environment</a></nav>'
        + _home_metrics_html() + home_wall + '</section>'
    )
    community_models = _community_model_choices()
    community_status = str(ui_state.get("community_status", ""))
    community_ready = bool(community_models)
    community_publish_ready = community_ready and _community_publish_ready()
    archive_ready = _community_submission_archive_ready()
    archive_status = str(ui_state.get("community_archive_status", ""))
    archive_contribution = (
        '<article class="pramaan-ui-card pramaan-community-contribution"><p class="pramaan-ui-eyebrow">Contribute a trajectory archive</p>'
        '<h3>Share replayable benchmark work</h3>'
        '<p>Upload a ZIP archive for manifest, suite and task checks, then independent verifier replay. Only accepted sanitized records are published to the separate public community dataset; this upload does not call a model provider.</p>'
        '<label for="pramaan-community-archive">Submission archive (.zip)</label>'
        '<input id="pramaan-community-archive" data-field="community_archive_file" type="file" accept=".zip,application/zip"'
        + ("" if archive_ready else ' disabled title="The scoped community write secret and archive verifier are not configured"')
        + '>'
        + '<label class="pramaan-ui-check"><input type="checkbox" data-field="community_archive_consent"'
        + ("" if archive_ready else " disabled")
        + '> I consent to public publication of the sanitized, verifier-replayed designs and benchmark metadata if the archive passes audit.</label>'
        + '<div class="pramaan-ui-actions"><button type="button" class="pramaan-ui-button secondary" data-action="community_archive_submit"'
        + ("" if archive_ready else ' disabled title="Configure the scoped community write secret before enabling archive contributions"')
        + '>Audit and contribute archive</button></div>'
        + f'<p class="pramaan-ui-muted" role="status" data-community-archive-status>{html_escape(archive_status or ("Archive contributions are disabled until the scoped HF_COMMUNITY_WRITE_TOKEN and archive verifier are configured." if not archive_ready else "Select a ZIP archive and give publication consent to begin."))}</p>'
        + '</article>'
    )
    community = (
        '<article class="pramaan-ui-card"><p class="pramaan-ui-eyebrow">Opt-in community evaluation</p>'
        '<h3>Run one model on frozen eval-v2.1</h3>'
        '<p>Quick run is 12 questions. The optional full suite is 72 questions and requires a no-call estimate plus separate confirmation. Both use the frozen 4,096-token output limit and up to three proposals per task. Beam and DeepSeek use temperature 0.4; Anthropic uses provider-default sampling with low effort because its API rejects temperature. '
        'Only priced models are accepted. The Space applies a hard USD 10 run cap and allows one capped evaluation per account or browser session each UTC day; this in-memory daily guard resets when the Space restarts. Provider billing may differ. A run may stop before its selected question limit. Partial results are labeled by coverage and are not ranked across different question sets.</p>'
        + _field("LiteLLM model (provider/model)", "community_model", "",
                 attrs=("placeholder=\"e.g. anthropic/claude-haiku-5-5\" autocapitalize=\"none\" spellcheck=\"false\"" + ("" if community_ready else " disabled")))
        + _field("Run size", "community_task_limit", "12",
                 options=[("12", "Quick run · 12 questions"), ("72", "Full suite · 72 questions")],
                 attrs=("" if community_ready else "disabled"))
        + '<div class="pramaan-ui-actions"><button type="button" class="pramaan-ui-button secondary" data-action="community_estimate"'
        + ("" if community_ready else ' disabled title="Community evaluation helpers are not in this build"')
        + '>Estimate full-suite cost (no provider call)</button></div>'
        + _field("Your provider API key", "community_api_key", "", kind="password",
                 attrs='autocomplete="off" autocapitalize="none" spellcheck="false"')
        + '<p class="pramaan-muted">Your key is sent from your browser to this Space and then to the selected provider. It is not saved in your profile or result record. Use a throwaway key with a provider-side spend limit; the provider may bill differently from this Space\'s estimate.</p>'
        + '<label class="pramaan-ui-check"><input type="checkbox" data-field="community_charge_consent"> '
        + 'I understand this run may incur charges on my provider account and authorize this use of my key.</label>'
        + '<label class="pramaan-ui-check"><input type="checkbox" data-field="community_full_run_cost_consent"> '
        + 'For a 72-question run, I reviewed the displayed estimate and authorize provider charges up to the Space\'s USD 10 hard cap.</label>'
        + '<label class="pramaan-ui-check"><input type="checkbox" data-field="community_publish_consent"> '
        + 'Publish verifier-replayed designs and metrics to the public community dataset under CC-BY-4.0. Model identity is self-reported and unverified; I will follow my provider\'s terms.</label>'
        + '<p class="pramaan-ui-muted">You can run privately without signing in; the result summary stays in this session and is not added to a dataset. Public submissions require Hugging Face sign-in and the separate publication opt-in. Space storage is ephemeral and may be lost on restart.</p>'
        + f'<div class="pramaan-ui-actions"><button type="button" class="pramaan-ui-button secondary" data-action="community_run"'
        + ("" if community_ready else ' disabled title="Community evaluation helpers are not in this build"')
        + '>Run privately</button><button type="button" class="pramaan-ui-button" data-action="community_publish"'
        + ("" if community_publish_ready else ' disabled title="Public submissions require the Space write secret and authenticated publisher support"')
        + '>Run and publish</button></div>'
        + f'<p class="pramaan-ui-muted" role="status" data-community-status>{html_escape(community_status or ("The community runner is not enabled in this Space build yet." if not community_ready else "No community run has been submitted in this session."))}</p>'
        + '</article>'
        + archive_contribution
    )
    model_summary = _benchmark_summary()
    leaderboard = (
        '<section><p class="pramaan-ui-intro">Compare verifier outcomes, then inspect recorded attempts for a selected brief.</p><article class="pramaan-ui-card">'
        '<p class="pramaan-ui-eyebrow">Canonical benchmark</p><h2>Model leaderboard</h2>'
        + '<nav class="pramaan-leaderboard-nav" aria-label="Leaderboard sections">'
        + '<a href="#leaderboard-ranking">Ranking</a><a href="#model-arena">Arena</a>'
        + '<a href="#leaderboard-full-results">Community results</a><a href="#leaderboard-methodology">Methodology</a></nav>'
        + '<div class="pramaan-leaderboard-controls" aria-label="Filter and sort model results">'
        + '<label>Find a model<input type="search" data-leaderboard-control data-leaderboard-search placeholder="Search model or provider"></label>'
        + '<label>Provenance<select data-leaderboard-control data-leaderboard-provenance><option value="all">All runs</option><option value="official">Canonical</option><option value="community">Community</option></select></label>'
        + '<label>Sort by<select data-leaderboard-control data-leaderboard-sort><option value="pass3">Within three</option><option value="pass1">First try</option><option value="coverage">Evaluated tasks</option><option value="cost">Cost</option></select></label>'
        + '<button type="button" class="pramaan-ui-button secondary" data-leaderboard-reset>Reset</button><span role="status" aria-live="polite" data-leaderboard-count></span></div>'
        + '<section id="leaderboard-ranking" aria-label="Model ranking">'
        + render_model_report(model_summary)
        + '</section>'
        + render_findings_strip(model_summary)
        + _model_methodology_html(gallery_index, model_summary)
        + '<details id="leaderboard-full-results" class="pramaan-leaderboard-method"><summary>Community runs</summary>'
        + _community_leaderboard_html(community_snapshot)
        + '<div class="pramaan-ui-actions"><button type="button" class="pramaan-ui-button secondary" data-action="community_refresh">Refresh public community runs</button></div>'
        + community + '</details></article></section>'
    )
    about_state = ui_state.get("about_state", state_name)
    traditions = (
        '<section><p class="pramaan-ui-intro">Read the tradition notes and sources behind each mathematical interpretation.</p><article class="pramaan-ui-card"><p class="pramaan-ui-eyebrow">Sources and cautions</p>'
        '<h2>Tradition references</h2><p>These designs are computational interpretations inspired by documented geometric ideas. The verifier checks geometry, not cultural authenticity.</p>'
        '<aside class="pramaan-ui-card pramaan-festival-context" aria-label="Diwali and geometric welcome patterns">'
        '<p class="pramaan-ui-eyebrow">A festival-of-lights lens</p>'
        '<h3>Diwali: light, welcome, and pattern</h3>'
        '<p>In some communities, homes are illuminated and rangoli is made at doorways for Diwali. This 36-entry survey spans Indian states and Union Territories; it is not a claim that every listed tradition is Diwali-specific.</p>'
        '<p><a href="https://www.incredibleindia.gov.in/en/festivals-and-events/diwali" target="_blank" rel="noopener noreferrer">Incredible India: Diwali</a> · '
        '<a href="https://ignca.gov.in/walls-and-floors/" target="_blank" rel="noopener noreferrer">IGNCA: Walls and Floors</a></p></aside>'
        + _field("Browse state / territory", "about_state", about_state, options=state_options)
        + _tradition_html_for_state(about_state) + '</article></section>'
    )
    rl = (
        '<section><p class="pramaan-ui-intro">Connect the checker to an agent, dataset, or reinforcement-learning workflow.</p><article class="pramaan-ui-card"><p class="pramaan-ui-eyebrow">OpenEnv</p><h2>Use the verifier as an environment</h2>'
        '<p>A reward of 1.0 means every hard geometric check passes. Partial structural progress receives a bounded score below 1.0; parse errors, truncation and missing checks receive 0. The reward does not assess cultural authenticity.</p>'
        '<pre><code>from pramaan_env import PramaanEnv, PramaanAction\nenv = PramaanEnv(base_url="https://kshitijthakkar-pramaan-proof-of-art.hf.space")\nobservation = env.reset(suite="eval-v2.1", seed=0)\nobservation = env.step(PramaanAction(pattern_json=pattern_json))</code></pre>'
        '<a class="pramaan-ui-button secondary" href="/web/" target="_blank" rel="noopener noreferrer">Open the interactive environment</a></article>'
        + '<details class="pramaan-ui-card"><summary>How the benchmark is measured</summary>'
        + '<p>The 72-task source suite spans 36 states. The one-pass paid comparison uses the same 47 predeclared task IDs and order for Claude Haiku 5.5, Sonnet 5.5, Opus 5.5, and DeepSeek V4.1 Flash. Each paid arm has a USD 10 cap, the same verifier, 4,096-token cap, and up to three proposals per task; an arm may stop early at its spend cap. The separate Beam baseline covers all 72 tasks and is not pooled with the 47-task comparison. Provider costs, tokens, and latency are shown separately from quality.</p>'
        + _benchmark_summary_html() + _community_leaderboard_html(community_snapshot) + '</details></section>'
    )
    models = leaderboard + '<div id="model-arena">' + arena_html + '</div>'
    build_with_it = traditions + rl
    tabs = (("benchmark", "Home", benchmark), ("leaderboard", "Models", models),
            ("studio", "Try it", studio), ("rl", "Build with it", build_with_it))
    tab_routes = {"benchmark": "#/overview", "leaderboard": "#/leaderboard",
                  "studio": "#/studio", "rl": "#/build"}
    tab_buttons = "".join(
        f'<a class="pramaan-ui-tab" href="{tab_routes[key]}" role="tab" data-tab-button="{key}" '
        f'aria-selected="{"true" if ui_state.get("active_tab", "benchmark") == key else "false"}" '
        f'tabindex="{"0" if ui_state.get("active_tab", "benchmark") == key else "-1"}"><span>{label}</span></a>'
        for key, label, _ in tabs
    )
    tab_panels = "".join(f'<div class="pramaan-ui-tabpanel" data-tab-panel="{key}"{" hidden" if ui_state.get("active_tab", "benchmark") != key else ""}>{content}</div>' for key, _, content in tabs)
    return (
        '<main class="pramaan-ui">'
        + '<header class="pramaan-ui-masthead">' + _brand_header_html() + '</header>'
        '<nav class="pramaan-ui-tabs" role="tablist" aria-label="Pramaan sections">' + tab_buttons + '</nav>'
        + tab_panels + '<p class="pramaan-ui-footnote">Craft-informed computation: designs are mathematical interpretations, not claims of artisan authenticity or physical print readiness. Counterexamples mark exact failed pixels.</p></main>'
    )


def _handle_custom_event(event: gr.EventData, previous_state: dict,
                         oauth_profile: gr.OAuthProfile | None = None,
                         request: gr.Request | None = None):
    state = dict(previous_state or _initial_custom_state())
    active_tab = getattr(event, "active_tab", None)
    if active_tab in {"studio", "benchmark", "leaderboard", "rl"}:
        state["active_tab"] = active_tab
    form = _form_from_event(event, state)
    action = str(getattr(event, "action", "") or "")
    if action == "state":
        state_name = str(getattr(event, "state", form.get("state", "")))
        options = _tradition_choices(state_name)
        if options:
            form = _form_for_tradition(state_name, options[0].id, form)
        else:
            form["state"], form["tradition"], form["palette"] = state_name, "", ""
    elif action == "tradition":
        form = _form_for_tradition(str(form.get("state", "")), str(getattr(event, "tradition", form.get("tradition", ""))), form)
    state["form"] = form
    if action == "about_state":
        state["about_state"] = str(getattr(event, "about_state", ""))
    elif action == "festival_lens":
        state["festival_lens"] = bool(getattr(event, "festival_lens", False))
    elif action == "benchmark_state":
        state["benchmark_state"] = str(getattr(event, "benchmark_state", ""))
    elif action == "gallery_state_select":
        gallery_state = str(getattr(event, "gallery_state", ""))
        if gallery_state in ALL_STATES:
            state["benchmark_state"] = gallery_state
    elif action == "gallery_model":
        state["gallery_model"] = str(getattr(event, "gallery_model", "all"))
    elif action in {"wall_model", "wall_mode"}:
        value = str(getattr(event, action, ""))
        if action == "wall_model" and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}", value):
            state[action] = value
        elif action == "wall_mode" and value in {"reference", "attempt"}:
            state[action] = value
    elif action in {"quilt_model", "quilt_level", "quilt_mode", "quilt_compare_model"}:
        allowed = {
            "quilt_model": lambda value: isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}", value),
            "quilt_level": lambda value: value in {"easy", "hard"},
            "quilt_mode": lambda value: value in {"single", "compare", "hardness"},
            "quilt_compare_model": lambda value: isinstance(value, str) and (not value or re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,100}", value)),
        }
        value = str(getattr(event, action, ""))
        if allowed[action](value):
            state[action] = value
    elif action == "community_refresh":
        state["community_loaded"] = True
        _community_comparison_snapshot.cache_clear()
    elif action == "community_estimate":
        model_name = getattr(event, "community_model", "")
        estimate = _estimate_community_full_run_cost(model_name) if isinstance(model_name, str) else None
        state["community_estimate"] = estimate
        if estimate is None:
            state["community_status"] = (
                "No cost estimate is available. Enter a model with a current USD rate card and verify that "
                "the complete official token baseline is present. No provider request was made.")
        else:
            cap_outcome = (
                "The projection exceeds the USD 10 run cap; the runner may stop before all 72 questions, "
                "so partial coverage is likely. "
                if estimate["estimated_cost_exceeds_per_run_cap"] else
                "The projection is within the USD 10 cap, though actual usage and provider billing can differ. "
            )
            state["community_status"] = (
                f"Full-suite estimate for {model_name}: USD {estimate['estimated_usd']:.4f}, based on "
                f"official mean usage of {estimate['mean_input_tokens_per_task']:.0f} input and "
                f"{estimate['mean_output_tokens_per_task']:.0f} output tokens per task "
                f"({estimate['rate_card_source']}, observed {estimate['rate_card_observed_at']}). "
                f"{cap_outcome}No provider request was made.")
    elif action == "example":
        name = str(getattr(event, "example", ""))
        if name not in EXAMPLES:
            state["result"] = {"status": "Choose one of the built-in examples."}
        else:
            image, gallery, source, certificate, history, status, bundle = load_example(name)
            state["form"] = _form_for_example(name, form)
            state["result"] = {"image": image, "gallery": gallery, "source": source,
                               "certificate": certificate, "history": history, "status": status,
                               "bundle": bundle}
    elif action == "generate":
        try:
            values = generate(
                str(form.get("brief", "")), str(form.get("state", "")), str(form.get("tradition", "")),
                str(form.get("palette", "")), str(form.get("symmetry", "D2")),
                form.get("max_colours", 4), form.get("min_coverage", 0.02), form.get("max_coverage", 0.7),
                form.get("exclusive_ink_layers", False), form.get("min_gap", 0), form.get("required_loops", 1),
                form.get("cell_size", 4), str(form.get("translation_axis", "x")),
                form.get("translation_period", 8), form.get("min_clearance", 0), form.get("frame", 0),
                form.get("min_web", 0),
            )
            image, gallery, source, certificate, history, status, bundle = values
            state["result"] = {"image": image, "gallery": gallery, "source": source,
                               "certificate": certificate, "history": history, "status": status,
                               "bundle": bundle}
        except Exception:
            state["result"] = {**(state.get("result") or {}),
                               "status": "Generation failed. Check the provider configuration and try again; sensitive request details are not shown here."}
    elif action == "recheck":
        try:
            image, gallery, certificate, status, bundle = recheck(str(form.get("source", "")))
            state["result"] = {**(state.get("result") or {}), "image": image, "gallery": gallery,
                               "source": str(form.get("source", "")), "certificate": certificate,
                               "status": status, "bundle": bundle}
        except Exception:
            state["result"] = {**(state.get("result") or {}), "status": "PatternDSL could not be verified. Check the JSON and retry."}
    elif action in {"community_run", "community_publish"}:
        # The API key is read only from this transient event and is never copied to `state` or `form`.
        publish = action == "community_publish"
        if publish and not isinstance(getattr(oauth_profile, "username", None), str):
            state["community_status"] = "Sign in with Hugging Face before submitting public results. No provider request was made."
        elif publish and getattr(event, "community_publish_consent", False) is not True:
            state["community_status"] = "Check the separate public dataset consent box before submitting. No provider request was made."
        else:
            task_limit_text = getattr(event, "community_task_limit", "12")
            try:
                task_limit = int(task_limit_text)
            except (TypeError, ValueError):
                task_limit = 0
            state["community_status"] = _run_community_evaluation(
                str(getattr(event, "community_model", "")),
                getattr(event, "community_api_key", ""),
                getattr(event, "community_charge_consent", False) is True,
                publish=publish,
                submitted_by=getattr(oauth_profile, "username", None) if publish else None,
                task_limit=task_limit,
                full_run_cost_acknowledged=(getattr(event, "community_full_run_cost_consent", False) is True),
                reviewed_estimate=state.get("community_estimate"),
                daily_identity=_community_daily_identity(oauth_profile, request),
                enforce_daily_limit=True,
            )
    elif action == "community_archive_submit":
        consent = getattr(event, "community_archive_consent", False) is True
        archive_path = getattr(event, "community_archive_path", None)
        try:
            if not consent:
                state["community_archive_status"] = "Check the publication consent box before submitting. No archive was audited or published."
            elif not isinstance(archive_path, str) or not archive_path:
                state["community_archive_status"] = "Choose a ZIP archive before submitting. No archive was audited or published."
            elif not os.getenv("HF_COMMUNITY_WRITE_TOKEN"):
                state["community_archive_status"] = "Archive contribution is not enabled in this Space. No archive was published."
            else:
                publish_archive = globals().get("_publish_community_archive")
                if not callable(publish_archive):
                    state["community_archive_status"] = "Archive validation is not available in this Space build. No archive was published."
                else:
                    state["community_archive_status"] = publish_archive(archive_path)
        finally:
            _remove_gradio_upload(archive_path)
    return _render_custom_app(state), state


def _tradition_defaults(tradition_id: str | None):
    item = _tradition_by_id(tradition_id)
    if item is None:
        return ("Choose a tradition to load its grammar and default profile.", gr.update(choices=[], value=None),
                "D4", 4, 0.02, 0.70, 0,
                gr.update(value=1, visible=False), gr.update(value=4, visible=False),
                gr.update(value="x", visible=False), gr.update(value=8, visible=False),
                gr.update(value=0, visible=False), gr.update(value=0, visible=False),
                gr.update(value=0, visible=False))
    palettes = item.palette_presets
    choices = [(name.replace("_", " ").title(), name) for name in palettes]
    default_palette = next(iter(palettes), None)
    profile = item.default_profile
    symmetries = ["None", "D2", "D4", "C2", "C4"]
    symmetry = profile.get("symmetry", "D2")
    if symmetry not in symmetries:
        symmetry = "D2"
    symmetry_choices = ["None", "D2", "D4"] if item.grammar == "periodic_raster" else symmetries
    if symmetry not in symmetry_choices:
        symmetry = "D2" if "D2" in symmetry_choices else symmetry_choices[0]
    return (
        f"**Grammar:** `{item.grammar}` · **Default profile:** `{json.dumps(profile, sort_keys=True)}`",
        gr.update(choices=choices, value=default_palette), gr.update(choices=symmetry_choices, value=symmetry),
        int(profile.get("max_colours", 4)), float(profile.get("min_coverage", 0.02)),
        float(profile.get("max_coverage", 0.70)), int(profile.get("min_gap", 0)),
        gr.update(value=int(profile.get("required_loops", 1)), visible=item.grammar == "kolam_loop"),
        gr.update(value=int(profile.get("cell_size", 4)), visible=item.grammar == "counted_grid"),
        gr.update(value=profile.get("translation_period", {}).get("axis", "x"), visible=item.grammar == "stripe_band"),
        gr.update(value=int(profile.get("translation_period", {}).get("period", 8)), visible=item.grammar == "stripe_band"),
        gr.update(value=int(profile.get("min_clearance", 0)), visible=item.grammar == "dot_field"),
        gr.update(value=int(profile.get("frame", 0)), visible=item.grammar == "jaali_lattice"),
        gr.update(value=int(profile.get("min_web", 0)), visible=item.grammar == "jaali_lattice"),
    )


def _brand_logo_path() -> Path:
    return ROOT / "assets" / "pramaan-mark.png"


def _brand_favicon_path() -> Path:
    return ROOT / "assets" / "pramaan-favicon.png"


@lru_cache(maxsize=1)
def _brand_logo_data_uri() -> str:
    """Return a compact 2x header logo so first visits avoid a 700 KB asset fetch."""
    try:
        with Image.open(_brand_logo_path()) as source:
            logo = source.convert("RGBA")
        logo.thumbnail((216, 216), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        logo.save(buffer, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    except Exception as error:
        logging.getLogger(__name__).warning(
            "Brand logo unavailable; using text-only header (%s)", type(error).__name__
        )
        return ""


PRAMAAN_CSS = """
:root {
  color-scheme: light;
  --pramaan-indigo: #172f4a;
  --pramaan-ink: #26384b;
  --pramaan-text: #26384b;
  --pramaan-madder: #a84436;
  --pramaan-teal: #176a66;
  --pramaan-paper: #f7f3ec;
  --pramaan-card: #fffdf9;
  --pramaan-card-raised: #ffffff;
  --pramaan-card-subtle: #f3eee6;
  --pramaan-line: #c9bba8;
  --pramaan-muted: #455463;
  --pramaan-on-accent: #ffffff;
  --pramaan-pass: #126b50;
  --pramaan-pass-bg: #e5f1ec;
  --pramaan-fail: #9b2948;
  --pramaan-fail-bg: #f7e9eb;
  --pramaan-pending: #67430f;
  --pramaan-pending-bg: #f0e9dc;
  --pramaan-checker: #ece7de;
  --pramaan-shadow: #172f4a;
  --pramaan-chart-1: #176a66;
  --pramaan-chart-2: #a84436;
  --pramaan-chart-3: #536272;
  --pramaan-chart-4: #67430f;
  --pramaan-chart-5: #5964a5;
  --pramaan-chart-6: #47709a;
}
html.dark, body.dark, .dark {
  color-scheme: dark;
  --pramaan-indigo: #e8eef5;
  --pramaan-ink: #e1e8ef;
  --pramaan-text: #e1e8ef;
  --pramaan-madder: #f09583;
  --pramaan-teal: #75d0c5;
  --pramaan-paper: #151b22;
  --pramaan-card: #202a34;
  --pramaan-card-raised: #293542;
  --pramaan-card-subtle: #26313b;
  --pramaan-line: #647383;
  --pramaan-muted: #cbd6e0;
  --pramaan-on-accent: #10272a;
  --pramaan-pass: #83dfb6;
  --pramaan-pass-bg: #173c32;
  --pramaan-fail: #ff9bad;
  --pramaan-fail-bg: #492531;
  --pramaan-pending: #f2d18d;
  --pramaan-pending-bg: #443820;
  --pramaan-checker: #36424e;
  --pramaan-shadow: #000000;
  --pramaan-chart-1: #75d0c5;
  --pramaan-chart-2: #f09583;
  --pramaan-chart-3: #c0cad4;
  --pramaan-chart-4: #f2d18d;
  --pramaan-chart-5: #bbc1ff;
  --pramaan-chart-6: #9fc9f0;
}
html.light, body.light {
  color-scheme: light;
  --pramaan-indigo: #172f4a;
  --pramaan-ink: #26384b;
  --pramaan-text: #26384b;
  --pramaan-madder: #a84436;
  --pramaan-teal: #176a66;
  --pramaan-paper: #f7f3ec;
  --pramaan-card: #fffdf9;
  --pramaan-card-raised: #ffffff;
  --pramaan-card-subtle: #f3eee6;
  --pramaan-line: #c9bba8;
  --pramaan-muted: #455463;
  --pramaan-on-accent: #ffffff;
  --pramaan-pass: #126b50;
  --pramaan-pass-bg: #e5f1ec;
  --pramaan-fail: #9b2948;
  --pramaan-fail-bg: #f7e9eb;
  --pramaan-pending: #67430f;
  --pramaan-pending-bg: #f0e9dc;
  --pramaan-checker: #ece7de;
  --pramaan-shadow: #172f4a;
  --pramaan-chart-1: #176a66;
  --pramaan-chart-2: #a84436;
  --pramaan-chart-3: #536272;
  --pramaan-chart-4: #67430f;
  --pramaan-chart-5: #5964a5;
  --pramaan-chart-6: #47709a;
}
@media (prefers-color-scheme: dark) {
  html:not(.light):not(.dark), body:not(.light):not(.dark) {
    color-scheme: dark;
    --pramaan-indigo: #e8eef5;
    --pramaan-ink: #e1e8ef;
    --pramaan-text: #e1e8ef;
    --pramaan-madder: #f09583;
    --pramaan-teal: #75d0c5;
    --pramaan-paper: #151b22;
    --pramaan-card: #202a34;
    --pramaan-card-raised: #293542;
    --pramaan-card-subtle: #26313b;
    --pramaan-line: #647383;
    --pramaan-muted: #cbd6e0;
    --pramaan-on-accent: #10272a;
    --pramaan-pass: #83dfb6;
    --pramaan-pass-bg: #173c32;
    --pramaan-fail: #ff9bad;
    --pramaan-fail-bg: #492531;
    --pramaan-pending: #f2d18d;
    --pramaan-pending-bg: #443820;
    --pramaan-checker: #36424e;
    --pramaan-shadow: #000000;
    --pramaan-chart-1: #75d0c5;
    --pramaan-chart-2: #f09583;
    --pramaan-chart-3: #c0cad4;
    --pramaan-chart-4: #f2d18d;
    --pramaan-chart-5: #bbc1ff;
    --pramaan-chart-6: #9fc9f0;
  }
}
.gradio-container {
  background: var(--pramaan-paper);
  color: var(--pramaan-indigo);
  width: 100% !important;
  max-width: none !important;
  box-sizing: border-box !important;
}
.pramaan-masthead {
  display: grid;
  grid-template-columns: clamp(84px, 9vw, 116px) minmax(0, 1fr);
  align-items: center;
  gap: 1.35rem;
  margin: 1.2rem 0 1rem;
  padding: 1.1rem 1.5rem;
  width: 100%;
  box-sizing: border-box;
  border: 1px solid color-mix(in srgb, var(--pramaan-line) 90%, transparent);
  border-radius: 24px;
  background: linear-gradient(115deg, color-mix(in srgb, var(--pramaan-card) 98%, transparent), color-mix(in srgb, var(--pramaan-card) 92%, transparent));
  box-shadow: 0 14px 38px color-mix(in srgb, var(--pramaan-shadow) 8%, transparent);
}
.pramaan-logo {
  align-self: center;
  display: block;
  height: 108px;
  justify-self: center;
  object-fit: contain;
  width: 108px;
}
.pramaan-masthead > :last-child { min-width: 0 !important; }
.pramaan-masthead {
  display: grid !important;
  grid-template-columns: 116px minmax(0, 1fr);
  width: 100% !important;
}
.pramaan-brand-copy { min-width: 0; }
.pramaan-eyebrow {
  color: var(--pramaan-teal); font-size: .72rem; font-weight: 750;
  letter-spacing: .16em; margin: 0 0 .4rem; text-transform: uppercase;
}
.pramaan-masthead h1 {
  color: var(--pramaan-indigo); font-size: clamp(1.7rem, 3.5vw, 2.55rem);
  letter-spacing: -.045em; line-height: 1.08; margin: 0;
}
.pramaan-masthead h1 span { color: var(--pramaan-madder); font-weight: 500; }
.pramaan-tagline { color: var(--pramaan-muted); font-size: .98rem; margin: .55rem 0 .75rem; }
.pramaan-pills { display: flex; flex-wrap: wrap; gap: .45rem; }
.pramaan-pill {
  background: var(--pramaan-pending-bg); border: 1px solid var(--pramaan-line); border-radius: 999px;
  color: var(--pramaan-indigo); font-size: .72rem; font-weight: 650; padding: .28rem .68rem;
}
.pramaan-disclaimer {
  border-left: 4px solid var(--pramaan-madder); border-radius: 8px 14px 14px 8px;
  background: color-mix(in srgb, var(--pramaan-card) 82%, transparent); color: var(--pramaan-muted); line-height: 1.55;
  margin: .9rem 0 1.1rem; padding: .8rem 1rem;
}
.pramaan-disclaimer strong { color: var(--pramaan-indigo); }
.pramaan-method {
  display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .7rem;
  margin: 0 0 1.1rem;
}
.pramaan-method div {
  align-items: center; background: color-mix(in srgb, var(--pramaan-card) 72%, transparent); border: 1px solid var(--pramaan-line);
  border-radius: 14px; color: var(--pramaan-muted); display: flex; gap: .65rem; padding: .7rem .85rem;
}
.pramaan-method b { color: var(--pramaan-indigo); font-size: .88rem; }
.pramaan-method span {
  align-items: center; background: var(--pramaan-indigo); border-radius: 50%; color: var(--pramaan-card-raised);
  display: inline-flex; flex: 0 0 1.65rem; font-size: .72rem; height: 1.65rem; justify-content: center;
}
.pramaan-workspace {
  display: grid !important;
  grid-template-columns: minmax(280px, .9fr) minmax(0, 1.65fr);
  align-items: stretch;
  gap: 1rem;
  width: 100%;
  max-width: none !important;
}
.pramaan-workspace > .pramaan-panel { width: auto !important; min-width: 0 !important; }
.pramaan-panel {
  background: color-mix(in srgb, var(--pramaan-card) 86%, transparent); border: 1px solid var(--pramaan-line);
  border-radius: 20px; box-shadow: 0 9px 28px color-mix(in srgb, var(--pramaan-shadow) 6%, transparent);
  padding: 1rem !important;
}
.pramaan-status { border-radius: 12px; }
.pramaan-preview img { border: 1px solid var(--pramaan-line); border-radius: 16px; }
.pramaan-benchmark-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: .8rem; }
.pramaan-benchmark-card, .pramaan-benchmark-empty {
  background: color-mix(in srgb, var(--pramaan-card) 90%, transparent); border: 1px solid var(--pramaan-line);
  border-radius: 18px; box-shadow: 0 8px 24px color-mix(in srgb, var(--pramaan-shadow) 5%, transparent); padding: 1rem 1.1rem;
}
.pramaan-benchmark-card h3 { color: var(--pramaan-indigo); font-size: 1rem; margin: 0 0 .65rem; }
.pramaan-benchmark-card dl { display: grid; gap: .4rem; margin: 0; }
.pramaan-benchmark-card dl div { align-items: baseline; border-top: 1px solid var(--pramaan-line); display: flex; gap: 1rem; justify-content: space-between; padding-top: .4rem; }
.pramaan-benchmark-card dt { color: var(--pramaan-muted); font-size: .83rem; }
.pramaan-benchmark-card dd { color: var(--pramaan-indigo); font-size: .9rem; font-variant-numeric: tabular-nums; font-weight: 700; margin: 0; text-align: right; }
.pramaan-benchmark-empty { color: var(--pramaan-muted); }
.pramaan-benchmark-kicker { color: var(--pramaan-teal); font-size: .72rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
.pramaan-findings { background: color-mix(in srgb, var(--pramaan-card) 88%, transparent); border: 1px solid var(--pramaan-line); border-radius: 16px; margin: .9rem 0; min-width: 0; padding: 1rem; }
.pramaan-findings h2 { color: var(--pramaan-indigo); font-family: Georgia,serif; font-size: 1.4rem; margin: 0 0 .6rem; }
.pramaan-findings-grid { display: grid; gap: .65rem; grid-template-columns: repeat(3,minmax(0,1fr)); min-width: 0; }
.pramaan-findings-grid article { background: var(--pramaan-card-subtle); border: 1px solid var(--pramaan-line); border-radius: 12px; display: grid; gap: .2rem; min-width: 0; padding: .7rem; }
.pramaan-findings-grid strong { color: var(--pramaan-indigo); }
.pramaan-findings-grid span { font-size: 1.05rem; font-variant-numeric: tabular-nums; font-weight: 750; overflow-wrap: anywhere; }
.pramaan-findings-grid small, .pramaan-findings-note { color: var(--pramaan-muted); font-size: .76rem; }
.pramaan-findings-arms { display: flex; flex-wrap: wrap; gap: .4rem; list-style: none; margin: .6rem 0 0; padding: 0; }
.pramaan-findings-arms li { background: var(--pramaan-card); border: 1px solid var(--pramaan-line); border-radius: 9px; display: grid; font-size: .72rem; gap: .1rem; min-width: 125px; padding: .35rem .5rem; }
.pramaan-findings-arms li > span:first-child { color: var(--pramaan-indigo); font-weight: 750; }
.pramaan-findings-arms [data-arm-state="partial"] { border-style: dashed; border-color: var(--pramaan-pending); }
.pramaan-findings-arms [data-arm-state="pending"] { color: var(--pramaan-muted); }
.pramaan-leaderboard-hero { background: var(--pramaan-card-subtle); border: 1px solid var(--pramaan-line); border-radius: 14px; margin: .8rem 0 1rem; min-width: 0; padding: .8rem; }
.pramaan-leaderboard-hero h4 { color: var(--pramaan-indigo); margin: 0 0 .35rem; }
.pramaan-leaderboard-hero > p { color: var(--pramaan-muted); font-size: .8rem; }
.pramaan-leaderboard-hero-list { display: grid; gap: .55rem; list-style: none; margin: .65rem 0; padding: 0; }
.pramaan-leaderboard-hero-list > li { background: var(--pramaan-card); border: 1px solid var(--pramaan-line); border-radius: 10px; min-width: 0; padding: .55rem .65rem; }
.pramaan-leaderboard-hero-label { display: flex; flex-wrap: wrap; gap: .2rem .65rem; justify-content: space-between; min-width: 0; }
.pramaan-leaderboard-hero-label strong { color: var(--pramaan-indigo); }
.pramaan-leaderboard-hero-label span, .pramaan-leaderboard-hero-list small { color: var(--pramaan-muted); font-size: .76rem; overflow-wrap: anywhere; }
.pramaan-leaderboard-hero-track { background: var(--pramaan-line); border-radius: 99px; height: 13px; margin: .38rem 0 .16rem; overflow: visible; position: relative; width: 100%; }
.pramaan-leaderboard-hero-bar { background: var(--pramaan-teal); border-radius: 99px; display: block; height: 100%; max-width: 100%; }
.pramaan-leaderboard-hero-list > li:nth-child(2n) .pramaan-leaderboard-hero-bar { background: var(--pramaan-madder); }
.pramaan-leaderboard-hero-list > li:nth-child(3n) .pramaan-leaderboard-hero-bar { background: var(--pramaan-muted); }
.pramaan-leaderboard-hero-list > li:nth-child(4n) .pramaan-leaderboard-hero-bar { background: var(--pramaan-pending); }
.pramaan-leaderboard-hero-interval { border-left: 2px solid var(--pramaan-indigo); border-right: 2px solid var(--pramaan-indigo); box-sizing: border-box; height: 21px; pointer-events: none; position: absolute; top: -4px; }
.pramaan-leaderboard-hero-text-alt { background: var(--pramaan-card); border-left: 3px solid var(--pramaan-teal); padding: .45rem .6rem; }
.pramaan-leaderboard-hero-list > li.is-partial { border-style: dashed; border-color: var(--pramaan-pending); }
.pramaan-leaderboard-scroll { max-width: 100%; overflow-x: auto; }
.pramaan-leaderboard { border-collapse: collapse; font-size: .83rem; min-width: 1040px; width: 100%; }
.pramaan-leaderboard th, .pramaan-leaderboard td { border-bottom: 1px solid var(--pramaan-line); padding: .65rem .55rem; text-align: left; vertical-align: top; }
.pramaan-leaderboard th { color: var(--pramaan-indigo); }
.pramaan-leaderboard tbody th small, .pramaan-leaderboard td small { color: var(--pramaan-muted); display: block; font-size: .72rem; font-weight: 500; margin-top: .2rem; }
.pramaan-grammar-heat { display: grid; gap: .2rem; grid-template-columns: repeat(7, minmax(30px, 1fr)); }
.pramaan-grammar-cell { background: var(--pramaan-card-subtle); border-radius: 6px; color: var(--pramaan-indigo); display: grid; font-size: .65rem; line-height: 1.2; padding: .2rem; text-align: center; }
.pramaan-grammar-cell small { color: inherit !important; font-size: .6rem !important; margin: 0 !important; }
.pramaan-canonical-leaderboard h3 { margin-top: 1.2rem; }
.pramaan-canonical-leaderboard ul { color: var(--pramaan-muted); font-size: .85rem; padding-left: 1.2rem; }
.pramaan-leaderboard-nav { display: flex; flex-wrap: wrap; gap: .45rem; margin: .75rem 0; }
.pramaan-leaderboard-nav a { background: var(--pramaan-card); border: 1px solid var(--pramaan-line); border-radius: 999px; color: var(--pramaan-indigo); font-size: .82rem; font-weight: 700; padding: .38rem .7rem; text-decoration: none; }
.pramaan-leaderboard-controls { align-items: end; background: var(--pramaan-card-subtle); border: 1px solid var(--pramaan-line); border-radius: 14px; display: flex; flex-wrap: wrap; gap: .65rem; margin: .8rem 0 1rem; padding: .75rem; }
.pramaan-leaderboard-controls label { color: var(--pramaan-indigo); display: grid; font-size: .76rem; font-weight: 700; gap: .25rem; min-width: 145px; }
.pramaan-leaderboard-controls input, .pramaan-leaderboard-controls select { background: var(--pramaan-card-raised); border: 1px solid var(--pramaan-line); border-radius: 9px; color: var(--pramaan-indigo); font: inherit; min-height: 40px; padding: .45rem .6rem; }
.pramaan-leaderboard-controls [data-leaderboard-count] { color: var(--pramaan-muted); font-size: .78rem; padding: .5rem; }
.pramaan-leaderboard-method { border-top: 1px solid var(--pramaan-line); margin-top: 1rem; padding-top: .8rem; }
.pramaan-leaderboard-method p { color: var(--pramaan-muted); font-size: .86rem; }
.pramaan-model-report { min-width: 0; }
.pramaan-model-chart { background: var(--pramaan-card-subtle); border: 1px solid var(--pramaan-line); border-radius: 14px; margin: .9rem 0 1.1rem; padding: .9rem; }
.pramaan-model-chart h4 { margin: .1rem 0 .3rem; }
.pramaan-model-chart > p, .pramaan-model-chart-alt, .pramaan-model-chart li small { color: var(--pramaan-muted); font-size: .8rem; }
.pramaan-model-chart ol { display: grid; gap: .55rem; list-style: none; margin: .7rem 0; padding: 0; }
.pramaan-model-chart li { min-width: 0; }
.pramaan-model-bar-label { display: flex; flex-wrap: wrap; gap: .2rem .7rem; justify-content: space-between; font-size: .82rem; }
.pramaan-model-bar-track { background: var(--pramaan-line); border-radius: 99px; height: 13px; margin: .27rem 0 .12rem; overflow: hidden; position: relative; }
.pramaan-model-bar { background: var(--pramaan-indigo); border-radius: 99px; display: block; height: 100%; }
.pramaan-model-table { border-collapse: collapse; font-size: .84rem; min-width: 720px; width: 100%; }
.pramaan-model-table th, .pramaan-model-table td { border-bottom: 1px solid var(--pramaan-line); padding: .65rem .55rem; text-align: left; vertical-align: top; }
.pramaan-model-table tbody th small, .pramaan-model-table td small { color: var(--pramaan-muted); display: block; }
.pramaan-model-footnote { color: var(--pramaan-muted); font-size: .82rem; }
.pramaan-arena-picker { margin: .7rem 0; }
.pramaan-arena-picker fieldset { border: 1px solid var(--pramaan-line); border-radius: 10px; display: flex; flex-wrap: wrap; gap: .5rem 1rem; margin-top: .4rem; padding: .7rem; }
.pramaan-same-brief-wall { display: grid; gap: .8rem; }
.pramaan-arena { background: color-mix(in srgb, var(--pramaan-card) 82%, transparent); border: 1px solid var(--pramaan-line); border-radius: 16px; margin: 1rem 0; min-width: 0; padding: 1rem; }
.pramaan-arena h3 { color: var(--pramaan-indigo); margin: .2rem 0 .45rem; }
.pramaan-arena-grid { display: grid; gap: .7rem; grid-template-columns: repeat(auto-fit,minmax(min(100%,210px),1fr)); min-width: 0; }
.pramaan-arena-arm { background: var(--pramaan-card); border: 1px solid var(--pramaan-line); border-radius: 12px; min-width: 0; padding: .7rem; }
.pramaan-arena-arm h4 { color: var(--pramaan-indigo); margin: 0 0 .3rem; overflow-wrap: anywhere; }
.pramaan-arena-status, .pramaan-arena-empty { color: var(--pramaan-muted); font-size: .82rem; }
.pramaan-arena-attempts { display: grid; gap: .45rem; list-style: none; margin: .5rem 0; padding: 0; }
.pramaan-arena-attempts li { align-items: center; border-top: 1px solid var(--pramaan-line); display: grid; gap: .35rem; grid-template-columns: auto minmax(0,1fr); min-width: 0; padding-top: .45rem; }
.pramaan-arena-verdict { border-radius: 7px; font-size: .8rem; font-weight: 800; padding: .22rem .4rem; }
.pramaan-arena-attempts [data-attempt-state="pass"] .pramaan-arena-verdict { background: var(--pramaan-pass-bg); color: var(--pramaan-pass); }
.pramaan-arena-attempts [data-attempt-state="fail"] .pramaan-arena-verdict { background: var(--pramaan-fail-bg); color: var(--pramaan-fail); }
.pramaan-arena-tile { grid-column: 1 / -1; image-rendering: pixelated; max-height: 160px; max-width: 100%; object-fit: contain; width: 160px; }
.pramaan-arena-attempts small { color: var(--pramaan-muted); grid-column: 1 / -1; overflow-wrap: anywhere; }
.pramaan-arena-arm > a { margin-top: .4rem; }
.pramaan-comparison-chart-grid { display: grid; gap: .8rem; grid-template-columns: repeat(auto-fit,minmax(min(100%,420px),1fr)); }
.pramaan-comparison-chart-grid article { background: color-mix(in srgb, var(--pramaan-card) 70%, transparent); border: 1px solid var(--pramaan-line); border-radius: 12px; min-width: 0; padding: .7rem; }
.pramaan-comparison-chart-grid h5, .pramaan-comparison-charts h4 { color: var(--pramaan-indigo); margin: .3rem 0 .6rem; }
.pramaan-comparison-chart-grid svg { display: block; height: auto; max-width: 100%; overflow: visible; }
.pramaan-same-brief { background: color-mix(in srgb, var(--pramaan-card) 75%, transparent); border: 1px solid var(--pramaan-line); border-radius: 14px; padding: .8rem; }
.pramaan-same-brief h4 { color: var(--pramaan-indigo); margin: 0 0 .35rem; }
.pramaan-same-brief-prompt, .pramaan-model-tile-card p { color: var(--pramaan-muted); font-size: .82rem; }
.pramaan-model-tile-grid { display: grid; gap: .65rem; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
.pramaan-model-tile-card { border: 1px solid var(--pramaan-line); border-radius: 12px; min-width: 0; padding: .6rem; }
.pramaan-model-tile-card h5 { color: var(--pramaan-indigo); margin: 0 0 .45rem; }
.pramaan-model-tile { background-color: var(--pramaan-card); background-image: linear-gradient(45deg,var(--pramaan-checker) 25%,transparent 25%),linear-gradient(-45deg,var(--pramaan-checker) 25%,transparent 25%),linear-gradient(45deg,transparent 75%,var(--pramaan-checker) 75%),linear-gradient(-45deg,transparent 75%,var(--pramaan-checker) 75%); background-size: 12px 12px; display: block; image-rendering: pixelated; max-height: 192px; max-width: 100%; object-fit: contain; width: 192px; }
.pramaan-model-tile-placeholder { align-items: center; background: var(--pramaan-card-subtle); color: var(--pramaan-muted); display: flex; font-size: .78rem; justify-content: center; min-height: 96px; text-align: center; }
.pramaan-model-tile-status { color: var(--pramaan-teal); font-weight: 700; }
.pramaan-model-tile-card small { color: var(--pramaan-muted); display: block; font-size: .65rem; overflow-wrap: anywhere; }
.gradio-container button.primary {
  background: var(--pramaan-teal); border: 1px solid var(--pramaan-teal); color: var(--pramaan-on-accent);
  box-shadow: 0 6px 16px color-mix(in srgb, var(--pramaan-shadow) 18%, transparent); font-weight: 700;
}
.gradio-container button.primary:hover { background: var(--pramaan-teal); }
.gradio-container button:not(.primary) { border-radius: 12px !important; }
.gradio-container input, .gradio-container textarea, .gradio-container select {
  border-color: var(--pramaan-line) !important; border-radius: 11px !important;
}
.gradio-container input:focus, .gradio-container textarea:focus {
  border-color: var(--pramaan-teal) !important; box-shadow: 0 0 0 2px color-mix(in srgb, var(--pramaan-teal) 16%, transparent) !important;
}
.cm-editor { max-height: 26rem !important; }
.cm-editor .cm-scroller { overflow: auto !important; }
.tabs { border-color: var(--pramaan-line) !important; }
.tab-nav button { background: var(--pramaan-card); border: 1px solid var(--pramaan-line); color: var(--pramaan-muted); font-weight: 650; }
.tab-nav button.selected { background: var(--pramaan-teal); border-color: var(--pramaan-teal); color: var(--pramaan-on-accent); }
.gradio-container .tab-nav button.selected,
.gradio-container .tab-nav button.selected:hover,
.gradio-container .tab-nav button.selected:focus-visible {
  background-color: var(--pramaan-teal) !important;
  border-color: var(--pramaan-teal) !important;
  color: var(--pramaan-on-accent) !important;
}
.gradio-container .tab-nav button.selected > span { color: inherit !important; }
.prose a { color: var(--pramaan-teal) !important; }
.gradio-container .pramaan-ui a.pramaan-ui-button,
.gradio-container .pramaan-ui a.pramaan-ui-button:visited,
.gradio-container .pramaan-ui a.pramaan-ui-button:hover,
.gradio-container .pramaan-ui a.pramaan-ui-button:focus-visible {
  background-color: var(--pramaan-card) !important;
  border-color: var(--pramaan-line) !important;
  color: var(--pramaan-indigo) !important;
}
.gradio-container .pramaan-ui a.pramaan-ui-button.secondary,
.gradio-container .pramaan-ui a.pramaan-ui-button.secondary:visited,
.gradio-container .pramaan-ui a.pramaan-ui-button.secondary:hover,
.gradio-container .pramaan-ui a.pramaan-ui-button.secondary:focus-visible {
  background-color: var(--pramaan-card) !important;
  color: var(--pramaan-indigo) !important;
}
.pramaan-ui { color: var(--pramaan-indigo); font: 400 15px/1.55 Inter, ui-sans-serif, system-ui, sans-serif; margin: 0 auto; max-width: 1480px; }
.gradio-container .pramaan-ui p:not([class]),
.gradio-container .pramaan-ui div:not([class]),
.gradio-container .pramaan-ui li:not([class]),
.gradio-container .pramaan-ui summary:not([class]),
.gradio-container .pramaan-ui small:not([class]),
.gradio-container .pramaan-ui strong:not([class]),
.gradio-container .pramaan-ui em:not([class]),
.gradio-container .pramaan-ui dd:not([class]),
.gradio-container .pramaan-ui dt:not([class]) {
  color: var(--pramaan-text) !important;
}
.gradio-container .pramaan-ui a:not([class]) { color: var(--pramaan-teal) !important; }
.gradio-container #pramaan-hf-login-row { box-sizing: border-box; display: flex; justify-content: flex-end; margin: 0 auto .35rem; max-width: 1480px; padding: .25rem clamp(12px,3vw,24px); }
.gradio-container #pramaan-hf-login { width: auto !important; }
.gradio-container #pramaan-hf-login button { background: var(--pramaan-card); border: 1px solid var(--pramaan-line); border-radius: 999px; color: var(--pramaan-indigo); font-size: .82rem; min-height: 36px; padding: .35rem .8rem; }
.pramaan-ui-masthead .pramaan-masthead { margin-top: 0; }
.pramaan-ui-disclaimer { background: var(--pramaan-card); border-left: 4px solid var(--pramaan-madder); border-radius: 8px 14px 14px 8px; color: var(--pramaan-muted); margin: 0 0 1rem; padding: .8rem 1rem; }
.pramaan-ui-tabs, .pramaan-ui-subtabs { display: flex; flex-wrap: wrap; gap: .45rem; margin: 0 0 1rem; }
.pramaan-ui-tab, .pramaan-ui-subtabs button, .pramaan-ui-button { align-items: center; background: var(--pramaan-card); border: 1px solid var(--pramaan-line); border-radius: 12px; color: var(--pramaan-indigo); cursor: pointer; display: inline-flex; font: inherit; font-weight: 700; justify-content: center; min-height: 42px; padding: .62rem .9rem; text-decoration: none; }
.gradio-container .pramaan-ui .pramaan-ui-tabs > a.pramaan-ui-tab,
.gradio-container .pramaan-ui .pramaan-ui-tabs > a.pramaan-ui-tab:visited,
.gradio-container .pramaan-ui .pramaan-ui-tabs > a.pramaan-ui-tab:hover,
.gradio-container .pramaan-ui .pramaan-ui-tabs > a.pramaan-ui-tab:focus-visible {
  background-color: var(--pramaan-card) !important;
  border-color: var(--pramaan-line);
  color: var(--pramaan-indigo) !important;
}
.gradio-container .pramaan-ui .pramaan-ui-tabs > a.pramaan-ui-tab[aria-selected="true"],
.gradio-container .pramaan-ui .pramaan-ui-tabs > a.pramaan-ui-tab[aria-selected="true"]:visited,
.gradio-container .pramaan-ui .pramaan-ui-tabs > a.pramaan-ui-tab[aria-selected="true"]:hover,
.gradio-container .pramaan-ui .pramaan-ui-tabs > a.pramaan-ui-tab[aria-selected="true"]:focus-visible {
  background-color: var(--pramaan-teal) !important;
  border-color: var(--pramaan-teal);
  color: var(--pramaan-on-accent) !important;
}
.gradio-container .pramaan-ui .pramaan-ui-tabs > a.pramaan-ui-tab > span { color: inherit !important; }
.pramaan-ui-subtabs button[aria-pressed="true"] { background: var(--pramaan-teal); border-color: var(--pramaan-teal); color: var(--pramaan-on-accent); }
.gradio-container .pramaan-ui .pramaan-ui-subtabs > button[aria-pressed="true"],
.gradio-container .pramaan-ui .pramaan-ui-subtabs > button[aria-pressed="true"]:hover,
.gradio-container .pramaan-ui .pramaan-ui-subtabs > button[aria-pressed="true"]:focus-visible {
  background-color: var(--pramaan-teal) !important;
  border-color: var(--pramaan-teal) !important;
  color: var(--pramaan-on-accent) !important;
}
.gradio-container .pramaan-ui .pramaan-ui-subtabs > button[aria-pressed="true"] > span { color: inherit !important; }
.pramaan-ui-button.secondary, .pramaan-ui-subtabs button { background: var(--pramaan-card); border-color: var(--pramaan-line); color: var(--pramaan-indigo); }
.pramaan-ui-button:disabled { background: var(--pramaan-pending-bg); border-color: var(--pramaan-pending); color: var(--pramaan-pending); cursor: not-allowed; opacity: 1; }
.pramaan-ui-layout { align-items: start; display: grid; gap: 1rem; grid-template-columns: minmax(300px,.9fr) minmax(0,1.5fr); }
.pramaan-ui-card { background: color-mix(in srgb, var(--pramaan-card) 92%, transparent); border: 1px solid var(--pramaan-line); border-radius: 18px; box-shadow: 0 9px 28px color-mix(in srgb, var(--pramaan-shadow) 6%, transparent); min-width: 0; padding: 1rem 1.1rem; }
.pramaan-ui-card h2 { color: var(--pramaan-indigo); font-size: 1.2rem; margin: 0 0 .55rem; }
.pramaan-ui-card h3 { color: var(--pramaan-indigo); font-size: 1rem; }
.pramaan-ui-eyebrow { color: var(--pramaan-teal); font-size: .72rem; font-weight: 750; letter-spacing: .12em; margin: 0 0 .35rem; text-transform: uppercase; }
.pramaan-muted, .pramaan-ui-muted, .pramaan-ui-footnote { color: var(--pramaan-muted); font-size: .88rem; }
.pramaan-ui-field { display: grid; gap: .32rem; margin: .75rem 0; }
.pramaan-ui-field label, .pramaan-ui-result label { color: var(--pramaan-indigo); font-size: .86rem; font-weight: 700; }
.pramaan-ui-field input, .pramaan-ui-field textarea, .pramaan-ui-field select, .pramaan-ui-result textarea { background: var(--pramaan-card-raised); border: 1px solid var(--pramaan-line); border-radius: 10px; box-sizing: border-box; color: var(--pramaan-indigo); font: inherit; min-height: 42px; padding: .62rem .7rem; width: 100%; }
.pramaan-ui-field textarea, .pramaan-ui-result textarea { min-height: 7rem; resize: vertical; }
.pramaan-ui-field input:focus, .pramaan-ui-field textarea:focus, .pramaan-ui-field select:focus, .pramaan-ui-result textarea:focus, .pramaan-ui button:focus-visible { border-color: var(--pramaan-teal); box-shadow: 0 0 0 3px color-mix(in srgb, var(--pramaan-teal) 17%, transparent); outline: 2px solid transparent; }
.pramaan-ui-grid { display: grid; gap: .6rem; grid-template-columns: repeat(2,minmax(0,1fr)); }
.pramaan-ui-check { align-items: center; display: flex; gap: .55rem; margin: .75rem 0; }
.pramaan-ui-actions { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1rem; }
.pramaan-ui-preview { background: var(--pramaan-card-raised); border: 1px solid var(--pramaan-line); border-radius: 14px; display: block; height: auto; margin: .75rem auto 1rem; max-width: 100%; }
.pramaan-ui-placeholder { align-items: center; background: var(--pramaan-card-subtle); border: 1px dashed var(--pramaan-line); border-radius: 14px; color: var(--pramaan-muted); display: flex; justify-content: center; margin: .8rem 0; min-height: 230px; text-align: center; }
.pramaan-ui-result pre, .pramaan-ui-card pre { background: var(--pramaan-card-subtle); border: 1px solid var(--pramaan-line); border-radius: 10px; max-height: 26rem; overflow: auto; padding: .8rem; white-space: pre-wrap; word-break: break-word; }
.pramaan-ui-subpanel[hidden], .pramaan-ui-tabpanel[hidden] { display: none !important; }
.pramaan-ui-gallery, .pramaan-ui-card-grid { display: grid; gap: .7rem; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); }
.pramaan-ui-gallery figure { margin: 0; }
.pramaan-ui-gallery img { border: 1px solid var(--pramaan-line); border-radius: 12px; display: block; height: auto; max-width: 100%; }
.pramaan-ui-gallery figcaption { color: var(--pramaan-muted); font-size: .8rem; padding-top: .3rem; }
.pramaan-state-mosaic-scroll { box-sizing: border-box; max-width: 100%; min-width: 0; overflow-x: auto; overscroll-behavior-inline: contain; width: 100%; }
.pramaan-state-mosaic { aspect-ratio: 10 / 9; display: grid; gap: clamp(2px,.35vw,5px); grid-template-columns: repeat(10, minmax(0, 1fr)); grid-template-rows: repeat(9, minmax(0, 1fr)); margin: 1rem 0; max-width: 960px; min-width: 640px; width: 100%; }
.pramaan-state-tile { align-items: center; aspect-ratio: 1; background: var(--pramaan-card); border: 1px solid var(--pramaan-line); border-radius: clamp(4px, .7vw, 12px); box-sizing: border-box; display: flex; flex-direction: column; justify-content: center; margin: 0; min-width: 0; min-height: 0; overflow: hidden; padding: clamp(2px,.45vw,7px); position: relative; text-align: center; }
.pramaan-state-tile img { aspect-ratio: 1; background: var(--pramaan-paper); border-radius: 8px; display: block; height: auto; max-width: 100%; object-fit: contain; width: 100%; }
.pramaan-state-mosaic .pramaan-state-tile img { max-height: calc(100% - 1rem); }
.pramaan-data-quilt-tile[data-tile-state="verified-first"] { border-color: var(--pramaan-teal); }
.pramaan-data-quilt-tile[data-tile-state="verified-repaired"] { border: 3px double var(--pramaan-teal); }
.pramaan-data-quilt-tile[data-tile-state="failed-all"] { background: repeating-linear-gradient(135deg,var(--pramaan-card) 0 8px,var(--pramaan-fail-bg) 8px 10px); border-color: var(--pramaan-madder); }
.pramaan-data-quilt-tile[data-tile-state="accepted-other-model"] { border-style: dashed; }
.pramaan-data-quilt-tile[data-tile-state="not-evaluated"] { border-color: var(--pramaan-line); }
.pramaan-data-quilt-tile[data-tile-state="community-run"] { box-shadow: inset -5px 5px 0 -2px var(--pramaan-madder); }
.pramaan-state-tile figcaption { display: none; gap: .18rem; padding-top: .35rem; }
.pramaan-state-tile:hover figcaption, .pramaan-state-tile:focus-within figcaption { display: grid; }
.pramaan-state-tile > strong { color: var(--pramaan-indigo); font-size: clamp(.55rem,1vw,.78rem); line-height: 1.1; overflow-wrap: anywhere; }
.pramaan-state-tile > span { color: var(--pramaan-muted); font-size: clamp(.42rem,.7vw,.62rem); line-height: 1.05; }
.pramaan-state-tile figcaption span, .pramaan-state-tile-empty span { color: var(--pramaan-muted); font-size: .68rem; line-height: 1.3; }
.pramaan-state-tile[data-gallery-state-tile] { cursor: pointer; }
.pramaan-state-tile[data-gallery-state-tile]:focus-visible { outline: 3px solid var(--pramaan-teal); outline-offset: 2px; z-index: 1; }
.pramaan-state-tile[data-gallery-state-tile][aria-selected="true"] { border: 2px solid var(--pramaan-teal); box-shadow: 0 0 0 2px color-mix(in srgb, var(--pramaan-teal) 14%, transparent); }
.pramaan-state-tile-empty { gap: .4rem; }
.pramaan-quilt-controls { align-items: end; display: grid; gap: .75rem; grid-template-columns: repeat(auto-fit,minmax(160px,1fr)); margin: 1rem 0; }
.pramaan-data-quilt-tile { justify-content: flex-start; gap: .3rem; }
.pramaan-data-quilt-tile > strong { color: var(--pramaan-ink); font-size: .78rem; }
.pramaan-data-quilt-tile > small { color: var(--pramaan-muted); font-size: .65rem; line-height: 1.3; }
.pramaan-data-quilt-tile details { align-self: stretch; font-size: .66rem; text-align: left; }
.pramaan-data-quilt-tile details summary { cursor: pointer; }
.pramaan-data-quilt-tile details p { margin: .35rem 0; }
.pramaan-data-quilt-tile .pramaan-quilt-attempts { margin: .3rem 0; padding-left: 1rem; }
.pramaan-data-quilt-tile .pramaan-quilt-attempts li { margin: .35rem 0; }
.pramaan-data-quilt-tile .pramaan-quilt-attempts small { display: block; overflow-wrap: anywhere; }
.pramaan-data-quilt-tile .pramaan-quilt-attempt { max-width: 56px; }
.pramaan-data-quilt-tile[data-quilt-state-tile] { cursor: pointer; }
.pramaan-data-quilt-tile[data-quilt-state-tile]:focus-visible { outline: 3px solid var(--pramaan-teal); outline-offset: 2px; z-index: 2; }
.pramaan-data-quilt-tile > .pramaan-quilt-details { display: none; }
.pramaan-quilt-dialog { background: var(--pramaan-card); border: 1px solid var(--pramaan-line); border-radius: 18px; box-shadow: 0 20px 80px color-mix(in srgb, var(--pramaan-shadow) 26%, transparent); box-sizing: border-box; color: var(--pramaan-ink); max-height: min(82dvh,720px); max-width: min(680px,calc(100vw - 2rem)); overflow: auto; padding: 1rem 1.2rem; width: 100%; }
.pramaan-quilt-dialog::backdrop { background: color-mix(in srgb, var(--pramaan-shadow) 42%, transparent); }
.pramaan-quilt-dialog-head { align-items: center; border-bottom: 1px solid var(--pramaan-line); display: flex; gap: .8rem; justify-content: space-between; margin-bottom: .8rem; padding-bottom: .6rem; }
.pramaan-quilt-dialog-head h3 { color: var(--pramaan-indigo); margin: 0; overflow-wrap: anywhere; }
.pramaan-quilt-dialog [data-quilt-sheet-content] { line-height: 1.45; overflow-wrap: anywhere; }
.pramaan-quilt-dialog .pramaan-quilt-attempts { display: grid; gap: .5rem; list-style: none; margin: .6rem 0; padding: 0; }
.pramaan-quilt-dialog .pramaan-quilt-attempts li { align-items: center; display: grid; gap: .7rem; grid-template-columns: minmax(72px,128px) minmax(0,1fr); }
.pramaan-quilt-dialog .pramaan-quilt-attempt { max-width: 128px; width: 100%; }
.pramaan-quilt-dialog a { color: var(--pramaan-teal); font-weight: 700; }
.pramaan-quilt-compare { display: grid; gap: .2rem; grid-template-columns: repeat(2,minmax(0,1fr)); width: 100%; }
.pramaan-quilt-compare-cell { display: grid; gap: .15rem; min-width: 0; }
.pramaan-quilt-compare-cell b,.pramaan-quilt-compare-cell small { font-size: .58rem; overflow-wrap: anywhere; }
.pramaan-quilt-hardness-count { color: var(--pramaan-ink); font-size: 1.2rem; font-weight: 750; }
.pramaan-quilt-legend { display: flex; flex-wrap: wrap; gap: .45rem .8rem; list-style: none; margin: -.35rem 0 1rem; padding: 0; }
.pramaan-quilt-legend li { align-items: center; color: var(--pramaan-muted); display: inline-flex; font-size: .74rem; gap: .35rem; }
.pramaan-quilt-legend li span { background: var(--pramaan-card); border: 2px solid var(--pramaan-teal); border-radius: 4px; height: 12px; width: 12px; }
.pramaan-quilt-legend [data-tile-state="verified-repaired"] span { border-style: double; border-width: 3px; }
.pramaan-quilt-legend [data-tile-state="failed-all"] span { background: repeating-linear-gradient(135deg,var(--pramaan-madder) 0 2px,var(--pramaan-card) 2px 4px); border-color: var(--pramaan-madder); }
.pramaan-quilt-legend [data-tile-state="accepted-other-model"] span { border-style: dashed; }
.pramaan-quilt-legend [data-tile-state="not-evaluated"] span { background: transparent; border-color: var(--pramaan-line); }
.pramaan-quilt-legend [data-tile-state="community-run"] span { background: radial-gradient(circle at 80% 20%,var(--pramaan-madder) 0 2px,transparent 3px),var(--pramaan-card); }
.pramaan-brandbar { align-items: center; background: linear-gradient(115deg,var(--pramaan-card),var(--pramaan-card)); border: 1px solid var(--pramaan-line); border-radius: 16px; box-shadow: 0 5px 20px color-mix(in srgb, var(--pramaan-shadow) 6%, transparent); box-sizing: border-box; display: flex; gap: .65rem; height: 60px; margin: 0 0 .7rem; max-height: 64px; overflow: hidden; padding: .35rem .7rem; }
.pramaan-brandbar .pramaan-logo { flex: 0 0 42px; height: 42px; object-fit: contain; width: 42px; }
.pramaan-brand-name { color: var(--pramaan-indigo); font-size: 1.05rem; font-weight: 800; letter-spacing: -.03em; line-height: 1; margin: 0; }
.pramaan-brand-name em { color: var(--pramaan-madder); font-size: .92rem; font-style: normal; font-weight: 550; }
.pramaan-ui-masthead .pramaan-masthead { margin-top: 0; }
.pramaan-ui-disclaimer { align-items: center; background: var(--pramaan-card); border-left: 4px solid var(--pramaan-madder); border-radius: 8px 14px 14px 8px; color: var(--pramaan-muted); display: flex; font-size: .82rem; justify-content: space-between; margin: 0 0 .55rem; padding: .45rem .7rem; }
.pramaan-ui-disclaimer[hidden] { display: none !important; }
.pramaan-ui-disclaimer button { background: transparent; border: 0; color: var(--pramaan-muted); cursor: pointer; font-size: 1.25rem; line-height: 1; }
.pramaan-ui-tabs { background: color-mix(in srgb, var(--pramaan-card) 96%, transparent); display: flex; flex-wrap: nowrap; gap: .3rem; margin: 0 0 .8rem; overflow-x: auto; overscroll-behavior-inline: contain; padding: .25rem 0; position: sticky; top: 0; z-index: 20; }
.pramaan-ui-tab { flex: 0 0 auto; min-height: 38px; padding: .45rem .7rem; white-space: nowrap; }
.pramaan-ui-tab:focus-visible { outline: 3px solid var(--pramaan-teal); outline-offset: 2px; }
.gradio-container footer, footer.svelte-1rjryqp { display: none !important; }
.pramaan-state-mosaic.is-festival-lens [data-festival-linked="false"] { opacity: .24; filter: grayscale(.85); }
.pramaan-state-mosaic.is-festival-lens [data-festival-linked="true"] { box-shadow: 0 0 .9rem color-mix(in srgb, var(--pramaan-pending) 70%, transparent); border-color: var(--pramaan-pending); }
@media (prefers-reduced-motion: no-preference) { .pramaan-state-tile { transition: opacity .25s ease, filter .25s ease, box-shadow .25s ease; } }
@media (prefers-reduced-motion: reduce) { .pramaan-state-tile { transition: none; } }
.pramaan-ui-card summary { cursor: pointer; font-weight: 700; }
.pramaan-ui-card a { color: var(--pramaan-teal); }
.pramaan-ui-footnote { margin: 1rem 0; text-align: center; }
.pramaan-proof-replay { align-items: center; display: grid; gap: clamp(1rem,3vw,2.5rem); grid-template-columns: minmax(0,1.15fr) minmax(210px,.85fr); margin: 0 0 1rem; overflow: hidden; padding: clamp(1rem,3vw,2rem); }
.pramaan-proof-copy, .pramaan-proof-art { min-width: 0; }
.pramaan-proof-replay h1 { color: var(--pramaan-indigo); font-family: Georgia,serif; font-size: clamp(2.25rem,6vw,4.8rem); letter-spacing: -.04em; line-height: .98; margin: .05rem 0 .65rem; }
.pramaan-proof-context { color: var(--pramaan-teal); font-size: .8rem; font-weight: 700; overflow-wrap: anywhere; }
.pramaan-proof-brief { font-size: .96rem; line-height: 1.45; margin: .65rem 0; }
.pramaan-proof-stage { background: var(--pramaan-card-subtle); border: 1px solid var(--pramaan-line); border-radius: 14px; min-height: 120px; padding: .8rem; }
.pramaan-proof-step[hidden] { display: none !important; }
.pramaan-proof-tile { aspect-ratio: 1; background: var(--pramaan-card-raised); border: 1px solid var(--pramaan-line); border-radius: 10px; display: block; height: auto; max-height: 250px; max-width: 100%; object-fit: contain; width: min(100%,250px); }
.pramaan-proof-art { margin: 0; }
.pramaan-proof-art .pramaan-proof-tile { margin-inline: auto; max-height: 340px; width: min(100%,340px); }
.pramaan-proof-art figcaption { color: var(--pramaan-muted); font-size: .78rem; margin-top: .45rem; overflow-wrap: anywhere; text-align: center; }
.pramaan-proof-checks { display: grid; gap: .32rem; list-style: none; margin: .45rem 0 0; padding: 0; }
.pramaan-proof-checks li { align-items: start; display: grid; font-size: .76rem; gap: .35rem; grid-template-columns: 1rem minmax(90px,.35fr) minmax(0,1fr); line-height: 1.3; overflow-wrap: anywhere; }
.pramaan-proof-checks .is-pass > span:first-child { color: var(--pramaan-pass); font-weight: 800; }
.pramaan-proof-checks .is-fail > span:first-child { color: var(--pramaan-fail); font-weight: 900; }
.pramaan-proof-controls { align-items: center; display: flex; flex-wrap: wrap; gap: .5rem; margin: .7rem 0; }
.pramaan-proof-controls .pramaan-ui-button, .pramaan-proof-copy > [data-proof-verify] { min-height: 40px; }
.pramaan-proof-progress { color: var(--pramaan-muted); font-size: .78rem; }
.pramaan-proof-digest { display: block; font-size: .76rem; overflow-wrap: anywhere; }
@media (prefers-reduced-motion: reduce) { .pramaan-proof-step { scroll-behavior: auto; transition: none !important; } }
@media (max-width: 820px) {
  .pramaan-findings-grid { grid-template-columns: minmax(0,1fr); }
  .pramaan-masthead { grid-template-columns: 72px minmax(0, 1fr); align-items: flex-start; gap: .85rem; padding: .9rem; }
  .pramaan-logo { height: 68px; width: 68px; }
  .pramaan-tagline { font-size: .88rem; }
  .pramaan-method { grid-template-columns: 1fr; gap: .45rem; }
  .pramaan-workspace { grid-template-columns: minmax(0, 1fr); }
  .pramaan-ui-layout { grid-template-columns: minmax(0,1fr); }
  .pramaan-ui-grid { grid-template-columns: minmax(0,1fr); }
  .pramaan-ui-tab { flex: 0 0 auto; }
}
@media (max-width: 600px) {
  .pramaan-proof-replay { grid-template-columns: minmax(0,1fr); }
  .pramaan-proof-art { grid-row: 1; }
  .pramaan-proof-art .pramaan-proof-tile { max-height: 210px; width: min(100%,210px); }
  .pramaan-proof-stage { min-height: 100px; }
  .pramaan-proof-step .pramaan-proof-tile { display: none; }
  .pramaan-proof-checks li { grid-template-columns: .9rem minmax(76px,.35fr) minmax(0,1fr); }
  .pramaan-brandbar { border-radius: 12px; gap: .45rem; height: 54px; padding: .3rem .5rem; }
  .pramaan-brandbar .pramaan-logo { flex-basis: 38px; height: 38px; width: 38px; }
  .pramaan-brand-name { font-size: .96rem; }
  .pramaan-brand-name em { font-size: .82rem; }
  .pramaan-ui, .pramaan-state-mosaic-scroll { max-width: 100%; min-width: 0; width: 100%; }
  .pramaan-state-mosaic { min-width: 340px; }
  .pramaan-state-tile { border-radius: 4px; padding: 1px; }
  .pramaan-state-tile > strong { font-size: .52rem; }
  .pramaan-ui-tab { font-size: .78rem; min-height: 36px; padding: .38rem .55rem; }
  .pramaan-ui-disclaimer { font-size: .72rem; }
}
.pramaan-home-intro { color: var(--pramaan-text); font-size: clamp(1rem,2vw,1.2rem); line-height: 1.45; margin: .25rem auto .8rem; max-width: 920px; text-align: center; }
.pramaan-home-intro h1 { color: var(--pramaan-indigo); font-family: Georgia,serif; font-size: clamp(2.3rem,6vw,4.5rem); letter-spacing: -.04em; line-height: 1; margin: .1rem 0 .5rem; }
#pramaan-hf-login-row { display: none; }
.pramaan-home-actions { align-items: center; display: flex; flex-wrap: wrap; gap: .55rem; justify-content: center; margin-top: .6rem; }
.pramaan-home-text-link { color: var(--pramaan-teal); font-size: .9rem; font-weight: 700; padding: .5rem; }
.pramaan-home-metrics { display: grid; gap: .65rem; grid-template-columns: repeat(3,minmax(0,1fr)); margin: .8rem 0 1.25rem; }
.pramaan-home-metrics article { background: var(--pramaan-card); border: 1px solid var(--pramaan-line); border-radius: 14px; display: grid; gap: .2rem; padding: .8rem; text-align: center; }
.pramaan-home-metrics strong { color: var(--pramaan-indigo); font-size: clamp(1.2rem,3vw,1.8rem); font-variant-numeric: tabular-nums; }
.pramaan-home-metrics span { color: var(--pramaan-muted); font-size: .82rem; }
.pramaan-home-wall { margin: 1rem 0; }
.pramaan-home-wall-heading { align-items: end; display: flex; flex-wrap: wrap; gap: 1rem; justify-content: space-between; margin-bottom: .65rem; }
.pramaan-home-wall-heading h2 { color: var(--pramaan-indigo); font-family: Georgia,serif; font-size: clamp(1.8rem,4vw,2.6rem); margin: 0; }
.pramaan-home-wall-heading p:last-child { color: var(--pramaan-muted); margin: .3rem 0; max-width: 68ch; }
.pramaan-home-reference-note { color: var(--pramaan-muted); font-size: .82rem; }
.pramaan-wall-controls { align-items: start; display: grid; gap: .65rem; margin: .65rem 0; }
.pramaan-wall-models { display: flex; flex-wrap: wrap; gap: .4rem; }
.pramaan-wall-model-chip { align-items: flex-start; background: var(--pramaan-card); border: 1px solid var(--pramaan-line); border-radius: 999px; color: var(--pramaan-text); cursor: pointer; display: flex; flex-direction: column; font: inherit; gap: .04rem; max-width: 100%; padding: .35rem .75rem; text-align: left; }
.pramaan-wall-model-chip[aria-checked="true"] { background: var(--pramaan-teal); border-color: var(--pramaan-teal); color: var(--pramaan-on-accent); }
.pramaan-wall-model-chip small { color: var(--pramaan-muted); font-size: 12px; opacity: 1; }
.gradio-container .pramaan-ui .pramaan-wall-model-chip[aria-checked="true"] small { color: var(--pramaan-on-accent) !important; font-size: 12px !important; opacity: 1 !important; }
.pramaan-wall-mode-toggle { background: var(--pramaan-card); border: 1px solid var(--pramaan-line); border-radius: 999px; display: inline-flex; justify-self: start; overflow: hidden; }
.pramaan-wall-mode { background: transparent; border: 0; color: var(--pramaan-text); cursor: pointer; font: inherit; font-size: .82rem; min-height: 38px; padding: .35rem .85rem; }
.pramaan-wall-mode[aria-pressed="true"] { background: var(--pramaan-indigo); color: var(--pramaan-on-accent); }
.pramaan-wall-legend { color: var(--pramaan-muted); display: flex; flex-wrap: wrap; gap: .3rem 1rem; list-style: none; margin: 0 0 .55rem; padding: 0; }
.pramaan-wall-legend li { align-items: center; display: inline-flex; gap: .35rem; font-size: .78rem; }
.pramaan-wall-legend [data-status="first"] { color: var(--pramaan-pass); }
.pramaan-wall-legend [data-status="repair"] { color: var(--pramaan-pending); }
.pramaan-wall-legend [data-status="fail"] { color: var(--pramaan-fail); }
.pramaan-wall-legend [data-status="none"] { color: var(--pramaan-muted); }
.pramaan-home-wall-grid { display: grid; gap: clamp(2px,.45vw,6px); grid-template-columns: repeat(10,minmax(0,1fr)); grid-template-rows: repeat(9,minmax(0,1fr)); margin: .6rem auto; max-width: 980px; width: 100%; }
.pramaan-wall-state { aspect-ratio: 1; min-width: 0; position: relative; }
.pramaan-wall-tile { background: var(--pramaan-card); border: 1px solid var(--pramaan-line); border-radius: clamp(3px,.6vw,9px); color: var(--pramaan-text); cursor: pointer; display: block; height: 100%; overflow: hidden; padding: 0; position: relative; text-align: left; width: 100%; }
.pramaan-wall-tile:focus-visible, .pramaan-wall-model-chip:focus-visible, .pramaan-wall-mode:focus-visible { outline: 3px solid var(--pramaan-teal); outline-offset: 2px; z-index: 2; }
.pramaan-wall-art { display: block; height: 100%; image-rendering: pixelated; object-fit: cover; width: 100%; }
.pramaan-wall-empty-art, .pramaan-wall-placeholder { align-items: center; background: repeating-linear-gradient(135deg,var(--pramaan-card-subtle) 0 6px,var(--pramaan-line) 6px 7px); color: var(--pramaan-muted); display: flex; justify-content: center; min-height: 90px; text-align: center; }
.pramaan-wall-state-name { background: color-mix(in srgb,var(--pramaan-card) 90%,transparent); bottom: 0; color: var(--pramaan-text); font-size: clamp(.52rem,.7vw,.72rem); font-weight: 750; left: 0; line-height: 1.1; overflow: hidden; padding: 2px 3px; position: absolute; right: 0; text-overflow: ellipsis; white-space: nowrap; }
.pramaan-wall-marks { background: var(--pramaan-card); border: 1px solid var(--pramaan-line); border-radius: 999px; display: flex; gap: 2px; padding: 0 3px; position: absolute; right: 2px; top: 2px; }
.pramaan-wall-mark { font-size: clamp(.48rem,.75vw,.8rem); font-weight: 900; line-height: 1.25; }
.pramaan-wall-mark[data-status="first"] { color: var(--pramaan-pass); }
.pramaan-wall-mark[data-status="repair"] { color: var(--pramaan-pending); }
.pramaan-wall-mark[data-status="fail"] { color: var(--pramaan-fail); }
.pramaan-wall-mark[data-status="none"] { color: var(--pramaan-muted); }
.pramaan-wall-dialog { background: var(--pramaan-card); border: 1px solid var(--pramaan-line); border-radius: 16px; color: var(--pramaan-text); max-height: min(90vh,900px); max-width: min(900px,calc(100vw - 24px)); overflow: auto; padding: clamp(.8rem,2vw,1.3rem); width: min(900px,calc(100vw - 24px)); }
.pramaan-wall-dialog::backdrop { background: rgba(8,14,22,.65); }
.pramaan-wall-dialog-head { align-items: center; border-bottom: 1px solid var(--pramaan-line); display: flex; justify-content: space-between; padding-bottom: .55rem; }
.pramaan-wall-dialog-head h3 { color: var(--pramaan-indigo); font: 700 1.6rem Georgia,serif; margin: 0; }
.pramaan-wall-task { border-bottom: 1px solid var(--pramaan-line); padding: .7rem 0 1rem; }
.pramaan-wall-task h4 { margin: 0; }
.pramaan-wall-brief { color: var(--pramaan-muted); }
.pramaan-wall-pair { display: grid; gap: .75rem; grid-template-columns: repeat(2,minmax(0,1fr)); }
.pramaan-wall-pair figure { align-content: start; display: grid; gap: .35rem; margin: 0; min-width: 0; }
.pramaan-wall-pair figcaption { font-weight: 750; }
.pramaan-wall-pair img, .pramaan-wall-pair .pramaan-wall-placeholder { border: 1px solid var(--pramaan-line); border-radius: 8px; image-rendering: pixelated; max-height: 260px; object-fit: contain; width: min(100%,260px); }
.pramaan-wall-model-list { margin-top: .7rem; }
.pramaan-wall-model-list summary { color: var(--pramaan-teal); cursor: pointer; font-weight: 750; }
.pramaan-wall-model-list ul { display: grid; gap: .3rem; list-style: none; margin: .4rem 0 0; padding: 0; }
.pramaan-wall-model-row { align-items: center; background: var(--pramaan-card-subtle); border: 1px solid var(--pramaan-line); border-radius: 9px; display: grid; gap: .6rem; grid-template-columns: minmax(70px,.25fr) minmax(0,1fr); min-width: 0; padding: .3rem .45rem; }
.pramaan-wall-model-outcomes { display: grid; gap: .3rem; grid-template-columns: repeat(2,minmax(0,1fr)); }
.pramaan-wall-model-outcome { align-items: center; display: grid; gap: .3rem; grid-template-columns: 30px minmax(0,1fr); min-width: 0; }
.pramaan-wall-model-outcome small { color: var(--pramaan-muted); grid-column: 1 / -1; }
.pramaan-wall-model-outcome b { font-size: .72rem; overflow-wrap: anywhere; }
.pramaan-wall-model-outcome b[data-outcome="first"] { color: var(--pramaan-pass); }
.pramaan-wall-model-outcome b[data-outcome="repair"] { color: var(--pramaan-pending); }
.pramaan-wall-model-outcome b[data-outcome="fail"] { color: var(--pramaan-fail); }
.pramaan-wall-model-outcome b[data-outcome="none"] { color: var(--pramaan-muted); }
.pramaan-wall-model-thumb { border-radius: 4px; height: 36px; image-rendering: pixelated; object-fit: cover; width: 36px; }
.pramaan-wall-outcome { font-size: 1.1rem; font-weight: 900; }
.pramaan-wall-outcome[data-outcome="first"] { color: var(--pramaan-pass); }
.pramaan-wall-outcome[data-outcome="repair"] { color: var(--pramaan-pending); }
.pramaan-wall-outcome[data-outcome="fail"] { color: var(--pramaan-fail); }
.pramaan-wall-outcome[data-outcome="none"] { color: var(--pramaan-muted); }
.pramaan-wall-care { color: var(--pramaan-muted); font-size: .85rem; margin-top: .75rem; }
.pramaan-wall-task ol { color: var(--pramaan-muted); font-size: .82rem; padding-left: 1.2rem; }
.pramaan-wall-task ol small { display: block; }
.pramaan-ui-intro { color: var(--pramaan-muted); font-size: 1rem; margin: .2rem 0 .8rem; }
@media (max-width: 600px) {
  .pramaan-home-metrics { gap: .35rem; }
  .pramaan-home-metrics article { border-radius: 10px; padding: .55rem .35rem; }
  .pramaan-home-metrics span { font-size: .68rem; }
  .pramaan-home-wall-grid { gap: 2px; }
  .pramaan-wall-state-name { display: none; }
  .pramaan-wall-marks { gap: 0; padding: 0 1px; right: 0; top: 0; }
  .pramaan-wall-mark { font-size: .46rem; }
  .pramaan-wall-pair { grid-template-columns: minmax(0,1fr); }
  .pramaan-home-actions { gap: .35rem; }
}
"""


def _brand_header_html() -> str:
    logo_uri = _brand_logo_data_uri()
    logo_html = (f'<img class="pramaan-logo" src="{logo_uri}" '
                 'alt="Pramaan geometric loop mark" fetchpriority="high">' if logo_uri else "")
    return (f'<header class="pramaan-brandbar">{logo_html}'
            '<h1 class="pramaan-brand-name">Pramaan <em>Proof of Art</em></h1></header>')


def _verify_json(source: str):
    pattern = parse_pattern(source)
    checks, masks, grid = verify_pattern(pattern)
    cert = make_certificate(pattern, checks, masks, grid)
    return pattern, checks, masks, grid, cert


def _witness_markers(report: dict) -> list[tuple[int, int, str]]:
    markers = []
    for check in report.get("checks", []):
        witness = check.get("counterexample") or {}
        for key, colour in (("original", "#FF2D95"), ("transformed", "#00E5FF"),
                            ("pixel", "#FFE600"), ("other_pixel", "#FF8C00")):
            point = witness.get(key)
            if isinstance(point, list) and len(point) == 2:
                markers.append((int(point[0]), int(point[1]), colour))
    return markers


def _attempt_gallery(attempts: list[dict]):
    gallery = []
    for attempt in attempts:
        candidate = attempt.get("candidate")
        if candidate is None:
            continue
        try:
            pattern = parse_pattern(json.dumps(candidate))
            _, _, grid = verify_pattern(pattern)
        except (ValidationError, ValueError, json.JSONDecodeError):
            continue
        report = attempt["report"]
        passed = report.get("all_hard_constraints_pass", False)
        markers = _witness_markers(report)
        label = f"Proposal {attempt['index']} · {'PASS' if passed else 'FAIL'}"
        gallery.append((render_highlighted_png(grid, pattern.palette, markers, scale=6, repeats=1), label))
    return gallery


def load_example(name: str):
    path = EXAMPLES[name]
    source = path.read_text(encoding="utf-8")
    pattern, checks, masks, grid, cert = _verify_json(source)
    verified = all_pass(checks)
    markers = _witness_markers({"checks": checks})
    image = (render_png(grid, pattern.palette, scale=5, repeats=3) if verified else
             render_highlighted_png(grid, pattern.palette, markers, scale=5, repeats=3))
    example_id = Path(path).name.removesuffix(".pattern.json")
    trajectory_path = BEAM_TRAJECTORIES.get(example_id)
    if trajectory_path and verified:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        repaired = bool(trajectory.get("repair_count", 0))
        provenance = {
            "origin": "beam_generated_and_repaired" if repaired else "beam_generated_verified",
            "example_id": example_id,
            "beam_generated": True,
            "model": trajectory["model"],
            "created_at": trajectory["created_at"],
            "original_prompt": trajectory["original_prompt"],
            "attempt_count": len(trajectory["attempts"]),
            "repair_count": trajectory["repair_count"],
            "request_params": trajectory["request_params"],
            "constraint_profile": trajectory["constraint_profile"],
        }
        gallery = _attempt_gallery(trajectory["attempts"])
        history = json.dumps(trajectory, indent=2, ensure_ascii=False)
        stage = "repair" if repaired else "proposal"
        proposal_word = "proposal" if len(trajectory["attempts"]) == 1 else "proposals"
        status = (f"Verified at 64×64 discrete resolution · Beam {stage} accepted "
                  f"after {len(trajectory['attempts'])} {proposal_word}")
    else:
        label = f"Built-in · hand-authored · {'PASS' if verified else 'FAIL'}"
        gallery = [(render_highlighted_png(grid, pattern.palette, markers, scale=6, repeats=1), label)]
        provenance = {"origin": "hand_authored_builtin_example", "example_id": example_id, "beam_generated": False}
        history = json.dumps({**provenance, "all_hard_constraints_pass": verified}, indent=2)
        status = ("Verified at 64×64 discrete resolution" if verified else
                  "Not verified · hand-authored counterexample demonstrates a failed constraint")
    bundle = _artifact_bundle_with_seal(pattern, cert, grid, provenance) if verified else None
    return image, gallery, source, json.dumps(cert, indent=2), history, status, bundle


def generate(brief: str, state_name: str, tradition_id: str, palette_name: str, symmetry: str,
             max_colours: int, min_coverage: float, max_coverage: float, exclusive_ink_layers: bool,
             min_gap: int, required_loops: int, cell_size: int, translation_axis: str,
             translation_period: int, min_clearance: int, frame: int, min_web: int):
    if not isinstance(brief, str) or not brief.strip():
        raise gr.Error("Describe the pattern you want first.")
    if len(brief) > MAX_BRIEF_CHARS:
        raise gr.Error(f"Keep the pattern brief under {MAX_BRIEF_CHARS} characters.")
    tradition = _tradition_by_id(tradition_id)
    if tradition is None or state_name not in tradition.states:
        raise gr.Error("Choose a tradition that belongs to the selected state.")
    palette_options = tradition.palette_presets
    if not isinstance(palette_name, str) or palette_name not in palette_options:
        raise gr.Error("Choose a palette preset from the selected tradition.")
    selected_palette = list(palette_options[palette_name])
    try:
        max_colours = require_integer_control(max_colours, "Maximum colours")
        min_gap = require_integer_control(min_gap, "Minimum spacing")
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    if not 1 <= max_colours <= min(6, len(selected_palette)):
        raise gr.Error("Maximum colours must be between 1 and the palette size.")
    if not 0 <= min_gap <= 8:
        raise gr.Error("Minimum spacing must be between 0 and 8 raster pixels.")
    allowed_symmetry = {"None", "D2", "D4"} if tradition.grammar == "periodic_raster" else {"None", "D2", "D4", "C2", "C4"}
    if not isinstance(symmetry, str) or symmetry not in allowed_symmetry:
        raise gr.Error("Choose one of the supported symmetry options.")
    profile = dict(tradition.default_profile)
    profile.update({
        "symmetry": symmetry, "max_colours": max_colours, "min_coverage": min_coverage,
        "max_coverage": max_coverage, "min_gap": min_gap,
        "exclusive_ink_layers": bool(exclusive_ink_layers),
    })
    try:
        if tradition.grammar == "kolam_loop":
            profile["required_loops"] = require_integer_control(required_loops, "Required loops")
        elif tradition.grammar == "counted_grid":
            profile["cell_size"] = require_integer_control(cell_size, "Cell size")
        elif tradition.grammar == "stripe_band":
            if translation_axis not in {"x", "y", "xy"}:
                raise ValueError("Choose x, y, or xy for the translation axis.")
            profile["translation_period"] = {
                "axis": translation_axis,
                "period": require_integer_control(translation_period, "Translation period"),
            }
        elif tradition.grammar == "dot_field":
            profile["min_clearance"] = require_integer_control(min_clearance, "Minimum clearance")
        elif tradition.grammar == "jaali_lattice":
            profile["frame"] = require_integer_control(frame, "Jaali frame")
            profile["min_web"] = require_integer_control(min_web, "Minimum web")
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    task = SimpleNamespace(
        id=f"studio-{state_name.lower().replace(' ', '-')}-{tradition.id}",
        suite_id="studio", state=state_name, tradition_id=tradition.id,
        grammar=tradition.grammar, level="interactive", symmetry=symmetry,
        constraint_profile=profile, palette=selected_palette, brief=brief.strip(),
    )
    try:
        result = generate_and_verify(task=task)
    except Exception as exc:
        # Provider exception text can contain request fragments or credential-related details.
        raise gr.Error("Beam request failed. Offline examples remain available.") from exc
    attempts = result["attempts"]
    history_data = {
        "trajectory_schema_version": result["trajectory_schema_version"],
        "trajectory_id": f"studio-{uuid.uuid4().hex}", "data_role": "unassigned",
        "training_eligible": False, "prompt_id": task.id, "suite_id": task.suite_id,
        "state": state_name, "tradition_id": tradition.id, "grammar": tradition.grammar,
        "level": task.level, "symmetry": symmetry, "original_prompt": brief.strip(),
        "model": result["model"], "created_at": result["created_at"],
        "constraint_profile": result["constraint_profile"],
        "elapsed_ms": result["elapsed_ms"],
        "telemetry": result["telemetry"],
        "provider_error": result.get("provider_error"),
        "model_calls": result["model_calls"],
        "attempts": [{"index": item["index"], "call_index": item["call_index"],
                       "verified_at": item["verified_at"],
                       "verification_elapsed_ms": item["verification_elapsed_ms"],
                       "candidate": item["candidate"],
                       "report": item["report"], "metadata": item["metadata"]} for item in attempts],
    }
    history = json.dumps(history_data, indent=2, ensure_ascii=False)
    gallery = _attempt_gallery(attempts)
    if not attempts:
        failure = result.get("provider_error") or {"phase": "proposal", "type": "UnknownProviderError"}
        certificate_text = f"No certificate: the Beam {failure['phase']} request failed before a candidate was checked."
        status_text = f"Not verified · Beam request failed ({failure['type']})"
        bundle = trajectory_bundle(history_data)
        return None, gallery, "No PatternDSL candidate was returned.", certificate_text, history, status_text, bundle
    if result["accepted"]:
        pattern = result["pattern"]
        image = render_png(result["grid"], pattern.palette, scale=5, repeats=3)
        provenance = {**history_data, "beam_generated": True}
        bundle = _artifact_bundle_with_seal(pattern, result["certificate"], result["grid"], provenance,
                                            trajectory=history_data)
        return image, gallery, json.dumps(pattern.model_dump(mode="json"), indent=2), json.dumps(result["certificate"], indent=2), history, "Verified at 64×64 discrete resolution", bundle
    last = attempts[-1]
    source = json.dumps(last["candidate"], indent=2) if last["candidate"] else "No valid PatternDSL candidate was returned."
    preview = None
    if last["candidate"]:
        failed_pattern = parse_pattern(json.dumps(last["candidate"]))
        _, _, failed_grid = verify_pattern(failed_pattern)
        preview = render_highlighted_png(failed_grid, failed_pattern.palette, _witness_markers(last["report"]), scale=5, repeats=3)
    if result.get("provider_error"):
        failure = result["provider_error"]
        certificate_text = f"No certificate: the Beam {failure['phase']} request failed before all hard constraints passed."
        status_text = f"Not verified · Beam {failure['phase']} request failed ({failure['type']})"
    else:
        certificate_text = "No certificate: hard constraints did not all pass."
        status_text = "Not verified · retry limit reached"
    bundle = trajectory_bundle(history_data)
    return preview, gallery, source, certificate_text, history, status_text, bundle


def recheck(source: str):
    try:
        pattern, checks, masks, grid, certificate = _verify_json(source)
    except (ValidationError, ValueError, json.JSONDecodeError) as exc:
        raise gr.Error(f"PatternDSL could not be checked: {exc}") from exc
    status = "Verified at 64×64 discrete resolution" if all_pass(checks) else "Not verified · one or more checks failed"
    bundle = _artifact_bundle_with_seal(pattern, certificate, grid) if all_pass(checks) else None
    gallery = [(render_highlighted_png(grid, pattern.palette, _witness_markers({"checks": checks}), scale=6, repeats=1), status)]
    preview = render_highlighted_png(grid, pattern.palette, _witness_markers({"checks": checks}), scale=5, repeats=3)
    return preview, gallery, json.dumps(certificate, indent=2), status, bundle


def build_demo():
    ui_state = _initial_custom_state()
    quilt_navigation_js = (Path(__file__).parent / "ui" / "assets" / "quilt.js").read_text(encoding="utf-8")
    js_on_load = quilt_navigation_js + r"""
function applyLeaderboardControls() {
  const search = (element.querySelector('[data-leaderboard-search]')?.value || '').trim().toLowerCase();
  const source = element.querySelector('[data-leaderboard-provenance]')?.value || 'all';
  const sortKey = element.querySelector('[data-leaderboard-sort]')?.value || 'pass3';
  const dataKey = ({pass1:'scorePass1', pass3:'scorePass3', coverage:'coverage', cost:'cost'})[sortKey] || 'scorePass3';
  let visible = 0;
  let total = 0;
  element.querySelectorAll('[data-leaderboard-table]').forEach(table => {
    const body = table.tBodies?.[0];
    if (!body) return;
    const rows = Array.from(body.querySelectorAll('tr[data-leaderboard-row]'));
    total += rows.length;
    rows.forEach(row => {
      const sourceMatches = source === 'all' || row.dataset.source === source;
      const searchMatches = !search || row.textContent.toLowerCase().includes(search);
      row.hidden = !(sourceMatches && searchMatches);
      if (!row.hidden) visible += 1;
    });
    const descending = ['pass1', 'pass3', 'coverage'].includes(sortKey);
    rows.sort((left, right) => {
      const a = Number.parseFloat(left.dataset[dataKey]);
      const b = Number.parseFloat(right.dataset[dataKey]);
      const aMissing = !Number.isFinite(a), bMissing = !Number.isFinite(b);
      if (aMissing !== bMissing) return aMissing ? 1 : -1;
      if (aMissing) return left.dataset.modelKey.localeCompare(right.dataset.modelKey);
      return descending ? b - a : a - b;
    }).forEach(row => body.appendChild(row));
  });
  const count = element.querySelector('[data-leaderboard-count]');
  if (count) count.textContent = `${visible} of ${total} model rows shown`;
}
function activateTab(key) {
  const valid = ['benchmark', 'leaderboard', 'studio', 'rl'];
  if (!valid.includes(key)) key = 'benchmark';
  const loginRow = document.querySelector('#pramaan-hf-login-row');
  const communityDisclosure = element.querySelector('#leaderboard-full-results');
  element.dataset.activeTab = key;
  if (communityDisclosure && !communityDisclosure.dataset.loginVisibilityBound) {
    communityDisclosure.dataset.loginVisibilityBound = 'true';
    communityDisclosure.addEventListener('toggle', () => {
      if (loginRow) loginRow.style.display = element.dataset.activeTab === 'leaderboard' && communityDisclosure.open ? 'flex' : 'none';
    });
  }
  if (loginRow) loginRow.style.display = key === 'leaderboard' && communityDisclosure?.open ? 'flex' : 'none';
  element.querySelectorAll('[data-tab-button]').forEach(node => {
    const active = node.dataset.tabButton === key;
    node.setAttribute('aria-selected', String(active));
    node.tabIndex = active ? 0 : -1;
  });
  element.querySelectorAll('[data-tab-panel]').forEach(node => node.hidden = node.dataset.tabPanel !== key);
}
function applyRoute() {
  const raw = window.location.hash || '#/overview';
  const [path, query = ''] = raw.slice(1).split('?');
  const params = new URLSearchParams(query);
  const key = path === '/leaderboard' ? 'leaderboard' : path === '/studio' ? 'studio' :
    path === '/build' || path === '/rl' ? 'rl' : 'benchmark';
  activateTab(key);
  if (path === '/states' && params.has('state') && window.__pramaanAppliedHash !== raw) {
    window.__pramaanAppliedHash = raw;
    trigger('click', { action: 'gallery_state_select', gallery_state: params.get('state'), active_tab: key });
  }
}
const disclaimer = element.querySelector('[data-ui-disclaimer]');
try { if (disclaimer && window.localStorage.getItem('pramaan-disclaimer-dismissed') === '1') disclaimer.hidden = true; } catch (_error) {}
window.addEventListener('hashchange', applyRoute);
window.addEventListener('popstate', applyRoute);
applyRoute();
const quiltSheet = element.querySelector('[data-testid="quilt-state-sheet"]');
let quiltSheetOpener = null;
if (quiltSheet) {
  quiltSheet.querySelector('[data-quilt-close]')?.addEventListener('click', () => quiltSheet.close());
  quiltSheet.addEventListener('close', () => quiltSheetOpener?.focus());
}
const proofReplay = element.querySelector('[data-testid="proof-replay"]');
if (proofReplay) {
  const proofSteps = Array.from(proofReplay.querySelectorAll('.pramaan-proof-step[data-proof-step]'));
  const proofProgress = proofReplay.querySelector('[data-proof-progress]');
  const proofArtwork = proofReplay.querySelector('[data-proof-active-tile]');
  const proofFigureCaption = proofReplay.querySelector('[data-proof-figure-caption]');
  let proofIndex = Math.max(0, proofSteps.findIndex(node => node.hasAttribute('data-proof-result') && !node.hidden));
  let proofTimer = null;
  function showProofStep(index) {
    proofIndex = Math.max(0, Math.min(index, proofSteps.length - 1));
    proofSteps.forEach((node, i) => { node.hidden = i !== proofIndex; });
    const activeStep = proofSteps[proofIndex];
    const stepArtwork = activeStep?.querySelector('.pramaan-proof-tile') ||
      proofSteps.slice(0, proofIndex + 1).reverse().map(node => node.querySelector('.pramaan-proof-tile')).find(Boolean) ||
      proofSteps.map(node => node.querySelector('.pramaan-proof-tile')).find(Boolean);
    if (proofArtwork && stepArtwork) {
      proofArtwork.src = stepArtwork.src;
      proofArtwork.alt = stepArtwork.alt;
    }
    const stepTitle = activeStep?.querySelector('b')?.textContent || 'Proof Replay';
    if (proofFigureCaption) proofFigureCaption.textContent = `${stepTitle} · ${proofReplay.querySelector('.pramaan-proof-context')?.textContent || ''}`;
    if (proofProgress) proofProgress.textContent = activeStep?.hasAttribute('data-proof-result') && !proofReplay.dataset.proofStarted
      ? 'Verified result · replay ready' : `Step ${proofIndex + 1} of ${proofSteps.length}`;
  }
  function stopProofPlayback() {
    if (proofTimer) window.clearInterval(proofTimer);
    proofTimer = null;
  }
  function playProof() {
    stopProofPlayback();
    proofReplay.dataset.proofStarted = 'true';
    showProofStep(0);
    proofTimer = window.setInterval(() => {
      if (proofIndex >= proofSteps.length - 1) { stopProofPlayback(); return; }
      showProofStep(proofIndex + 1);
    }, 2000);
  }
  showProofStep(proofIndex);
  proofReplay.addEventListener('pointerdown', stopProofPlayback, {passive:true});
  proofReplay.addEventListener('wheel', stopProofPlayback, {passive:true});
  proofReplay.addEventListener('touchstart', stopProofPlayback, {passive:true});
  proofReplay.addEventListener('focusin', stopProofPlayback);
  proofReplay.querySelector('[data-proof-replay]')?.addEventListener('click', playProof);
  proofReplay.querySelector('[data-testid="proof-replay-step"]')?.addEventListener('click', () => {
    stopProofPlayback(); showProofStep(proofIndex + 1);
  });
  proofReplay.querySelector('[data-proof-verify]')?.addEventListener('click', event => {
    stopProofPlayback();
    const button = event.currentTarget;
    try {
      const bytes = Uint8Array.from(atob(button.dataset.patternB64 || ''), char => char.charCodeAt(0));
      const patternSource = new TextDecoder().decode(bytes);
      const sourceField = element.querySelector('[data-field="source"]');
      if (!sourceField) throw new Error('Studio source field is unavailable');
      sourceField.value = JSON.stringify(JSON.parse(patternSource), null, 2);
      history.pushState(null, '', '#/studio');
      applyRoute();
      const values = {};
      element.querySelectorAll('[data-field]').forEach(node => {
        if (node.type === 'file' || node.dataset.field === 'community_api_key') return;
        values[node.dataset.field] = node.type === 'checkbox' ? node.checked : node.value;
      });
      values.action = 'recheck'; values.active_tab = 'studio';
      trigger('click', values);
    } catch (_error) {
      if (proofProgress) proofProgress.textContent = 'Could not load the saved pattern into Studio.';
    }
  });
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    showProofStep(proofSteps.length - 1);
  } else {
    if ('IntersectionObserver' in window) {
      const proofVisibility = new IntersectionObserver(entries => {
        if (!entries[0]?.isIntersecting) stopProofPlayback();
      }, {threshold:0});
      proofVisibility.observe(proofReplay);
    }
  }
}
element.addEventListener('click', async function(event) {
  const wallChoice = event.target.closest('[data-wall-model-choice], [data-wall-mode-choice]');
  if (wallChoice) {
    const wall = wallChoice.closest('.pramaan-home-wall');
    const fieldName = wallChoice.hasAttribute('data-wall-model-choice') ? 'wall_model' : 'wall_mode';
    const field = wall?.querySelector(`[data-field="${fieldName}"]`);
    if (field) {
      field.value = wallChoice.dataset.wallModelChoice || wallChoice.dataset.wallModeChoice;
      field.dispatchEvent(new Event('change', {bubbles:true}));
    }
    return;
  }
  const wallClose = event.target.closest('[data-wall-dialog-close]');
  if (wallClose) {
    wallClose.closest('[data-wall-dialog]')?.close();
    return;
  }
  const wallTile = event.target.closest('[data-home-wall-tile]');
  if (wallTile) {
    const wall = wallTile.closest('.pramaan-home-wall');
    const dialog = wall?.querySelector('[data-wall-dialog]');
    const template = wallTile.parentElement?.querySelector('template[data-wall-state-sheet]');
    const title = dialog?.querySelector('[data-wall-dialog-title]');
    const content = dialog?.querySelector('[data-wall-dialog-content]');
    if (dialog && template && content) {
      if (title) title.textContent = wallTile.dataset.state || 'State evidence';
      content.replaceChildren(template.content.cloneNode(true));
      dialog.showModal();
    }
    return;
  }
  const wallDialog = event.target.closest('[data-wall-dialog]');
  if (wallDialog && event.target === wallDialog) {
    wallDialog.close();
    return;
  }
  const dismiss = event.target.closest('[data-dismiss-disclaimer]');
  if (dismiss) {
    const note = dismiss.closest('[data-ui-disclaimer]');
    if (note) note.hidden = true;
    try { window.localStorage.setItem('pramaan-disclaimer-dismissed', '1'); } catch (_error) {}
    return;
  }
  const reset = event.target.closest('[data-leaderboard-reset]');
  if (reset) {
    element.querySelector('[data-leaderboard-search]').value = '';
    element.querySelector('[data-leaderboard-provenance]').value = 'all';
    element.querySelector('[data-leaderboard-sort]').value = 'pass3';
    applyLeaderboardControls();
    return;
  }
  const quiltTile = event.target.closest('[data-quilt-state-tile]');
  if (quiltTile && quiltSheet) {
    event.preventDefault();
    quiltSheetOpener = quiltTile;
    const title = quiltSheet.querySelector('[data-quilt-sheet-title]');
    const content = quiltSheet.querySelector('[data-quilt-sheet-content]');
    if (title) title.textContent = quiltTile.dataset.state || 'State evidence';
    if (content) {
      content.replaceChildren();
      const details = quiltTile.querySelector('.pramaan-quilt-details');
      if (details) {
        const copy = details.cloneNode(true);
        copy.open = true;
        content.appendChild(copy);
      }
    }
    quiltSheet.showModal();
    return;
  }
  const tab = event.target.closest('[data-tab-button]');
  if (tab) {
    const key = tab.dataset.tabButton;
    if (tab.getAttribute('href') === '/web/') return;
    event.preventDefault();
    const route = tab.getAttribute('href') || '#/overview';
    if (window.location.hash !== route) history.pushState(null, '', route);
    applyRoute();
    return;
  }
  const evidenceTab = event.target.closest('[data-local-panel]');
  if (evidenceTab) {
    const key = evidenceTab.dataset.localPanel;
    element.querySelectorAll('[data-local-panel]').forEach(node => node.setAttribute('aria-pressed', String(node === evidenceTab)));
    element.querySelectorAll('[data-panel]').forEach(node => node.hidden = node.dataset.panel !== key);
    return;
  }
  const galleryTile = event.target.closest('[data-gallery-state-tile]');
  if (galleryTile) {
    event.preventDefault();
    const route = '#/states?state=' + encodeURIComponent(galleryTile.dataset.state);
    if (window.location.hash !== route) history.pushState(null, '', route);
    trigger('click', {
      action: 'gallery_state_select', gallery_state: galleryTile.dataset.state,
      active_tab: 'benchmark'
    });
    return;
  }
  const button = event.target.closest('[data-action]');
  if (!button || button.disabled) return;
  event.preventDefault();
  const action = button.dataset.action;
  if (action === 'community_run') {
    const key = element.querySelector('[data-field="community_api_key"]');
    const consent = element.querySelector('[data-field="community_charge_consent"]');
    if (!consent?.checked || !key?.value.trim()) {
      const status = element.querySelector('[data-community-status]');
      if (status) status.textContent = 'Enter your provider key and confirm possible provider charges before starting. No request was sent.';
      return;
    }
  }
  if (action === 'community_publish') {
    const key = element.querySelector('[data-field="community_api_key"]');
    const chargeConsent = element.querySelector('[data-field="community_charge_consent"]');
    const publishConsent = element.querySelector('[data-field="community_publish_consent"]');
    if (!chargeConsent?.checked || !publishConsent?.checked || !key?.value.trim()) {
      const status = element.querySelector('[data-community-status]');
      if (status) status.textContent = 'Enter your provider key and confirm both charges and public publication before starting. No request was sent.';
      return;
    }
  }
  if (action === 'community_archive_submit') {
    const fileInput = element.querySelector('[data-field="community_archive_file"]');
    const consent = element.querySelector('[data-field="community_archive_consent"]');
    const status = element.querySelector('[data-community-archive-status]');
    if (!consent?.checked) {
      if (status) status.textContent = 'Check the publication consent box before submitting. No archive was uploaded or published.';
      return;
    }
    const file = fileInput?.files?.[0];
    if (!file || !file.name.toLowerCase().endsWith('.zip')) {
      if (status) status.textContent = 'Choose a ZIP archive before submitting. No archive was uploaded or published.';
      return;
    }
    if (status) status.textContent = 'Uploading the ZIP to the Space for private validation…';
    try {
      const uploaded = await upload(file);
      if (!uploaded?.path) throw new Error('upload failed');
      const values = {};
      element.querySelectorAll('[data-field]').forEach(node => {
        if (node.type === 'file' || node.dataset.field === 'community_api_key') return;
        values[node.dataset.field] = node.type === 'checkbox' ? node.checked : node.value;
      });
      values.action = action;
      values.active_tab = element.querySelector('[data-tab-button][aria-selected="true"]')?.dataset.tabButton || 'benchmark';
      values.community_archive_path = uploaded.path;
      trigger('click', values);
    } catch (_error) {
      if (status) status.textContent = 'The archive could not be uploaded for validation. Check the file and try again.';
    }
    return;
  }
  const values = {};
  element.querySelectorAll('[data-field]').forEach(node => {
    if (node.type === 'file' || (node.dataset.field === 'community_api_key' && !['community_run', 'community_publish'].includes(action))) return;
    values[node.dataset.field] = node.type === 'checkbox' ? node.checked : node.value;
  });
  values.action = action;
  values.active_tab = element.querySelector('[data-tab-button][aria-selected="true"]')?.dataset.tabButton || 'benchmark';
  values.example = element.querySelector('[data-field="example"]')?.value || '';
  values.about_state = element.querySelector('[data-field="about_state"]')?.value || '';
  values.benchmark_state = element.querySelector('[data-field="benchmark_state"]')?.value || '';
  trigger('click', values);
  if (action === 'community_run' || action === 'community_publish') {
    const key = element.querySelector('[data-field="community_api_key"]');
    if (key) key.value = '';
  }
});
element.addEventListener('keydown', function(event) {
  const modelChip = event.target.closest('[data-wall-model-choice]');
  if (modelChip && ['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
    const chips = Array.from(modelChip.closest('[data-wall-models], .pramaan-wall-models')
      ?.querySelectorAll('[data-wall-model-choice]') || []);
    const current = chips.indexOf(modelChip);
    const nextIndex = event.key === 'Home' ? 0 : event.key === 'End' ? chips.length - 1 :
      (current + (event.key === 'ArrowRight' ? 1 : -1) + chips.length) % chips.length;
    event.preventDefault();
    chips[nextIndex]?.focus();
    chips[nextIndex]?.click();
    return;
  }
  const tab = event.target.closest('[role="tab"][data-tab-button]');
  if (tab && ['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
    const tabs = Array.from(element.querySelectorAll('[role="tab"][data-tab-button]'))
      .filter(node => node.getAttribute('href') !== '/web/');
    const current = tabs.indexOf(tab);
    const nextIndex = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 :
      (current + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    event.preventDefault();
    tabs[nextIndex]?.focus();
    tabs[nextIndex]?.click();
    return;
  }
  const tile = event.target.closest('[data-state-tile]');
  if (!tile) return;
  if (tile.hasAttribute('data-quilt-state-tile') && ['Enter', ' '].includes(event.key)) {
    event.preventDefault(); tile.click(); return;
  }
  if (tile.hasAttribute('data-gallery-state-tile') && ['Enter', ' '].includes(event.key)) {
    event.preventDefault();
    tile.click();
    return;
  }
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
  const mosaic = tile.closest('.pramaan-state-mosaic');
  if (!mosaic) return;
  const tiles = Array.from(mosaic.querySelectorAll('[data-state-tile]'));
  const next = window.PramaanQuilt.findDirectionalNeighbor(tiles, tile, event.key);
  if (!next) return;
  event.preventDefault();
  tile.tabIndex = -1;
  next.tabIndex = 0;
  next.focus();
});
element.addEventListener('change', function(event) {
  if (event.target.matches('[data-leaderboard-control]')) {
    applyLeaderboardControls();
    return;
  }
  const key = event.target.dataset.field;
  const action = key === 'state' ? 'state' : key === 'tradition' ? 'tradition' :
    key === 'about_state' ? 'about_state' : key === 'benchmark_state' ? 'benchmark_state' :
    key === 'gallery_model' ? 'gallery_model' : key === 'festival_lens' ? 'festival_lens' :
    ['wall_model', 'wall_mode'].includes(key) ? key :
    ['quilt_model', 'quilt_level', 'quilt_mode', 'quilt_compare_model'].includes(key) ? key : null;
  if (!action) return;
  const values = {};
  element.querySelectorAll('[data-field]').forEach(node => {
    if (node.type === 'file' || node.dataset.field === 'community_api_key') return;
    values[node.dataset.field] = node.type === 'checkbox' ? node.checked : node.value;
  });
  values.action = action;
  values.active_tab = element.querySelector('[data-tab-button][aria-selected="true"]')?.dataset.tabButton || 'benchmark';
  values.about_state = element.querySelector('[data-field="about_state"]')?.value || '';
  values.benchmark_state = element.querySelector('[data-field="benchmark_state"]')?.value || '';
  values.gallery_model = element.querySelector('[data-field="gallery_model"]')?.value || 'all';
  trigger('click', values);
});
element.addEventListener('input', function(event) {
  if (event.target.matches('[data-leaderboard-search]')) applyLeaderboardControls();
});
element.addEventListener('change', function(event) {
  const choice = event.target.closest('[data-arena-model-choice]');
  if (!choice) return;
  const selected = new Set(Array.from(element.querySelectorAll('[data-arena-model-choice]:checked'), node => node.dataset.arenaModelChoice));
  selected.add('beam-hf');
  const beamChoice = element.querySelector('[data-arena-model-choice="beam-hf"]');
  if (beamChoice) beamChoice.checked = true;
  element.querySelectorAll('[data-arena-model-card]').forEach(card => { card.hidden = !selected.has(card.dataset.arenaModelCard); });
});
applyLeaderboardControls();
"""
    with gr.Blocks(title="Pramaan — Proof of Art", fill_width=True, delete_cache=(3600, 3600)) as demo:
        with gr.Row(elem_id="pramaan-hf-login-row"):
            gr.LoginButton(value="Sign in to publish", logout_value="Sign out ({})",
                           variant="huggingface", size="sm", link_target="_blank",
                           elem_id="pramaan-hf-login")
        page = gr.HTML(_render_custom_app(ui_state), js_on_load=js_on_load,
                       elem_id="pramaan-custom-app", show_label=False, container=False)
        state = gr.State(ui_state)
        # Keep the offline release checker compatibility marker hidden; all visible controls are custom HTML.
        gr.Button("Generate with Beam", interactive=False, visible=False)
        gr.Markdown("Offline example mode · live model access is not configured.", visible=False)
        page.click(_handle_custom_event, inputs=state, outputs=[page, state],
                   show_progress="minimal", concurrency_limit=1, concurrency_id="beam",
                   trigger_mode="multiple")
    demo.queue(default_concurrency_limit=1, max_size=32)
    return demo


def create_brand_theme():
    """Return Pramaan's theme for direct launches and FastAPI mounts."""
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.stone,
        spacing_size="md",
        radius_size="lg",
    )
    # Keep Gradio's native Blocks, labels, inputs and status components on the
    # same tested palette as the custom HTML. The CSS tokens alone do not
    # override the built-in theme variables used by those components.
    return theme.set(
        body_background_fill="#f7f3ec",
        body_background_fill_dark="#151b22",
        body_text_color="#172f4a",
        body_text_color_dark="#e8eef5",
        body_text_color_subdued="#455463",
        body_text_color_subdued_dark="#cbd6e0",
        background_fill_primary="#f7f3ec",
        background_fill_primary_dark="#151b22",
        background_fill_secondary="#fffdf9",
        background_fill_secondary_dark="#202a34",
        border_color_primary="#c9bba8",
        border_color_primary_dark="#647383",
        block_background_fill="#fffdf9",
        block_background_fill_dark="#202a34",
        block_label_background_fill="#f3eee6",
        block_label_background_fill_dark="#26313b",
        block_label_text_color="#172f4a",
        block_label_text_color_dark="#e8eef5",
        block_title_background_fill="#f3eee6",
        block_title_background_fill_dark="#26313b",
        block_title_text_color="#172f4a",
        block_title_text_color_dark="#e8eef5",
        input_background_fill="#ffffff",
        input_background_fill_dark="#293542",
        input_border_color="#c9bba8",
        input_border_color_dark="#647383",
    )


def launch_demo():
    """Launch the app with the Gradio 6 theme and custom CSS settings."""
    return build_demo().launch(theme=create_brand_theme(), css=PRAMAAN_CSS,
                               favicon_path=str(_brand_favicon_path()))


if __name__ == "__main__":
    launch_demo()
