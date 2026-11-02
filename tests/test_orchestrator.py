import json
from types import SimpleNamespace

from pramaan.orchestrate import MAX_ATTEMPTS, MAX_BRIEF_CHARS, _parse_task_candidate, apply_host_profile, generate_and_verify, require_integer_control
from pramaan.prompts import (
    SYSTEM_PROMPT_V2, design_prompt, design_prompt_for_task, repair_prompt_for_task,
    system_prompt_for_task,
)
from tests.helpers import pattern_obj


class FixtureClient:
    def __init__(self):
        self.calls = 0
        self.valid = json.loads(pattern_obj().model_dump_json())
        self.invalid = json.loads(json.dumps(self.valid))
        self.invalid["layers"][0]["motifs"] = self.invalid["layers"][0]["motifs"][:1]

    def complete(self, system, prompt):
        self.calls += 1
        body = self.invalid if self.calls == 1 else self.valid
        return json.dumps(body), {"model_id": "fixture", "provider": "test"}


def test_beam_repair_attempts_pass_and_preserve_constraint_profile():
    client = FixtureClient()
    result = generate_and_verify("balanced geometry", ["#000000", "#FFFFFF"], client=client)
    assert result["accepted"] is True
    assert len(result["attempts"]) == 2
    assert result["attempts"][0]["report"]["all_hard_constraints_pass"] is False
    assert result["certificate"]["all_hard_constraints_pass"] is True
    assert result["trajectory_schema_version"] == "1.0"
    assert [call["phase"] for call in result["model_calls"]] == ["proposal", "repair"]
    assert result["model_calls"][0]["messages"][0]["role"] == "system"
    assert result["model_calls"][0]["raw_response"] == result["attempts"][0]["raw"]
    assert result["attempts"][1]["call_index"] == 2


def test_task_aware_run_records_task_fields():
    task = SimpleNamespace(
        id="gujarat-ajrakh-easy-0", suite_id="eval-v2", state="Gujarat",
        tradition_id="gj_ajrakh", grammar="periodic_raster", level="easy", symmetry="D4",
        constraint_profile={"symmetry": "D4"}, palette=["#000000", "#FFFFFF"],
        brief="balanced geometry",
    )
    result = generate_and_verify(client=FixtureClient(), task=task)
    assert result["accepted"] is True
    assert {key: result[key] for key in ("task_id", "state", "tradition_id", "grammar", "suite_id", "level")} == {
        "task_id": task.id, "state": task.state, "tradition_id": task.tradition_id,
        "grammar": task.grammar, "suite_id": task.suite_id, "level": task.level,
    }
    assert result["model_calls"][0]["messages"][1]["content"] == design_prompt_for_task(task)


def test_apply_host_profile_overrides_constraints_palette_background():
    task = SimpleNamespace(
        id="tn_kolam-easy-0", suite_id="eval-v2", state="Tamil Nadu",
        tradition_id="tn_sikku_kolam", grammar="kolam_loop", level="easy", symmetry="D2",
        constraint_profile={"required_loops": 2, "max_colours": 2},
        palette=["#123456", "#ABCDEF"], brief="two loops",
    )
    candidate = {
        "schema_version": "0.2", "grammar": "dot_field", "tradition_id": "wrong",
        "palette": ["#000000"], "background_index": 4,
        "constraints": {"symmetry": "D4", "required_loops": 99},
    }
    result = apply_host_profile(candidate, task)
    assert result["constraints"]["symmetry"] == "D2"
    assert result["constraints"]["required_loops"] == 2
    assert result["constraints"]["max_colours"] == 2
    assert result["palette"] == task.palette
    assert result["background_index"] == 0
    assert result["grammar"] == task.grammar
    assert result["tradition_id"] == task.tradition_id


def test_kolam_host_profile_forces_bounded_tile():
    task = SimpleNamespace(
        grammar="kolam_loop", tradition_id="tn_sikku_kolam", palette=["#123456"],
        symmetry="D4", constraint_profile={}, brief="one closed Sikku Kolam loop",
    )
    candidate = {"tile": {"width": 64, "height": 64, "wrap_x": True, "wrap_y": True}}

    result = apply_host_profile(candidate, task)

    assert result["tile"] == {"width": 64, "height": 64, "wrap_x": False, "wrap_y": False}
    assert candidate["tile"]["wrap_x"] is True
    assert '"wrap_x":false' in design_prompt_for_task(task)


def test_host_profile_normalizes_tile_shape_and_wrap_by_grammar():
    cases = [
        ("counted_grid", {"cell_size": 4}, True),
        ("rotational_field", {}, False),
        ("jaali_lattice", {"frame": 4}, False),
    ]
    for grammar, constraint_profile, wraps in cases:
        task = SimpleNamespace(
            grammar=grammar, tradition_id="test_tradition", palette=["#123456"],
            symmetry="D2", constraint_profile=constraint_profile,
        )
        candidate = {"tile": [64, 64]}

        result = apply_host_profile(candidate, task)

        assert result["tile"] == {
            "width": 64, "height": 64, "wrap_x": wraps, "wrap_y": wraps,
        }
        assert candidate["tile"] == [64, 64]


def test_task_candidate_uses_pattern_v2_and_host_owned_dot_profile():
    task = SimpleNamespace(
        id="gj_dots-easy-0", suite_id="eval-v2", state="Gujarat",
        tradition_id="gj_dots", grammar="dot_field", level="easy", symmetry="D2",
        constraint_profile={"min_clearance": 3}, palette=["#123456", "#ABCDEF"],
        brief="separated dots",
    )
    candidate = {
        "schema_version": "0.2", "grammar": "jaali_lattice", "tradition_id": "wrong",
        "title": "Dots", "tile": {"width": 64, "height": 64, "wrap_x": True, "wrap_y": True},
        "palette": ["#000000"], "background_index": 7,
        "layers": [{"id": "dots", "colour_index": 1, "anchor": [0, 0],
                    "motifs": [{"kind": "dot", "center": [16, 16], "radius": 2}]}],
        "constraints": {"symmetry": "D4", "min_clearance": 0},
    }
    parsed = _parse_task_candidate(json.dumps(candidate), task)
    assert parsed.grammar == task.grammar
    assert parsed.tradition_id == task.tradition_id
    assert parsed.palette == task.palette
    assert parsed.background_index == 0
    assert parsed.constraints.symmetry == "D2"
    assert parsed.constraints.min_clearance == 3
    assert system_prompt_for_task(task) == SYSTEM_PROMPT_V2


def test_jaali_frame_is_host_owned():
    task = SimpleNamespace(
        grammar="jaali_lattice", tradition_id="gj_jaali", palette=["#123456", "#ABCDEF"],
        symmetry="D2", constraint_profile={"frame": 4, "min_web": 2},
    )
    candidate = {"layers": [], "jaali": {"frame": 1, "open_colour_index": 1, "holes": []}}
    prepared = apply_host_profile(candidate, task)
    assert prepared["jaali"]["frame"] == 4
    assert prepared["constraints"]["min_web"] == 2
    assert candidate["jaali"]["frame"] == 1
    assert '"frame": 4' in design_prompt_for_task(task)
    assert '"frame": 4' in repair_prompt_for_task(task, json.dumps(candidate), [])


def test_retry_limit_never_claims_verification():
    class AlwaysBad(FixtureClient):
        def complete(self, system, prompt):
            self.calls += 1
            return json.dumps(self.invalid), {"model_id": "fixture", "provider": "test"}
    result = generate_and_verify("unbalanced geometry", ["#000000", "#FFFFFF"], client=AlwaysBad())
    assert result["accepted"] is False
    assert len(result["attempts"]) == 3


def test_ui_owned_disjoint_layer_requirement_cannot_be_removed_by_candidate():
    client = FixtureClient()
    client.valid["layers"].append({"id": "second", "colour_index": 1, "anchor": [0, 0],
                                   "motifs": client.valid["layers"][0]["motifs"]})
    result = generate_and_verify("separate inks", ["#000000", "#FFFFFF"],
                                 exclusive_ink_layers=True, client=client)
    assert result["accepted"] is False
    check = next(c for c in result["attempts"][1]["report"]["checks"] if c["id"] == "exclusive_ink_collision")
    assert check["status"] == "FAIL"
    assert check["counterexample"]["pixel"]


def test_ui_owned_minimum_gap_survives_candidate_and_repair():
    client = FixtureClient()
    client.valid["constraints"]["min_gap"] = 0
    result = generate_and_verify("keep motifs spaced", ["#000000", "#FFFFFF"],
                                 min_gap=5, client=client)
    assert result["accepted"] is True
    assert result["constraint_profile"]["min_gap"] == 5
    assert all(item["candidate"]["constraints"]["min_gap"] == 5 for item in result["attempts"])


def test_oversized_brief_is_rejected_before_provider_call():
    client = FixtureClient()
    try:
        generate_and_verify("x" * (MAX_BRIEF_CHARS + 1), ["#000000", "#FFFFFF"], client=client)
    except ValueError as exc:
        assert "character limit" in str(exc)
    else:
        raise AssertionError("oversized brief was accepted")
    assert client.calls == 0


def test_initial_provider_error_returns_not_verified_without_attempt():
    class Unavailable:
        def complete(self, system, prompt):
            raise RuntimeError("sensitive provider response body")

    result = generate_and_verify("simple pattern", ["#000000", "#FFFFFF"], client=Unavailable())
    assert result["accepted"] is False
    assert result["attempts"] == []
    assert result["provider_error"] == {
        "phase": "proposal", "type": "RuntimeError", "status_code": None,
        "message": "sensitive provider response body",
    }
    assert result["provider_error"]["message"] == "sensitive provider response body"
    assert result["model_calls"][0]["error"] == result["provider_error"]
    assert result["model_calls"][0]["messages"][1]["content"]


def test_provider_error_message_redacts_prompt_keys_and_local_paths():
    prompt_secret = "private prompt phrase that must not escape"
    local_path = "D:" + chr(92) + chr(92).join(("Projects", "Proof-of-Art", "private", "run.jsonl"))

    class Unavailable:
        def complete(self, _system, _prompt):
            raise RuntimeError(
                f"Request failed for {prompt_secret}; key=sk-proj-1234567890abcdef "
                f"at {local_path}"
            )

    result = generate_and_verify(prompt_secret, ["#000000", "#FFFFFF"], client=Unavailable())
    error_json = json.dumps(result["provider_error"])
    for private_value in (prompt_secret, "sk-proj-1234567890abcdef", local_path):
        assert private_value not in error_json
    assert "[REDACTED_PROMPT]" in result["provider_error"]["message"]
    assert "[REDACTED_KEY]" in result["provider_error"]["message"]
    assert "[LOCAL_PATH]" in result["provider_error"]["message"]


def test_anthropic_refusal_error_keeps_only_a_bounded_category():
    class AnthropicRefusalError(RuntimeError):
        def __init__(self, category):
            self.category = category
            super().__init__("private provider refusal body")

    class RefusingClient:
        def __init__(self, category):
            self.error = AnthropicRefusalError(category)

        def complete(self, _system, _prompt):
            raise self.error

    safe = generate_and_verify("simple pattern", ["#000000", "#FFFFFF"],
                               client=RefusingClient("safety"))
    unsafe = generate_and_verify("simple pattern", ["#000000", "#FFFFFF"],
                                 client=RefusingClient("safety: private refusal detail"))

    assert safe["provider_error"] == {
        "phase": "proposal", "type": "AnthropicRefusalError", "status_code": None,
        "refusal_category": "safety",
    }
    assert safe["model_calls"][0]["error"] == safe["provider_error"]
    assert "private provider refusal body" not in json.dumps(safe)
    from benchmarks.run import SecretRedactingClient
    wrapped = generate_and_verify(
        "simple pattern", ["#000000", "#FFFFFF"],
        client=SecretRedactingClient(RefusingClient("safety"), "sk-secret-refusal-wrapper"))
    assert wrapped["provider_error"] == {
        "phase": "proposal", "type": "AnthropicRefusalError", "status_code": None,
        "refusal_category": "safety",
    }
    assert "message" not in wrapped["provider_error"]
    assert "private provider refusal body" not in json.dumps(wrapped)
    assert unsafe["provider_error"] == {
        "phase": "proposal", "type": "AnthropicRefusalError", "status_code": None,
    }
    assert "private provider refusal body" not in json.dumps(unsafe)


def test_attempt_limit_is_enforced_before_provider_call():
    client = FixtureClient()
    for invalid in (0, -1, MAX_ATTEMPTS + 1, True, 1.5):
        try:
            generate_and_verify("bounded retries", ["#000000", "#FFFFFF"],
                                max_attempts=invalid, client=client)
        except ValueError as exc:
            assert "max_attempts" in str(exc)
        else:
            raise AssertionError(f"invalid max_attempts value was accepted: {invalid!r}")
    assert client.calls == 0


def test_integral_ui_control_values_are_not_silently_truncated():
    assert require_integer_control(2, "control") == 2
    assert require_integer_control(2.0, "control") == 2
    for invalid in (2.5, float("nan"), float("inf"), True, "2"):
        try:
            require_integer_control(invalid, "control")
        except ValueError:
            continue
        raise AssertionError(f"invalid slider value was accepted: {invalid!r}")


def test_design_prompt_spells_out_exact_tile_schema_and_app_only_control():
    prompt = design_prompt("balanced geometry", ["#000000", "#FFFFFF"], {
        "symmetry": "D4", "exclusive_ink_layers": False,
    })
    assert 'tile object with exactly {"width":64,"height":64,"wrap_x":true,"wrap_y":true}' in prompt
    assert "Do not copy app-only controls such as exclusive_ink_layers" in prompt
    assert "no extra fields" in prompt


def test_repair_provider_error_preserves_failed_candidate_and_witness():
    class RepairUnavailable(FixtureClient):
        def complete(self, system, prompt):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("provider unavailable")
            return json.dumps(self.invalid), {"model_id": "fixture", "provider": "test"}

    client = RepairUnavailable()
    result = generate_and_verify("pattern with exact symmetry", ["#000000", "#FFFFFF"], client=client)
    assert result["accepted"] is False
    assert len(result["attempts"]) == 1
    assert result["attempts"][0]["candidate"] is not None
    assert result["attempts"][0]["report"]["all_hard_constraints_pass"] is False
    assert result["provider_error"]["phase"] == "repair"


def test_malformed_json_gets_bounded_format_repair():
    class MalformedThenValid(FixtureClient):
        def complete(self, system, prompt):
            self.calls += 1
            body = "not json" if self.calls == 1 else self.valid
            return body if isinstance(body, str) else json.dumps(body), {"model_id": "fixture", "provider": "test"}
    result = generate_and_verify("balanced geometry", ["#000000", "#FFFFFF"], client=MalformedThenValid())
    assert result["accepted"] is True
    assert result["attempts"][0]["candidate"] is None
    assert result["attempts"][0]["report"]["checks"][0]["id"] == "candidate_parse"


def test_truncated_completion_is_never_certified_even_if_candidate_parses():
    class TruncatedThenComplete(FixtureClient):
        def complete(self, system, prompt):
            self.calls += 1
            metadata = {"model_id": "fixture", "provider": "test"}
            if self.calls == 1:
                metadata["finish_reason"] = "length"
            else:
                metadata["finish_reason"] = "stop"
            return json.dumps(self.valid), metadata

    result = generate_and_verify("balanced geometry", ["#000000", "#FFFFFF"], client=TruncatedThenComplete())
    assert result["accepted"] is True
    first = result["attempts"][0]
    assert first["candidate"] is not None
    assert first["report"]["all_hard_constraints_pass"] is False
    assert any(check["id"] == "candidate_truncated" for check in first["report"]["checks"])
