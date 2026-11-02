import sys
import types
from unittest.mock import patch

from pramaan.beam_client import BASE_URL, MAX_COMPLETION_TOKENS, MODEL_ID, REASONING_EFFORT, BeamClient, OpenAICompatibleClient


def _response():
    usage = types.SimpleNamespace(model_dump=lambda: {"total_tokens": 12})
    choice = types.SimpleNamespace(message=types.SimpleNamespace(content='{"schema_version":"0.1"}'), finish_reason="stop")
    return types.SimpleNamespace(choices=[choice], usage=usage)


def test_client_uses_hugging_face_router_and_expected_beam_model():
    assert MODEL_ID == "reflection/Beam-1.0-501B-A23B:baseten"
    seen = {}
    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
        def create(self, **kwargs):
            seen["request"] = kwargs
            return _response()
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    with patch.dict(sys.modules, {"openai": fake_module}):
        client = BeamClient(token="test-token")
        content, metadata = client.complete("system", "prompt")
    assert seen["init"]["base_url"] == BASE_URL
    assert seen["request"]["model"] == MODEL_ID
    assert content == '{"schema_version":"0.1"}'
    assert seen["request"]["max_tokens"] == MAX_COMPLETION_TOKENS == 4096
    assert seen["request"]["reasoning_effort"] == REASONING_EFFORT == "minimal"
    assert metadata["request_params"] == {
        "temperature": 0.4, "max_tokens": MAX_COMPLETION_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
    }
    assert metadata["provider_attempts"] == 1
    assert metadata["provider_elapsed_ms"] >= 0


def test_transient_provider_failure_retries_with_bounded_backoff():
    seen = {"calls": 0, "sleeps": []}
    class TemporaryUnavailable(Exception):
        status_code = 503
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
        def create(self, **kwargs):
            seen["calls"] += 1
            if seen["calls"] == 1:
                raise TemporaryUnavailable()
            return _response()
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    with patch.dict(sys.modules, {"openai": fake_module}), patch("pramaan.beam_client.time.sleep", side_effect=seen["sleeps"].append):
        _, metadata = BeamClient(token="test-token").complete("system", "prompt")
    assert seen["calls"] == 2
    assert seen["sleeps"] == [1]
    assert metadata["provider_attempts"] == 2


def test_openai_compatible_adapter_uses_injected_model_configuration():
    seen = {}
    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen["init"] = kwargs
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
        def create(self, **kwargs):
            seen["request"] = kwargs
            return _response()
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    config = {"temperature": 0.1, "max_tokens": 123}
    with patch.dict(sys.modules, {"openai": fake_module}):
        client = OpenAICompatibleClient(token="test-token", model_id="vendor/model",
                                        provider="Example", base_url="https://provider.test/v1",
                                        request_params=config)
        _, metadata = client.complete("system", "prompt")
    assert seen["init"]["base_url"] == "https://provider.test/v1"
    assert seen["request"]["model"] == "vendor/model"
    assert seen["request"]["max_tokens"] == 123
    assert metadata["provider"] == "Example"


def test_openai_compatible_adapter_does_not_consume_beam_daily_budget():
    seen = {}
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **_kwargs: _response()))
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    with patch.dict(sys.modules, {"openai": fake_module}), \
            patch("pramaan.beam_client.MODEL_CALL_BUDGET.reserve", side_effect=AssertionError("Beam-only guard")):
        client = OpenAICompatibleClient(token="test-token", model_id="vendor/model",
                                        provider="Example", base_url="https://provider.test/v1",
                                        request_params={})
        content, _ = client.complete("system", "prompt")
    assert content == '{"schema_version":"0.1"}'


def test_beam_client_still_consumes_beam_daily_budget():
    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = lambda **_kwargs: types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(
            create=lambda **_request: _response())))
    with patch.dict(sys.modules, {"openai": fake_module}), \
            patch("pramaan.beam_client.MODEL_CALL_BUDGET.reserve") as reserve:
        BeamClient(token="test-token").complete("system", "prompt")
    reserve.assert_called_once_with()


def test_openai_compatible_adapter_rebuilds_client_after_any_exception():
    seen = {"instances": 0, "calls": 0, "closed": 0}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            seen["instances"] += 1
            self.instance = seen["instances"]
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))
        def create(self, **kwargs):
            seen["calls"] += 1
            if self.instance == 1:
                raise RuntimeError("client transport failed")
            return _response()
        def close(self):
            seen["closed"] += 1

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    with patch.dict(sys.modules, {"openai": fake_module}):
        client = OpenAICompatibleClient(token="test-token", model_id="vendor/model",
                                        provider="Example", base_url="https://provider.test/v1",
                                        request_params={}, max_transient_retries=0)
        try:
            client.complete("system", "first request")
        except RuntimeError:
            pass
        else:
            raise AssertionError("first transport failure was hidden")
        content, _ = client.complete("system", "second request")

    assert content == '{"schema_version":"0.1"}'
    assert seen == {"instances": 2, "calls": 2, "closed": 2}


def test_openai_compatible_adapter_rejects_insecure_endpoint():
    try:
        OpenAICompatibleClient(token="test-token", model_id="model", provider="provider",
                               base_url="http://provider.test/v1", request_params={})
    except ValueError as exc:
        assert "HTTPS" in str(exc)
    else:
        raise AssertionError("insecure model endpoint was accepted")
