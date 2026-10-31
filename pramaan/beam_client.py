"""Thin hosted Beam adapter. The app remains usable without a configured token."""

from __future__ import annotations

import os
import time
from urllib.parse import urlsplit

from .budget import MODEL_CALL_BUDGET

# Hugging Face currently lists Baseten as Beam's live Inference Provider.
# Pinning it avoids relying on a router policy suffix not documented for this model.
MODEL_ID = "reflection/Beam-1.0-501B-A23B:baseten"
BASE_URL = "https://router.huggingface.co/v1"
REASONING_EFFORT = "minimal"
MAX_COMPLETION_TOKENS = 4096


class OpenAICompatibleClient:
    """Provider-neutral chat-completion adapter for compatible endpoints."""

    def __init__(self, *, token: str, model_id: str, provider: str, base_url: str,
                 request_params: dict, pricing: dict | None = None,
                 timeout: float = 75.0, max_transient_retries: int = 2):
        if not token:
            raise RuntimeError("A provider token is required for model inference.")
        parsed_url = urlsplit(base_url)
        if (parsed_url.scheme != "https" or not parsed_url.hostname or parsed_url.username
                or parsed_url.password or parsed_url.query or parsed_url.fragment):
            raise ValueError("Model endpoint must be an HTTPS URL without embedded credentials or query parameters.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The OpenAI client is not installed. Install requirements.txt to enable hosted inference.") from exc
        self.model_id = model_id
        self.provider = provider
        self.request_params = dict(request_params)
        self.pricing = dict(pricing) if pricing else None
        self.max_transient_retries = max_transient_retries
        self.call_budget = None
        self._openai_client_factory = OpenAI
        self._client_kwargs = {"base_url": base_url, "api_key": token,
                               "timeout": timeout, "max_retries": 0}

    def complete(self, system: str, prompt: str) -> tuple[str, dict]:
        response = None
        provider_attempts = 0
        started = time.perf_counter()
        for attempt in range(self.max_transient_retries + 1):
            if self.call_budget is not None:
                self.call_budget.reserve()
            provider_attempts += 1
            client = self._openai_client_factory(**self._client_kwargs)
            try:
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    **self.request_params,
                )
                break
            except Exception as exc:
                status = getattr(exc, "status_code", None)
                if status not in (429, 500, 502, 503, 504) or attempt == self.max_transient_retries:
                    try:
                        exc.provider_error_type = type(exc).__name__
                        exc.provider_attempts = provider_attempts
                        exc.provider_elapsed_ms = round((time.perf_counter() - started) * 1000)
                    except Exception:
                        pass
                    raise
                time.sleep(2 ** attempt)
            finally:
                close = getattr(client, "close", None)
                if callable(close):
                    close()
        if response is None:
            raise RuntimeError("Beam request did not return a response")
        choice = response.choices[0]
        content = choice.message.content or ""
        usage = response.usage.model_dump() if response.usage else {}
        metadata = {
            "model_id": self.model_id,
            "provider": self.provider,
            "request_params": self.request_params,
            "usage": usage,
            "finish_reason": choice.finish_reason,
            "provider_attempts": provider_attempts,
            "provider_elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
        if self.pricing:
            metadata["pricing"] = self.pricing
        return content, metadata


class BeamClient(OpenAICompatibleClient):
    """Pramaan's Beam preset; the benchmark can inject any compatible model client."""

    def __init__(self, token: str | None = None):
        token = token or os.getenv("HF_TOKEN")
        if not token:
            raise RuntimeError("Live Beam generation is unavailable: set HF_TOKEN in the environment or Space Secrets.")
        super().__init__(
            token=token,
            model_id=MODEL_ID,
            provider="Hugging Face Inference Providers",
            base_url=BASE_URL,
            request_params={
                "temperature": 0.4,
                "max_tokens": MAX_COMPLETION_TOKENS,
                "reasoning_effort": REASONING_EFFORT,
            },
            pricing={
                "currency": "USD",
                "per_million_tokens": {"input": 0.40, "cached_input": 0.04, "output": 1.20},
                "source": "Baseten Model APIs pricing",
                "source_url": "https://www.baseten.co/pricing/",
                "observed_at": "2026-10-29",
            },
        )
        self.call_budget = MODEL_CALL_BUDGET
