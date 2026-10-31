"""Run paired first-proposal vs verifier-repair evaluations for configured models."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)

from pramaan.beam_client import OpenAICompatibleClient
from pramaan import VALIDATOR_VERSION
from pramaan.normalize import normalize_model_text
from pramaan.tasks import canonical_jsonl_bytes
from pramaan.orchestrate import generate_and_verify
from pramaan.budget import ModelCallBudget

PALETTE = ["#172F4A", "#A84436", "#F4E5C5", "#262321"]
BENCHMARK_VERSION = "pramaan-paired-shared-first-v2"
MODEL_SPEND_CAP_USD = 10.0
DEFAULT_INPUT_TOKEN_OVERHEAD = 256


def output_token_limit(request_params):
    """Return the single configured output cap across provider API field names."""
    values = [request_params[key] for key in ("max_tokens", "max_completion_tokens")
              if key in request_params]
    if len(values) != 1 or type(values[0]) is not int or values[0] < 1:
        raise ValueError("request parameters must set exactly one positive output token limit")
    return values[0]


class BudgetExceeded(BaseException):
    """Stop a model before spending beyond its cap; do not persist a partial prompt pair."""


@dataclass(frozen=True)
class ModelPrice:
    key: str
    provider: str
    input_per_million: float
    output_per_million: float

    def __post_init__(self):
        if not isinstance(self.provider, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", self.provider):
            raise ValueError("provider must be a safe provider key")
        if any(not math.isfinite(value) or value < 0 for value in
               (self.input_per_million, self.output_per_million)):
            raise ValueError("model prices must be finite and non-negative")


def preflight_cost(models, *, prompts, attempts_per_prompt, max_input_bytes,
                   input_token_overhead, max_output_tokens, caps=None):
    """Price worst-case requests, allowing one input token per UTF-8 byte."""
    if not models or len({item.key for item in models}) != len(models):
        raise ValueError("preflight requires unique model keys")
    values = (prompts, attempts_per_prompt, max_input_bytes, input_token_overhead, max_output_tokens)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("preflight counts and caps must be non-negative integers")
    calls = prompts * attempts_per_prompt
    input_bound = max_input_bytes + input_token_overhead
    totals = {"anthropic": 0.0, "cometapi": 0.0, "openai": 0.0}
    model_totals = {}
    for model in models:
        amount = calls * (input_bound * model.input_per_million
                          + max_output_tokens * model.output_per_million) / 1_000_000
        totals[model.provider] += amount
        model_totals[model.key] = round(amount, 8)
    model_caps = {item.key: MODEL_SPEND_CAP_USD for item in models}
    model_caps.update(caps or {})
    model_fits = {key: model_totals[key] <= model_caps[key] for key in model_totals}
    return {
        "calls_per_model": calls,
        "max_input_bytes_per_call": max_input_bytes,
        "input_token_overhead_per_call": input_token_overhead,
        "max_input_tokens_per_call": input_bound,
        "max_output_tokens_per_call": max_output_tokens,
        "model_usd": model_totals,
        "anthropic_usd": round(totals["anthropic"], 8),
        "cometapi_usd": round(totals["cometapi"], 8),
        "openai_usd": round(totals["openai"], 8),
        "model_cap_usd": model_caps,
        "model_fits": model_fits,
        "anthropic_fits": all(model_fits[item.key] for item in models if item.provider == "anthropic"),
        "cometapi_fits": all(model_fits[item.key] for item in models if item.provider == "cometapi"),
    }


_ledger_thread_lock = threading.RLock()


@contextlib.contextmanager
def _ledger_lock(path):
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file, _ledger_thread_lock:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class CostLedger:
    """Append-only reserve/settle ledger; orphan reservations remain charged."""

    def __init__(self, path, *, caps=None):
        self.path = Path(path)
        self.caps = {}
        self.caps.update(caps or {})

    def _events(self):
        events = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise RuntimeError(f"spend ledger is corrupt at line {line_number}") from exc
                    if not isinstance(event, dict):
                        raise RuntimeError(f"spend ledger line {line_number} is invalid")
                    events.append(event)
        return events

    @staticmethod
    def _totals(events, *, model_key=None):
        reservations = {}
        for event in events:
            rid = event.get("reservation_id")
            if not isinstance(rid, str):
                raise RuntimeError("spend ledger entry has no reservation id")
            if event.get("event") == "reserve":
                if rid in reservations:
                    raise RuntimeError("duplicate spend reservation")
                reservations[rid] = {
                    "provider": event.get("provider"),
                    "model_key": event.get("model_key", event.get("provider")),
                    "amount": event.get("amount_usd"),
                    "state": "pending",
                }
            elif event.get("event") in {"settle", "release"} and rid in reservations:
                reservation = reservations[rid]
                if event.get("provider") != reservations[rid]["provider"]:
                    raise RuntimeError("spend ledger provider changed during settlement")
                if event.get("model_key", reservations[rid]["model_key"]) != reservations[rid]["model_key"]:
                    raise RuntimeError("spend ledger model changed during settlement")
                if reservation["state"] != "pending":
                    raise RuntimeError("spend ledger reservation has multiple terminal events")
                if event.get("event") == "settle":
                    reservation.update(amount=event.get("amount_usd"), state="settled")
                else:
                    evidence = event.get("evidence")
                    if (not isinstance(event.get("reason"), str) or not event["reason"]
                            or not isinstance(evidence, dict)
                            or not isinstance(evidence.get("elapsed_ms"), int)
                            or isinstance(evidence.get("elapsed_ms"), bool)
                            or not 0 <= evidence["elapsed_ms"] < 50
                            or evidence.get("usage_reported") is not False
                            or evidence.get("provider_error") is not True
                            or event.get("released_amount_usd") != reservation["amount"]):
                        raise RuntimeError("spend ledger release is missing valid evidence")
                    reservation.update(amount=0.0, state="released")
            else:
                raise RuntimeError("spend ledger contains an invalid event")
        totals = {}
        for reservation in reservations.values():
            provider, amount = reservation["provider"], reservation["amount"]
            if not isinstance(provider, str) or not provider or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
                raise RuntimeError("spend ledger contains an invalid provider or amount")
            if model_key is not None and reservation["model_key"] != model_key:
                continue
            totals[provider] = totals.get(provider, 0.0) + amount
        return totals

    def _append(self, event):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as target:
            target.write(json.dumps(event, sort_keys=True) + "\n")
            target.flush()
            os.fsync(target.fileno())

    def committed_usd(self, provider, *, model_key=None):
        if not isinstance(provider, str) or not provider:
            raise ValueError("budget key must be a non-empty string")
        with _ledger_lock(self.path):
            return self._totals(self._events(), model_key=model_key).get(provider, 0.0)

    def available_usd(self, provider):
        cap = self.caps.get(provider, MODEL_SPEND_CAP_USD)
        return max(0.0, cap - self.committed_usd(provider))

    def committed_by_scope_usd(self, provider):
        with _ledger_lock(self.path):
            events = self._events()
            terminals = {event["reservation_id"]: event for event in events
                         if event.get("event") in {"settle", "release"}}
            result = {}
            for reserve in events:
                if reserve.get("event") != "reserve" or reserve.get("provider") != provider:
                    continue
                model_key = reserve.get("model_key", provider)
                scope = reserve.get("scope", "current-scope")
                terminal = terminals.get(reserve["reservation_id"])
                amount = float(terminal.get("amount_usd", 0.0) if terminal else reserve["amount_usd"])
                result.setdefault(scope, {})[model_key] = result.setdefault(scope, {}).get(model_key, 0.0) + amount
            return result

    def reserve(self, reservation_id, *, provider, amount_usd, model_key=None, scope=None):
        if not isinstance(provider, str) or not provider or not reservation_id or not math.isfinite(amount_usd) or amount_usd < 0:
            raise ValueError("invalid provider reservation")
        model_key = model_key or provider
        if not isinstance(model_key, str) or not model_key:
            raise ValueError("model key must be a non-empty string")
        with _ledger_lock(self.path):
            events = self._events()
            totals = self._totals(events)
            if any(item.get("reservation_id") == reservation_id for item in events):
                raise ValueError("reservation id already exists")
            committed, cap = totals.get(provider, 0.0), self.caps.get(provider, MODEL_SPEND_CAP_USD)
            if committed + amount_usd > cap + 1e-12:
                raise BudgetExceeded(f"{provider} cap USD {cap:.2f} would be exceeded")
            event = {"event": "reserve", "reservation_id": reservation_id,
                     "provider": provider, "model_key": model_key, "amount_usd": amount_usd}
            if scope:
                event["scope"] = scope
            self._append(event)

    def release(self, reservation_id, *, provider, model_key=None, reason, evidence):
        """Release a reservation only with explicit evidence the call never reached a provider."""
        if (not isinstance(provider, str) or not provider or not isinstance(reason, str) or not reason
                or not isinstance(evidence, dict) or evidence.get("usage_reported") is not False
                or evidence.get("provider_error") is not True
                or not isinstance(evidence.get("elapsed_ms"), int)
                or isinstance(evidence.get("elapsed_ms"), bool)
                or not 0 <= evidence["elapsed_ms"] < 50):
            raise ValueError("release requires a failed provider call under 50 ms with no usage")
        with _ledger_lock(self.path):
            events = self._events()
            self._totals(events)
            reserve = next((item for item in events
                            if item.get("event") == "reserve" and item.get("reservation_id") == reservation_id), None)
            if not reserve or reserve.get("provider") != provider:
                raise ValueError("cannot release an unknown reservation")
            if model_key is not None and reserve.get("model_key", provider) != model_key:
                raise ValueError("cannot release a reservation for another model")
            if any(item.get("reservation_id") == reservation_id and item.get("event") in {"settle", "release"}
                   for item in events):
                raise ValueError("cannot release a completed reservation")
            event = {"event": "release", "reservation_id": reservation_id,
                     "provider": provider, "model_key": model_key or reserve.get("model_key", provider),
                     "released_amount_usd": reserve["amount_usd"], "reason": reason,
                     "evidence": evidence}
            if reserve.get("scope"):
                event["scope"] = reserve["scope"]
            self._append(event)

    def reconcile(self, reservation_id, *, provider, actual_usd, estimated_usd, usage_reported,
                  model_key=None, scope=None):
        if not usage_reported and actual_usd is None:
            with _ledger_lock(self.path):
                events = self._events()
                reserves = {event["reservation_id"]: event for event in events
                            if event.get("event") == "reserve"}
                reserve = reserves.get(reservation_id)
                if not reserve or reserve.get("provider") != provider:
                    raise ValueError("cannot reconcile unknown reservation")
                existing = next((event for event in events if event.get("event") in {"settle", "release"}
                                 and event.get("reservation_id") == reservation_id), None)
                if existing:
                    if existing.get("event") == "release":
                        raise ValueError("cannot settle a released reservation")
                    return float(existing["amount_usd"])
                amount = float(reserve["amount_usd"])
                event = {"event": "settle", "reservation_id": reservation_id,
                         "provider": provider, "model_key": model_key or reserve.get("model_key", provider),
                         "amount_usd": amount, "actual_usd": None, "estimated_usd": None,
                         "usage_reported": False}
                if scope:
                    event["scope"] = scope
                self._append(event)
                return amount
        values = [value for value in (actual_usd, estimated_usd) if value is not None]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("settled costs must be finite and non-negative")
        settled = max(values, default=0.0)
        with _ledger_lock(self.path):
            events = self._events()
            reserves = {event["reservation_id"]: event for event in events if event.get("event") == "reserve"}
            existing = next((event for event in events
                             if event.get("event") in {"settle", "release"}
                             and event.get("reservation_id") == reservation_id), None)
            if existing:
                if existing.get("event") == "release":
                    raise ValueError("cannot settle a released reservation")
                return settled
            reserve = reserves.get(reservation_id)
            if not reserve or reserve.get("provider") != provider:
                raise ValueError("cannot reconcile unknown reservation")
            event = {"event": "settle", "reservation_id": reservation_id,
                     "provider": provider, "model_key": model_key or reserve.get("model_key", provider),
                     "amount_usd": settled, "actual_usd": actual_usd,
                     "estimated_usd": estimated_usd, "usage_reported": bool(usage_reported)}
            if scope:
                event["scope"] = scope
            self._append(event)
        return settled

    def import_prior_ledger(self, source_path, *, source_provider, budget_key, model_key, scope):
        """Copy prior usage into this ledger without changing the source ledger."""
        source = CostLedger(source_path)
        events = source._events()
        self._totals(events)
        reserves = {event["reservation_id"]: event for event in events
                    if event.get("event") == "reserve" and event.get("provider") == source_provider}
        settles = {event["reservation_id"]: event for event in events
                   if event.get("event") == "settle" and event.get("provider") == source_provider}
        releases = {event["reservation_id"]: event for event in events
                    if event.get("event") == "release" and event.get("provider") == source_provider}
        for reservation_id, event in reserves.items():
            imported_id = f"prior:{model_key}:{reservation_id}"
            self.reserve(imported_id, provider=budget_key, model_key=model_key,
                         amount_usd=float(event["amount_usd"]), scope=scope)
            settlement = settles.get(reservation_id)
            if settlement is not None:
                self.reconcile(
                    imported_id, provider=budget_key, model_key=model_key, scope=scope,
                    actual_usd=settlement.get("actual_usd"),
                    estimated_usd=settlement.get("estimated_usd"),
                    usage_reported=bool(settlement.get("usage_reported")),
                )
            elif reservation_id in releases:
                release = releases[reservation_id]
                self.release(imported_id, provider=budget_key, model_key=model_key,
                             reason=release["reason"], evidence=release["evidence"])
        return len(reserves)

class GuardedCompletionClient:
    def __init__(self, client, *, ledger, model_key, provider, model_price,
                 max_input_bytes, input_token_overhead, max_output_tokens, budget_key=None):
        self.client, self.ledger = client, ledger
        self.model_key, self.provider, self.model_price = model_key, provider, model_price
        self.budget_key = budget_key or model_key
        self.max_input_bytes, self.input_token_overhead = max_input_bytes, input_token_overhead
        self.max_output_tokens = max_output_tokens

    def _request_reservation(self, system, prompt):
        input_bytes = len(system.encode("utf-8")) + len(prompt.encode("utf-8"))
        if input_bytes > self.max_input_bytes:
            raise BudgetExceeded(f"request needs {input_bytes} UTF-8 bytes; cap is {self.max_input_bytes}")
        input_bound = input_bytes + self.input_token_overhead
        reserve = (input_bound * self.model_price.input_per_million
                   + self.max_output_tokens * self.model_price.output_per_million) / 1_000_000
        return input_bytes, input_bound, reserve

    def can_reserve_complete_question(self, system, prompt, max_attempts=3):
        if type(max_attempts) is not int or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        _input_bytes, _input_bound, request_reserve = self._request_reservation(system, prompt)
        return self.ledger.available_usd(self.budget_key) + 1e-12 >= max_attempts * request_reserve

    def complete(self, system, prompt):
        input_bytes, input_bound, reserve = self._request_reservation(system, prompt)
        rid = str(uuid.uuid4())
        self.ledger.reserve(rid, provider=self.budget_key, model_key=self.model_key, amount_usd=reserve)
        call_started = time.perf_counter()
        try:
            content, metadata = self.client.complete(system, prompt)
        except Exception as exc:
            elapsed = getattr(exc, "provider_elapsed_ms", None)
            if not isinstance(elapsed, int) or isinstance(elapsed, bool) or elapsed < 0:
                elapsed = round((time.perf_counter() - call_started) * 1000)
            usage_reported = bool(getattr(exc, "usage_reported", False))
            if elapsed < 50 and not usage_reported:
                evidence = {
                    "elapsed_ms": elapsed,
                    "usage_reported": False,
                    "provider_error": True,
                    "provider_error_type": getattr(exc, "provider_error_type", type(exc).__name__),
                    "status_code": getattr(exc, "status_code", None),
                }
                self.ledger.release(
                    rid, provider=self.budget_key, model_key=self.model_key,
                    reason="provider_call_failed_before_send", evidence=evidence)
                exc.budget_guard = {"reservation_id": rid, "released": True,
                                    "released_amount_usd": reserve, "elapsed_ms": elapsed}
            raise
        metadata = dict(metadata or {})
        usage = metadata.get("usage") if isinstance(metadata.get("usage"), dict) else {}
        reported = usage.get("cost_usd", metadata.get("cost_usd"))
        if reported is None:
            cost = metadata.get("cost", usage.get("cost"))
            if isinstance(cost, dict):
                amount = cost.get("amount", cost.get("value", cost.get("total")))
                currency = cost.get("currency", metadata.get("cost_currency", usage.get("currency")))
                if isinstance(amount, (int, float)) and not isinstance(amount, bool) and currency == "USD":
                    reported = amount
            elif isinstance(cost, (int, float)) and not isinstance(cost, bool):
                if metadata.get("cost_currency", usage.get("cost_currency", usage.get("currency"))) == "USD":
                    reported = cost
        if not isinstance(reported, (int, float)) or isinstance(reported, bool):
            reported = None
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        estimated = None
        if isinstance(input_tokens, int) and isinstance(output_tokens, int) and input_tokens >= 0 and output_tokens >= 0:
            estimated = (input_tokens * self.model_price.input_per_million
                         + output_tokens * self.model_price.output_per_million) / 1_000_000
        usage_reported = isinstance(input_tokens, int) and isinstance(output_tokens, int)
        settled = self.ledger.reconcile(rid, provider=self.budget_key, model_key=self.model_key,
                                        actual_usd=reported, estimated_usd=estimated,
                                        usage_reported=usage_reported)
        metadata["budget_guard"] = {
            "reservation_id": rid, "model_key": self.model_key, "budget_key": self.budget_key,
            "provider": self.provider,
            "input_bytes": input_bytes, "input_token_upper_bound": input_bound,
            "max_output_tokens": self.max_output_tokens, "reserved_usd": reserve,
            "settled_usd": settled, "settlement_source": "provider_reported" if reported is not None
            else "rate_card_estimate" if estimated is not None else "reservation_retained",
        }
        return content, metadata


class AnthropicRefusalError(RuntimeError):
    """Bounded refusal signal that preserves only the provider's category."""

    def __init__(self, category):
        value = str(category or "unknown")[:64]
        self.category = value if re.fullmatch(r"[A-Za-z0-9_.-]+", value) else "unknown"
        super().__init__(f"Anthropic refused request ({self.category})")


class AnthropicMessagesClient:
    """One-attempt adapter for the Anthropic Messages API."""

    def __init__(self, *, token, model_id, provider, base_url, request_params, pricing=None, timeout=75.0):
        parsed = urlsplit(base_url)
        if not token or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("Anthropic requires a token and a clean HTTPS endpoint")
        if "temperature" in request_params:
            raise ValueError("Anthropic Messages does not accept temperature")
        from anthropic import Anthropic

        self.token, self.model_id, self.provider = token, model_id, provider
        self.base_url, self.request_params, self.pricing = base_url.rstrip("/"), dict(request_params), pricing
        self.timeout = timeout
        self.client = Anthropic(api_key=token, base_url=self.base_url, max_retries=0, timeout=timeout)

    def complete(self, system, prompt):
        params = dict(self.request_params)
        max_tokens = params.pop("max_tokens", None)
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("Anthropic model config requires max_tokens")
        recorded_params = {"max_tokens": max_tokens, **params}
        body = {"model": self.model_id, "max_tokens": max_tokens, "system": system,
                "messages": [{"role": "user", "content": prompt}], **params}
        started = time.perf_counter()
        try:
            response = self.client.messages.create(**body)
        except Exception as exc:
            exc.provider_attempts = 1
            exc.provider_elapsed_ms = round((time.perf_counter() - started) * 1000)
            raise
        elapsed = round((time.perf_counter() - started) * 1000)
        stop_reason = response.stop_reason
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            error = AnthropicRefusalError(getattr(details, "category", None))
            error.provider_attempts = 1
            error.provider_elapsed_ms = elapsed
            raise error
        content = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        source_usage = response.usage
        in_tokens, out_tokens = source_usage.input_tokens, source_usage.output_tokens
        usage = {}
        if isinstance(in_tokens, int) and isinstance(out_tokens, int):
            usage = {"input_tokens": in_tokens, "output_tokens": out_tokens,
                     "prompt_tokens": in_tokens, "completion_tokens": out_tokens,
                     "total_tokens": in_tokens + out_tokens,
                     "prompt_tokens_details": {"cached_tokens": getattr(source_usage, "cache_read_input_tokens", 0)},
                     "cache_creation_input_tokens": getattr(source_usage, "cache_creation_input_tokens", None)}
        metadata = {"model_id": self.model_id,
                    "resolved_model_id": response.model,
                    "request_id": getattr(response, "_request_id", None),
                    "provider": self.provider, "request_params": recorded_params,
                    "usage": usage, "finish_reason": "length" if stop_reason == "max_tokens" else stop_reason,
                    "content_block_types": [getattr(block, "type", None) for block in response.content],
                    "provider_attempts": 1, "provider_elapsed_ms": elapsed}
        if self.pricing:
            metadata["pricing"] = self.pricing
        return content, metadata


class CometOpenAICompatibleClient:
    """One-attempt CometAPI adapter that records the returned model alias."""

    def __init__(self, *, token, model_id, provider, base_url, request_params, pricing=None,
                 timeout=75.0, daily_limit=1000):
        parsed = urlsplit(base_url)
        if not token or parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("CometAPI requires a token and a clean HTTPS endpoint")
        from openai import OpenAI
        self.model_id, self.provider = model_id, provider
        self.request_params, self.pricing = dict(request_params), pricing
        self._openai_client_factory = OpenAI
        self._client_kwargs = {"base_url": base_url, "api_key": token,
                               "timeout": timeout, "max_retries": 0}
        self.call_budget = ModelCallBudget(daily_limit)

    def complete(self, system, prompt):
        self.call_budget.reserve()
        started = time.perf_counter()
        client = self._openai_client_factory(**self._client_kwargs)
        try:
            response = client.chat.completions.create(
                model=self.model_id, messages=[{"role": "system", "content": system},
                                               {"role": "user", "content": prompt}], **self.request_params)
        except Exception as exc:
            exc.provider_attempts = 1
            exc.provider_elapsed_ms = round((time.perf_counter() - started) * 1000)
            exc.provider_error_type = type(exc).__name__
            raise
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        elapsed = round((time.perf_counter() - started) * 1000)
        choice = response.choices[0]
        usage = response.usage.model_dump() if response.usage else {}
        metadata = {"model_id": self.model_id,
                    "resolved_model_id": getattr(response, "model", None) or self.model_id,
                    "provider": self.provider, "request_params": self.request_params, "usage": usage,
                    "finish_reason": choice.finish_reason, "provider_attempts": 1,
                    "provider_elapsed_ms": elapsed}
        if self.pricing:
            metadata["pricing"] = self.pricing
        return choice.message.content or "", metadata


PAID_MODELS = [
    {"key": "claude-haiku-5-5", "provider": "Anthropic", "budget_provider": "anthropic",
     "budget_group": "claude-haiku-5-5",
     "api_style": "anthropic-messages", "model_id": "claude-haiku-5-5", "base_url": "https://api.anthropic.com",
     "api_key_env": "ANTHROPIC_API_KEY", "request_params": {"max_tokens": 4096, "output_config": {"effort": "low"}},
     "budget": {"cap_usd": 10.0, "max_input_bytes": 100000, "input_token_overhead": 256},
     "pricing": {"currency": "USD", "per_million_tokens": {"input": .1, "output": .5},
                 "source": "Anthropic first-party standard pricing; prompt <=100k tokens",
                 "source_url": "https://platform.claude.com/docs/en/about-claude/pricing", "observed_at": "2026-10-29"}},
    {"key": "claude-sonnet-5-5", "provider": "Anthropic", "budget_provider": "anthropic",
     "budget_group": "claude-sonnet-5-5",
     "api_style": "anthropic-messages", "model_id": "claude-sonnet-5-5", "base_url": "https://api.anthropic.com",
     "api_key_env": "ANTHROPIC_API_KEY", "request_params": {"max_tokens": 4096, "output_config": {"effort": "low"}},
     "budget": {"cap_usd": 10.0, "max_input_bytes": 100000, "input_token_overhead": 256},
     "pricing": {"currency": "USD", "per_million_tokens": {"input": 2., "output": 10.},
                 "source": "Anthropic first-party standard pricing",
                 "source_url": "https://platform.claude.com/docs/en/about-claude/pricing", "observed_at": "2026-10-29"}},
    {"key": "claude-opus-5-5", "provider": "Anthropic", "budget_provider": "anthropic",
     "budget_group": "claude-opus-5-5",
     "api_style": "anthropic-messages", "model_id": "claude-opus-5-5", "base_url": "https://api.anthropic.com",
     "api_key_env": "ANTHROPIC_API_KEY", "request_params": {"max_tokens": 4096, "output_config": {"effort": "low"}},
     "budget": {"cap_usd": 10.0, "max_input_bytes": 100000, "input_token_overhead": 256},
     "pricing": {"currency": "USD", "per_million_tokens": {"input": 4., "output": 20.},
                 "source": "Anthropic first-party standard pricing",
                 "source_url": "https://platform.claude.com/docs/en/about-claude/pricing", "observed_at": "2026-10-29"}},
    {"key": "claude-fable-5-1", "provider": "Anthropic", "budget_provider": "anthropic",
     "budget_group": "claude-fable-5-1",
     "api_style": "anthropic-messages", "model_id": "claude-fable-5-1", "base_url": "https://api.anthropic.com",
     "api_key_env": "ANTHROPIC_API_KEY", "request_params": {"max_tokens": 4096, "output_config": {"effort": "low"}},
     "budget": {"cap_usd": 10.0, "max_input_bytes": 100000, "input_token_overhead": 256},
     "pricing": {"currency": "USD", "per_million_tokens": {"input": 10., "output": 50.},
                 "source": "Anthropic first-party standard pricing",
                 "source_url": "https://platform.claude.com/docs/en/about-claude/pricing", "observed_at": "2026-10-29"}},
    {"key": "deepseek-v4.1-flash", "provider": "CometAPI", "budget_provider": "cometapi",
     "budget_group": "deepseek-v4.1-flash",
     "api_style": "openai-compatible-chat-completions", "model_id": "deepseek-v4.1-flash",
     "base_url": "https://api.cometapi.com/v1", "api_key_env": "COMETAPI_KEY",
     "request_params": {"temperature": .4, "max_tokens": 4096,
                        "extra_body": {"thinking": {"type": "disabled"}}},
     "budget": {"cap_usd": 10.0, "max_input_bytes": 100000, "input_token_overhead": 256},
     "pricing": {"currency": "USD", "per_million_tokens": {"input": .24, "output": .96},
                 "source": "CometAPI public base rate doubled for weekday peak windows; account card unconfirmed",
                 "source_url": "https://www.cometapi.com/models/deepseek/deepseek-v4-1-flash/", "observed_at": "2026-10-29"}},
    {"key": "gpt-6.1-sol", "provider": "OpenAI", "budget_provider": "openai", "budget_group": "gpt-6.1-sol",
     "api_style": "openai-compatible-chat-completions", "model_id": "gpt-6.1-sol",
     "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY",
     "request_params": {"reasoning_effort": "low", "max_completion_tokens": 4096},
     "budget": {"cap_usd": 10.0, "max_input_bytes": 100000, "input_token_overhead": 256},
     "pricing": {"currency": "USD", "per_million_tokens": {"input": 2.0, "cached_input": 0.10, "output": 10.0},
                 "source": "OpenAI standard API pricing, short context; cached credit ignored in cap guard",
                 "source_url": "https://developers.openai.com/api/docs/pricing", "observed_at": "2026-10-31"}},
    {"key": "gpt-6.1-terra", "provider": "OpenAI", "budget_provider": "openai", "budget_group": "gpt-6.1-terra",
     "api_style": "openai-compatible-chat-completions", "model_id": "gpt-6.1-terra",
     "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY",
     "request_params": {"reasoning_effort": "none", "max_completion_tokens": 4096},
     "budget": {"cap_usd": 10.0, "max_input_bytes": 100000, "input_token_overhead": 256},
     "pricing": {"currency": "USD", "per_million_tokens": {"input": 1.0, "cached_input": 0.05, "output": 5.0},
                 "source": "OpenAI standard API pricing, short context; cached credit ignored in cap guard",
                 "source_url": "https://developers.openai.com/api/docs/pricing", "observed_at": "2026-10-31"}},
    {"key": "gpt-6-luna", "provider": "OpenAI", "budget_provider": "openai", "budget_group": "gpt-6-luna",
     "api_style": "openai-compatible-chat-completions", "model_id": "gpt-6-luna",
     "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY",
     "request_params": {"reasoning_effort": "none", "max_completion_tokens": 4096},
     "budget": {"cap_usd": 10.0, "max_input_bytes": 100000, "input_token_overhead": 256},
     "pricing": {"currency": "USD", "per_million_tokens": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
                 "source": "OpenAI standard API pricing, short context; cached credit ignored in cap guard",
                 "source_url": "https://developers.openai.com/api/docs/pricing", "observed_at": "2026-10-31"}},
    {"key": "gpt-6-astra", "provider": "OpenAI", "budget_provider": "openai", "budget_group": "gpt-6-astra",
     "api_style": "openai-compatible-chat-completions", "model_id": "gpt-6-astra",
     "base_url": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY",
     "request_params": {"reasoning_effort": "low", "max_completion_tokens": 4096},
     "budget": {"cap_usd": 10.0, "max_input_bytes": 100000, "input_token_overhead": 256},
     "pricing": {"currency": "USD", "per_million_tokens": {"input": 10.0, "cached_input": 1.0, "output": 50.0},
                 "source": "OpenAI standard API pricing, short context; cached credit ignored in cap guard",
                 "source_url": "https://developers.openai.com/api/docs/pricing", "observed_at": "2026-10-31"}},
]


def load_model_registry(path, *, include_paid=False):
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != 1 or not isinstance(registry.get("models"), list):
        raise ValueError("model registry must use schema_version 1 and contain a models list")
    models = {}
    entries = list(registry["models"])
    if include_paid:
        entries.extend(json.loads(json.dumps(PAID_MODELS)))
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each model registry entry must be a JSON object")
        key = entry.get("key")
        required = ("provider", "api_style", "model_id", "base_url", "api_key_env", "request_params")
        if not isinstance(key, str) or not key or any(not entry.get(field) for field in required):
            raise ValueError("each model needs key, provider, api_style, model_id, base_url, api_key_env, and request_params")
        if entry["api_style"] not in {"openai-compatible-chat-completions", "anthropic-messages"}:
            raise ValueError(f"unsupported api_style for {key}: {entry['api_style']}")
        if any(not isinstance(entry[field], str) for field in ("provider", "model_id", "base_url", "api_key_env")):
            raise ValueError(f"model identity and endpoint fields must be text for {key}")
        parsed_url = urlsplit(entry["base_url"])
        if (parsed_url.scheme != "https" or not parsed_url.hostname or parsed_url.username
                or parsed_url.password or parsed_url.query or parsed_url.fragment):
            raise ValueError(f"model endpoint must be an HTTPS URL without embedded credentials or query for {key}")
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", entry["api_key_env"]):
            raise ValueError(f"api_key_env must be an uppercase environment variable name for {key}")
        if not isinstance(entry["request_params"], dict):
            raise ValueError(f"request_params must be a JSON object for {key}")
        if entry["api_style"] == "anthropic-messages" and "temperature" in entry["request_params"]:
            raise ValueError(f"Anthropic Messages does not accept temperature for {key}")
        pricing = entry.get("pricing")
        budget_provider = entry.get("budget_provider")
        if budget_provider is not None and budget_provider not in {"anthropic", "cometapi", "openai"}:
            raise ValueError(f"budget_provider is unsupported for {key}")
        budget = entry.get("budget")
        if budget_provider is not None:
            if not isinstance(budget, dict):
                raise ValueError(f"budget settings are required for guarded model {key}")
            if (type(budget.get("max_input_bytes")) is not int or budget["max_input_bytes"] < 1
                    or type(budget.get("input_token_overhead")) is not int
                    or budget["input_token_overhead"] < 0
                    or not isinstance(budget.get("cap_usd"), (int, float))
                    or isinstance(budget["cap_usd"], bool) or budget["cap_usd"] <= 0
                    or budget["cap_usd"] > MODEL_SPEND_CAP_USD):
                raise ValueError(f"budget settings are invalid for {key}")
            try:
                output_limit = output_token_limit(entry["request_params"])
            except ValueError as exc:
                raise ValueError(f"guarded model {key} needs one 4096-token output cap") from exc
            if pricing is None or output_limit != 4096:
                raise ValueError(f"guarded model {key} needs pricing and an output cap of 4096")
        if pricing is not None:
            if (not isinstance(pricing, dict) or not isinstance(pricing.get("currency"), str)
                    or not isinstance(pricing.get("source"), str) or not isinstance(pricing.get("observed_at"), str)
                    or not isinstance(pricing.get("per_million_tokens"), dict)):
                raise ValueError(f"pricing for {key} must declare currency, source, observed_at, and per_million_tokens")
            source_url = pricing.get("source_url")
            price_url = urlsplit(source_url) if isinstance(source_url, str) else None
            if not price_url or price_url.scheme != "https" or not price_url.hostname:
                raise ValueError(f"pricing source_url must be HTTPS for {key}")
            rates = pricing["per_million_tokens"]
            if any(not isinstance(rates.get(rate), (int, float)) or isinstance(rates.get(rate), bool)
                   or rates[rate] < 0 for rate in ("input", "output")):
                raise ValueError(f"pricing for {key} must include non-negative input and output token rates")
            cached_rate = rates.get("cached_input")
            if cached_rate is not None and (not isinstance(cached_rate, (int, float))
                                            or isinstance(cached_rate, bool) or cached_rate < 0):
                raise ValueError(f"cached_input pricing must be a non-negative number for {key}")
        if {"model", "messages", "stream"} & set(entry["request_params"]):
            raise ValueError(f"request_params cannot override model/messages or enable streaming for {key}")
        if key in models:
            raise ValueError(f"duplicate model key: {key}")
        models[key] = entry
    return models


def configured_token(env_name):
    token = os.getenv(env_name)
    if not token and env_name == "HF_TOKEN":
        from huggingface_hub import get_token

        token = get_token()
    return token


class SecretRedactingClient:
    """Keep an explicitly supplied provider secret out of exception text."""

    def __init__(self, client, secret):
        self.client, self.secret = client, secret

    def complete(self, system, prompt):
        try:
            return self.client.complete(system, prompt)
        except Exception as exc:
            message = str(exc)
            if self.secret:
                message = message.replace(self.secret, "[REDACTED]")
            safe = RuntimeError(message[:300])
            safe.provider_error_type = getattr(exc, "provider_error_type", type(exc).__name__)
            for name in ("provider_attempts", "provider_elapsed_ms"):
                if hasattr(exc, name):
                    setattr(safe, name, getattr(exc, name))
            for name in ("status_code", "request_id", "usage_reported", "category"):
                value = getattr(exc, name, None)
                if isinstance(value, (str, int)) and not isinstance(value, bool):
                    setattr(safe, name, value)
                    if name == "status_code":
                        safe.http_status_code = value
            raise safe from None


def make_client(model_config, ledger: CostLedger | None = None, *, api_key=None):
    if api_key is not None and (not isinstance(api_key, str) or not api_key.strip() or len(api_key) > 4096):
        raise ValueError("provider API key is invalid")
    token = api_key if api_key is not None else configured_token(model_config["api_key_env"])
    if not token:
        raise RuntimeError(f"Set {model_config['api_key_env']} for model {model_config['key']}.")
    try:
        if model_config["api_style"] == "anthropic-messages":
            client = AnthropicMessagesClient(
                token=token, model_id=model_config["model_id"], provider=model_config["provider"],
                base_url=model_config["base_url"], request_params=model_config["request_params"],
                pricing=model_config.get("pricing"),
            )
        elif model_config.get("budget_provider") == "cometapi":
            client = CometOpenAICompatibleClient(
                token=token, model_id=model_config["model_id"], provider=model_config["provider"],
                base_url=model_config["base_url"], request_params=model_config["request_params"],
                pricing=model_config.get("pricing"), daily_limit=1000,
            )
        else:
            client = OpenAICompatibleClient(
                token=token, model_id=model_config["model_id"], provider=model_config["provider"],
                base_url=model_config["base_url"], request_params=model_config["request_params"],
                pricing=model_config.get("pricing"),
                max_transient_retries=0 if model_config.get("budget_provider") else 2,
            )
    except Exception as exc:
        message = str(exc).replace(token, "[REDACTED]")[:1000]
        raise RuntimeError(message) from None
    client = SecretRedactingClient(client, token)
    budget = model_config.get("budget")
    if budget:
        rates = model_config["pricing"]["per_million_tokens"]
        budget_key = model_config.get("budget_group", model_config["key"])
        client = GuardedCompletionClient(
            client,
            ledger=ledger or CostLedger(
                ROOT / "benchmarks" / "results" / "provider-spend-ledger.jsonl",
                caps={budget_key: budget.get("cap_usd", MODEL_SPEND_CAP_USD)}),
            model_key=model_config["key"],
            provider=model_config["budget_provider"],
            model_price=ModelPrice(
                key=model_config["key"], provider=model_config["budget_provider"],
                input_per_million=float(rates["input"]), output_per_million=float(rates["output"]),
            ),
            max_input_bytes=budget["max_input_bytes"],
            input_token_overhead=budget.get("input_token_overhead", DEFAULT_INPUT_TOKEN_OVERHEAD),
            max_output_tokens=output_token_limit(model_config["request_params"]),
            budget_key=budget_key,
        )
    return client


def _price_for_config(model_config):
    rates = model_config["pricing"]["per_million_tokens"]
    return ModelPrice(key=model_config["key"], provider=model_config["budget_provider"],
                      input_per_million=float(rates["input"]), output_per_million=float(rates["output"]))


def paid_run_freeze(model_configs, suite_sha256):
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "suite_sha256": suite_sha256,
        "prompt_count": 72,
        "attempts_per_prompt": 3,
        "models": {
            key: {
                field: config.get(field)
                for field in ("model_id", "provider", "budget_provider", "api_style", "base_url",
                              "request_params", "pricing", "budget")
            }
            for key, config in sorted(model_configs.items())
        },
        "policy": "one stochastic pass; resume only missing model/prompt pairs",
    }


def _input_guard_limits(freeze: dict) -> dict[str, int]:
    models = freeze.get("models") if isinstance(freeze, dict) else None
    if not isinstance(models, dict):
        return {}
    return {key: config["budget"]["max_input_bytes"]
            for key, config in models.items()
            if isinstance(config, dict) and isinstance(config.get("budget"), dict)
            and type(config["budget"].get("max_input_bytes")) is int
            and config["budget"]["max_input_bytes"] > 0}


def _compatible_input_guard_increase(previous: dict, current: dict) -> dict[str, int] | None:
    """Allow only a non-decreasing input-safety ceiling change during resume."""
    try:
        old = json.loads(json.dumps(previous))
        new = json.loads(json.dumps(current))
    except (TypeError, ValueError):
        return None
    old_models, new_models = old.get("models"), new.get("models")
    if not isinstance(old_models, dict) or not isinstance(new_models, dict) or old_models.keys() != new_models.keys():
        return None
    limits = {}
    for key in old_models:
        old_budget = old_models[key].get("budget") if isinstance(old_models[key], dict) else None
        new_budget = new_models[key].get("budget") if isinstance(new_models[key], dict) else None
        old_limit = old_budget.get("max_input_bytes") if isinstance(old_budget, dict) else None
        new_limit = new_budget.get("max_input_bytes") if isinstance(new_budget, dict) else None
        if type(old_limit) is not int or type(new_limit) is not int or new_limit < old_limit:
            return None
        limits[key] = new_limit
        new_budget["max_input_bytes"] = old_limit
    return limits if new == old and any(
        limits[key] > _input_guard_limits(previous).get(key, 0) for key in limits) else None


def _write_paid_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        target.flush()
        os.fsync(target.fileno())
    os.replace(temporary, path)


def save_or_validate_paid_freeze(path: Path, freeze: dict, preflight: dict, *, resume: bool) -> str:
    """Durably save settings before the first paid call and require them on resume."""
    if resume:
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("paid --resume requires its saved preflight freeze manifest") from exc
        original_freeze = previous.get("freeze")
        if not isinstance(original_freeze, dict):
            raise ValueError("paid resume settings differ from the saved preflight freeze")
        freeze_hash = hashlib.sha256(json.dumps(original_freeze, sort_keys=True).encode("utf-8")).hexdigest()
        if previous.get("freeze_sha256") != freeze_hash:
            raise ValueError("saved preflight freeze manifest failed its SHA-256 check")
        if original_freeze != freeze:
            new_limits = _compatible_input_guard_increase(original_freeze, freeze)
            if new_limits is None:
                raise ValueError("paid resume settings differ from the saved preflight freeze")
            old_limits = previous.get("effective_input_guard_bytes_by_model") or _input_guard_limits(original_freeze)
            if any(new_limits.get(key, 0) < old_limits.get(key, 0) for key in new_limits):
                raise ValueError("paid resume cannot reduce the effective input safety ceiling")
            if new_limits != old_limits:
                previous.setdefault("input_guard_updates", []).append({
                    "from_bytes_by_model": old_limits,
                    "to_bytes_by_model": new_limits,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                    "reason": "safety ceiling increase only; request text and model settings are unchanged",
                })
                previous["effective_input_guard_bytes_by_model"] = new_limits
                previous.setdefault("resume_preflights", []).append({
                    "input_guard_bytes_by_model": new_limits, "preflight": preflight,
                    "recorded_at": datetime.now(timezone.utc).isoformat(),
                })
                _write_paid_manifest(path, previous)
        return freeze_hash
    if path.exists():
        raise ValueError(f"refusing to overwrite paid-run freeze manifest: {path}")
    freeze_hash = hashlib.sha256(json.dumps(freeze, sort_keys=True).encode("utf-8")).hexdigest()
    manifest = {
        "freeze": freeze,
        "freeze_sha256": freeze_hash,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "effective_input_guard_bytes_by_model": _input_guard_limits(freeze),
        "input_guard_updates": [],
        "preflight": preflight,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        target.flush()
        os.fsync(target.fileno())
    return freeze_hash


def _price_for_config(model_config):
    rates = model_config["pricing"]["per_million_tokens"]
    return ModelPrice(
        key=model_config["key"], provider=model_config["budget_provider"],
        input_per_million=float(rates["input"]), output_per_million=float(rates["output"]),
    )


def preflight_selected_models(model_configs, prompt_counts, ledger: CostLedger):
    """Estimate full coverage against each model's independent spend cap."""
    by_model, committed_by_model, model_caps, model_fits, assumptions = {}, {}, {}, {}, {}
    groups = {}
    for key, config in model_configs.items():
        budget = config.get("budget")
        if not budget:
            continue
        output_limit = output_token_limit(config["request_params"])
        if output_limit != 4096:
            raise ValueError(f"{key} must keep the frozen 4096-token output limit")
        count = prompt_counts.get(key, 0)
        plan = preflight_cost(
            [_price_for_config(config)], prompts=count, attempts_per_prompt=3,
            max_input_bytes=budget["max_input_bytes"],
            input_token_overhead=budget.get("input_token_overhead", DEFAULT_INPUT_TOKEN_OVERHEAD),
            max_output_tokens=output_limit,
        )
        provider = config["budget_provider"]
        model_cost = plan[f"{provider}_usd"]
        by_model[key] = model_cost
        cap = float(budget.get("cap_usd", MODEL_SPEND_CAP_USD))
        budget_key = config.get("budget_group", key)
        existing_group = groups.setdefault(budget_key, {
            "provider": provider, "cap_usd": cap, "model_keys": [], "future_usd": 0.0,
        })
        if existing_group["provider"] != provider or existing_group["cap_usd"] != cap:
            raise ValueError(f"models sharing budget group {budget_key} must use one provider and cap")
        existing_group["model_keys"].append(key)
        existing_group["future_usd"] += model_cost
        committed = ledger.committed_usd(budget_key, model_key=key)
        model_caps[key] = cap
        committed_by_model[key] = committed
        model_fits[key] = committed + model_cost <= cap + 1e-12
        group_committed = ledger.committed_usd(budget_key)
        call_reserve = model_cost / max(plan["calls_per_model"], 1)
        affordable_calls = int(max(0.0, cap - group_committed) / call_reserve) if call_reserve else 0
        affordable_prompts = min(count, affordable_calls // 3)
        assumptions[key] = {
            "remaining_prompts": count,
            "api_calls": plan["calls_per_model"],
            "max_input_bytes_per_call": plan["max_input_bytes_per_call"],
            "input_token_upper_bound_per_call": plan["max_input_tokens_per_call"],
            "max_output_tokens_per_call": output_limit,
            "cap_usd": cap,
            "budget_group": budget_key,
            "committed_usd": round(committed, 8),
            "group_committed_usd": round(group_committed, 8),
            "conservative_calls_remaining": affordable_calls,
            "conservative_complete_prompts_remaining": affordable_prompts,
            "cap_limited_full_prompts": affordable_prompts,
        }
    standalone_model_fits = dict(model_fits)
    model_budget_groups = {key: config.get("budget_group", key)
                           for key, config in model_configs.items() if config.get("budget")}
    for budget_key, group in groups.items():
        committed = ledger.committed_usd(budget_key)
        future = group["future_usd"]
        cap = group["cap_usd"]
        group.update({
            "committed_usd": committed,
            "committed_by_scope_usd": ledger.committed_by_scope_usd(budget_key),
            "projected_usd": committed + future,
            "remaining_usd": max(0.0, cap - committed - future),
            "fits": committed + future <= cap + 1e-12,
        })
    model_fits = {key: bool(groups[group_key]["fits"])
                  for key, group_key in model_budget_groups.items()}
    providers = {"anthropic", "cometapi", "openai"}
    provider_groups = {provider: [key for key, group in groups.items()
                                  if group["provider"] == provider]
                       for provider in providers}
    fits = {provider: all(groups[key]["fits"] for key in keys)
            for provider, keys in provider_groups.items()}
    committed_totals = {provider: round(sum(groups[key]["committed_usd"]
                                             for key in provider_groups[provider]), 8)
                        for provider in providers}
    future_totals = {provider: round(sum(groups[key]["future_usd"]
                                          for key in provider_groups[provider]), 8)
                     for provider in providers}
    return {
        "model_usd": {key: round(value, 8) for key, value in by_model.items()},
        "anthropic_future_usd": future_totals["anthropic"],
        "cometapi_future_usd": future_totals["cometapi"],
        "openai_future_usd": future_totals["openai"],
        "model_committed_usd": {key: round(value, 8) for key, value in committed_by_model.items()},
        "model_cap_usd": model_caps,
        "model_fits_full_suite": model_fits,
        "model_fits_standalone": standalone_model_fits,
        "anthropic_committed_usd": committed_totals["anthropic"],
        "cometapi_committed_usd": committed_totals["cometapi"],
        "openai_committed_usd": committed_totals["openai"],
        "anthropic_fits": fits["anthropic"],
        "cometapi_fits": fits["cometapi"],
        "openai_fits": fits["openai"],
        "budget_groups": groups,
        "assumptions": assumptions,
        "cost_source": "conservative published rate-card preflight; one UTF-8 byte counts as one input token; cache credits ignored",
    }


def usage_totals(model_calls):
    total = 0
    seen = False
    for call in model_calls:
        usage = call.get("telemetry", {}).get("token_usage", {})
        value = usage.get("total_tokens") if isinstance(usage, dict) else None
        if value is not None:
            total += int(value)
            seen = True
    return total if seen else None


def bound_parse_errors(result):
    """Keep direct-run diagnostics identical to the environment's bounded parse report."""
    for attempt in result.get("attempts", []):
        for check in attempt.get("report", {}).get("checks", []):
            if check.get("id") == "candidate_parse" and isinstance(check.get("message"), str):
                check["message"] = check["message"][:1000]


def initial_request_for_prompt(prompt):
    """Rebuild the exact first provider request used for this task's task-start budget check."""
    from pramaan.dsl import Constraints
    from pramaan.prompts import SYSTEM_PROMPT, design_prompt, design_prompt_for_task, system_prompt_for_task

    if prompt.get("grammar") and prompt.get("suite_id"):
        from pramaan.tasks import Task
        task = Task.model_validate(prompt)
        return system_prompt_for_task(task), design_prompt_for_task(task)
    palette = prompt.get("palette", PALETTE)
    profile = Constraints(symmetry=prompt.get("symmetry", "D4"), max_colours=4, max_primitives=100,
                          min_gap=0, min_coverage=0.02, max_coverage=0.70,
                          exclusive_layer_pairs=[])
    profile_json = {**profile.model_dump(mode="json"), "exclusive_ink_layers": False}
    return SYSTEM_PROMPT, design_prompt(prompt["brief"], palette, profile_json)


def run_prompt(prompt, client=None, suite_sha256=None, model_config=None, via_env=False):
    """Use one initial proposal per model for both paired outcomes."""
    task = None
    if prompt.get("grammar") and prompt.get("suite_id"):
        from pramaan.tasks import Task
        task = Task.model_validate(prompt)
    elif prompt.get("grammar") and prompt["grammar"] != "periodic_raster":
        raise ValueError(f"grammar {prompt['grammar']!r} requires a complete task-suite record")
    if via_env and task is None:
        raise ValueError("via-env runs require a task suite record with grammar and suite_id")
    started = time.perf_counter()
    palette = prompt.get("palette", PALETTE)
    if task is not None:
        result = generate_and_verify(max_attempts=3, client=client, task=task)
    else:
        result = generate_and_verify(prompt["brief"], palette, symmetry=prompt["symmetry"],
                                     max_colours=4, max_attempts=3, client=client)
    bound_parse_errors(result)
    if via_env:
        from pramaan_env import PramaanAction
        from pramaan_env.server.environment import PramaanEnvironment

        env = PramaanEnvironment()
        env.reset(task_id=prompt["id"], suite=prompt["suite_id"])
        provider_truncated = any(
            (attempt.get("metadata") or {}).get("finish_reason") == "length"
            for attempt in result["attempts"]
        )
        for attempt in result["attempts"]:
            normalized_raw, _normalizations = normalize_model_text(attempt["raw"] or "")
            observation = env.step(PramaanAction(pattern_json=normalized_raw))
            # The environment action carries only candidate JSON. Provider-only
            # truncation metadata is recorded by the direct runner, not replayed
            # as a verifier check through OpenEnv.
            env_checks = [check for check in observation.checks if check.get("id") != "candidate_truncated"]
            direct_checks = [check for check in attempt["report"]["checks"] if check.get("id") != "candidate_truncated"]
            if env_checks != direct_checks:
                direct_checks = attempt["report"]["checks"]
                def summarize_check(check):
                    summary = {key: check.get(key) for key in ("id", "status", "witness", "counterexample")
                               if key in check}
                    if isinstance(check.get("message"), str):
                        summary["message"] = check["message"][:240]
                    return summary
                direct_summary = [summarize_check(check) for check in direct_checks]
                env_summary = [summarize_check(check) for check in observation.checks]
                finish_reason = (attempt.get("metadata") or {}).get("finish_reason")
                raise RuntimeError(
                    f"direct/environment verifier mismatch for {prompt['id']} attempt {attempt['index']}: "
                    f"finish_reason={finish_reason!r}; direct={direct_summary}; env={env_summary}"
                )
            if observation.done:
                break
        env_accepted = bool(result["attempts"] and result["attempts"][-1]["report"]["all_hard_constraints_pass"])
        if env.state.accepted != env_accepted and not provider_truncated:
            raise RuntimeError(f"direct/environment acceptance mismatch for {prompt['id']}")
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    first_pass = bool(result["attempts"] and result["attempts"][0]["report"]["all_hard_constraints_pass"])
    first_schema_valid = bool(result["attempts"] and result["attempts"][0]["candidate"] is not None)
    accepted_mask = result.get("certificate", {}).get("mask_sha256")
    model = model_config or {}
    model_meta = result.get("model") or {}
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "validator_version": VALIDATOR_VERSION,
        "model_key": model.get("key") or model_meta.get("model_id", "unspecified"),
        "model_id": model.get("model_id") or model_meta.get("model_id"),
        "provider": model.get("provider") or model_meta.get("provider"),
        "api_style": model.get("api_style", "injected-client"),
        "request_params": model.get("request_params") or (result["attempts"][0].get("metadata", {}).get("request_params") if result["attempts"] else None),
        "pricing": model.get("pricing"),
        "resolved_model_ids": sorted({
            call.get("metadata", {}).get("resolved_model_id")
            for call in result.get("model_calls", [])
            if isinstance(call.get("metadata", {}).get("resolved_model_id"), str)
        }),
        "prompt_suite_sha256": suite_sha256,
        "prompt_id": prompt["id"], "level": prompt["level"],
        "state": prompt.get("state"), "grammar": prompt.get("grammar"),
        "suite_id": prompt.get("suite_id"), "tradition_id": prompt.get("tradition_id"),
        "symmetry": prompt["symmetry"], "brief": prompt["brief"],
        "valid_schema_at_1": first_schema_valid,
        "validity_at_1": first_pass, "validity_at_3": bool(result["accepted"]),
        "repair_rescued": (not first_pass) and bool(result["accepted"]),
        "accepted_mask_sha256": accepted_mask,
        "max_attempts": 3, "attempt_count": len(result["attempts"]), "elapsed_ms": elapsed_ms,
        "total_tokens": usage_totals(result["model_calls"]),
        "token_usage": result["telemetry"]["token_usage"],
        "cost": result["telemetry"]["cost"],
        "latency_ms": result["telemetry"]["latency_ms"],
        "provider_attempts": result["telemetry"]["provider_attempts"],
        "provider_retries": result["telemetry"]["provider_retries"],
        "verification_elapsed_ms": result["telemetry"]["latency_ms"]["verification_total"],
        "provider_error": result.get("provider_error"),
        "attempts": result["attempts"], "model_calls": result["model_calls"],
        "trajectory_schema_version": result["trajectory_schema_version"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


_COMMUNITY_PROVIDER_ENDPOINTS = {
    "Anthropic": ("api.anthropic.com", ""),
    "CometAPI": ("api.cometapi.com", "/v1"),
}


def _community_endpoint_is_allowlisted(config: dict) -> bool:
    """Restrict visitor-key requests to the canonical public provider endpoints."""
    provider = config.get("provider")
    target = _COMMUNITY_PROVIDER_ENDPOINTS.get(provider)
    if target is None:
        return False
    host, path = target
    base_url = config.get("base_url")
    if not isinstance(base_url, str):
        return False
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except ValueError:
        return False
    expected = f"https://{host}{path}"
    return (
        parsed.scheme == "https"
        and parsed.hostname == host
        and port is None
        and parsed.username is None
        and parsed.password is None
        and parsed.query == ""
        and parsed.fragment == ""
        and parsed.path.rstrip("/") == path
        and base_url.rstrip("/") == expected
    )



class LiteLLMCommunityClient:
    """Single-attempt LiteLLM adapter using a visitor key only for this request."""

    def __init__(self, *, token, model_id, provider, request_params):
        if not token or not isinstance(model_id, str) or not isinstance(provider, str):
            raise ValueError("LiteLLM client configuration is invalid")
        self.token, self.model_id, self.provider = token, model_id, provider
        self.request_params = dict(request_params)

    def complete(self, system, prompt):
        from litellm import completion
        started = time.perf_counter()
        try:
            response = completion(
                model=f"{self.provider}/{self.model_id}",
                api_key=self.token,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                stream=False, num_retries=0, max_retries=0, **self.request_params,
            )
        except Exception as exc:
            exc.provider_attempts = 1
            exc.provider_elapsed_ms = round((time.perf_counter() - started) * 1000)
            raise
        elapsed = round((time.perf_counter() - started) * 1000)
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("LiteLLM returned no completion choices")
        choice = choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", None) or ""
        usage_obj = getattr(response, "usage", None)
        if hasattr(usage_obj, "model_dump"):
            usage = usage_obj.model_dump()
        elif isinstance(usage_obj, dict):
            usage = dict(usage_obj)
        elif hasattr(usage_obj, "__dict__"):
            usage = dict(vars(usage_obj))
        else:
            usage = {}
        metadata = {
            "model_id": self.model_id,
            "resolved_model_id": getattr(response, "model", None) or self.model_id,
            "provider": self.provider,
            "api_style": "litellm-chat-completions",
            "request_params": self.request_params,
            "usage": usage,
            "finish_reason": getattr(choice, "finish_reason", None),
            "provider_attempts": 1,
            "provider_elapsed_ms": elapsed,
        }
        return content, metadata


_LITELLM_PROVIDER_LABELS = {
    "anthropic": "Anthropic", "openai": "OpenAI", "deepseek": "DeepSeek",
    "gemini": "Google Gemini", "mistral": "Mistral", "cohere": "Cohere",
    "groq": "Groq", "xai": "xAI", "together_ai": "Together AI",
    "fireworks_ai": "Fireworks AI", "perplexity": "Perplexity",
}
_LITELLM_MODEL_RE = re.compile(r"(?P<provider>[a-z][a-z0-9_-]{0,31})/(?P<model>[A-Za-z0-9_.:-]+(?:/[A-Za-z0-9_.:-]+)*)")


def community_litellm_model_config(model_name: str) -> dict:
    """Resolve a provider/model ID through LiteLLM's known cost map before any call."""
    if (not isinstance(model_name, str) or len(model_name) > 256
            or not (match := _LITELLM_MODEL_RE.fullmatch(model_name))):
        raise ValueError("enter a LiteLLM model as provider/model")
    provider, model_id = match.group("provider"), match.group("model")
    if provider not in _LITELLM_PROVIDER_LABELS or any(part in {".", ".."} for part in model_id.split("/")):
        raise ValueError("enter a supported LiteLLM provider/model")
    try:
        from litellm import get_model_info
        info = get_model_info(model_id, custom_llm_provider=provider)
    except Exception:
        info = None
    if not isinstance(info, dict):
        raise ValueError("model is not in LiteLLM's priced model catalog")
    input_rate = info.get("input_cost_per_token")
    output_rate = info.get("output_cost_per_token")
    reasoning_rate = info.get("output_cost_per_reasoning_token")
    if isinstance(reasoning_rate, (int, float)) and not isinstance(reasoning_rate, bool) and reasoning_rate > 0:
        output_rate = max(output_rate or 0, reasoning_rate)
    max_output = info.get("max_output_tokens")
    if isinstance(max_output, int) and max_output < 4096:
        raise ValueError("model does not support the fixed 4,096-token output limit")
    if any(not isinstance(rate, (int, float)) or isinstance(rate, bool)
           or not math.isfinite(rate) or rate <= 0 for rate in (input_rate, output_rate)):
        raise ValueError("model needs known non-zero token pricing before evaluation")
    return {
        "key": model_name, "provider": _LITELLM_PROVIDER_LABELS[provider],
        "budget_provider": provider, "budget_group": model_name,
        "api_style": "litellm-chat-completions", "model_id": model_id,
        "request_params": {"temperature": 0.4, "max_tokens": 4096},
        "budget": {"cap_usd": MODEL_SPEND_CAP_USD, "max_input_bytes": 16000,
                   "input_token_overhead": 256},
        "pricing": {"currency": "USD", "per_million_tokens": {
            "input": float(input_rate) * 1_000_000,
            "output": float(output_rate) * 1_000_000,
        }, "source": "LiteLLM model cost map", "source_url": "https://docs.litellm.ai/docs/proxy/model_cost_map",
            "observed_at": datetime.now(timezone.utc).date().isoformat()},
    }


def _make_litellm_community_client(model_config, *, api_key, ledger):
    token = api_key
    if not isinstance(token, str) or not token.strip() or len(token) > 4096:
        raise ValueError("provider API key is invalid")
    client = SecretRedactingClient(LiteLLMCommunityClient(
        token=token, model_id=model_config["model_id"],
        provider=model_config["budget_provider"], request_params=model_config["request_params"]), token)
    rates = model_config["pricing"]["per_million_tokens"]
    return GuardedCompletionClient(
        client, ledger=ledger, model_key=model_config["key"],
        provider=model_config["budget_provider"],
        model_price=ModelPrice(key=model_config["key"], provider=model_config["budget_provider"],
                               input_per_million=float(rates["input"]),
                               output_per_million=float(rates["output"])),
        max_input_bytes=model_config["budget"]["max_input_bytes"],
        input_token_overhead=model_config["budget"]["input_token_overhead"],
        max_output_tokens=model_config["request_params"]["max_tokens"],
        budget_key=model_config["budget_group"],
    )

def estimate_community_evaluation_cost(model_key: str, task_count: int = 72, *,
                                        trajectory_path: Path | None = None) -> dict:
    """Estimate visitor cost from the complete, canonical eval-v2.1 usage reference."""
    if isinstance(task_count, bool) or not isinstance(task_count, int) or not 1 <= task_count <= 72:
        raise ValueError("task_count must be between 1 and 72")
    config = community_litellm_model_config(model_key)
    from pramaan.tasks import load_suite as load_task_suite
    canonical_tasks, suite_sha256 = load_task_suite("eval-v2.1")
    expected_ids = {task.id for task in canonical_tasks}
    if len(canonical_tasks) != 72 or len(expected_ids) != 72:
        raise ValueError("canonical eval-v2.1 suite must contain 72 unique task IDs")

    path = trajectory_path or ROOT / "benchmarks/data/eval-v2.1-trajectories.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(canonical_tasks):
        raise ValueError(f"official eval-v2.1 usage reference must contain exactly {len(canonical_tasks)} task rows")

    observed_ids = set()
    input_tokens = output_tokens = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("official eval-v2.1 usage row must be an object")
        prompt_id = row.get("prompt_id")
        if not isinstance(prompt_id, str) or prompt_id not in expected_ids:
            raise ValueError("official usage task IDs do not match canonical eval-v2.1 suite")
        if prompt_id in observed_ids:
            raise ValueError(f"duplicate task ID in official eval-v2.1 usage reference: {prompt_id}")
        observed_ids.add(prompt_id)
        if row.get("suite_id") != "eval-v2.1":
            raise ValueError(f"official usage row has a non-canonical suite ID: {prompt_id}")
        if row.get("prompt_suite_sha256") != suite_sha256:
            raise ValueError(f"official usage row has a stale suite digest: {prompt_id}")

        calls = row.get("calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError(f"official usage row has no measured token usage: {prompt_id}")
        task_input = task_output = 0
        for call in calls:
            telemetry = call.get("telemetry") if isinstance(call, dict) else None
            usage = telemetry.get("token_usage") if isinstance(telemetry, dict) else None
            if not isinstance(usage, dict):
                raise ValueError(f"official call has no measured token usage: {prompt_id}")
            input_value = usage.get("prompt_tokens", usage.get("input_tokens"))
            output_value = usage.get("completion_tokens", usage.get("output_tokens"))
            if (isinstance(input_value, bool) or not isinstance(input_value, int) or input_value <= 0
                    or isinstance(output_value, bool) or not isinstance(output_value, int) or output_value < 0):
                raise ValueError(f"official call has invalid measured token usage: {prompt_id}")
            task_input += input_value
            task_output += output_value
        if task_input <= 0:
            raise ValueError(f"official usage row has no measured token usage: {prompt_id}")
        input_tokens += task_input
        output_tokens += task_output

    if observed_ids != expected_ids:
        raise ValueError("official usage task IDs do not cover canonical eval-v2.1 suite")
    means = {"input": input_tokens / len(canonical_tasks),
             "output": output_tokens / len(canonical_tasks)}
    rates = config["pricing"]["per_million_tokens"]
    estimated = task_count * (means["input"] * float(rates["input"])
                              + means["output"] * float(rates["output"])) / 1_000_000
    return {"task_count": task_count, "reference_suite": "eval-v2.1",
            "reference_tasks": len(canonical_tasks), "reference_suite_sha256": suite_sha256,
            "mean_input_tokens_per_task": means["input"],
            "mean_output_tokens_per_task": means["output"],
            "estimated_usd": round(estimated, 6),
            "per_run_cap_usd": MODEL_SPEND_CAP_USD,
            "estimated_cost_exceeds_per_run_cap": estimated > MODEL_SPEND_CAP_USD,
            "source": "official_beam_eval_v2.1_measured_usage"}

def run_community_evaluation(model_key: str, api_key: str, *, task_limit: int = 12,
                             full_run_cost_acknowledged: bool = False, progress=None) -> tuple[list[dict], str]:
    """Run one allowlisted visitor-key evaluation in memory under a per-run USD 10 cap.

    Results are returned to the caller for explicit consent-gated publication. The
    append-only spend ledger is temporary and contains no provider credential.
    """
    from pramaan.tasks import load_suite as load_task_suite
    from scripts.push_hub_datasets import sanitize_community_eval_rows

    if not isinstance(api_key, str) or not api_key.strip() or len(api_key) > 4096:
        raise ValueError("provider API key is invalid")
    if isinstance(task_limit, bool) or not isinstance(task_limit, int) or not 1 <= task_limit <= 72:
        raise ValueError("community task_limit must be between 1 and 72")
    if task_limit > 12 and not full_run_cost_acknowledged:
        raise ValueError("runs above 12 tasks require explicit full-run cost acknowledgment")
    config = community_litellm_model_config(model_key)

    tasks, suite_sha256 = load_task_suite("eval-v2.1")
    prompts = [task.model_dump(mode="json") for task in tasks[:task_limit]]
    rows = []
    with tempfile.TemporaryDirectory(prefix="pramaan-community-budget-") as ledger_dir:
        ledger = CostLedger(Path(ledger_dir) / "spend.jsonl",
                            caps={model_key: MODEL_SPEND_CAP_USD})
        client = _make_litellm_community_client(config, api_key=api_key, ledger=ledger)
        for prompt in prompts:
            system, first_prompt = initial_request_for_prompt(prompt)
            if not client.can_reserve_complete_question(system, first_prompt):
                break
            try:
                row = run_prompt(prompt, client=client, suite_sha256=suite_sha256,
                                 model_config=config)
            except BudgetExceeded:
                break
            rows.append(row)
            if progress is not None:
                progress(len(rows), len(prompts))
    if not rows:
        return [], suite_sha256
    safe_rows = sanitize_community_eval_rows(
        rows, run_id=str(uuid.uuid4()), expected_suite_sha256=suite_sha256,
        provider_api_key=api_key,
    )
    return safe_rows, suite_sha256


def load_resume_pairs(path, model_configs, prompt_ids, suite_sha256):
    """Validate saved rows; provider-error rows are not completed evaluations."""
    expected = {(key, prompt_id): config for key, config in model_configs.items() for prompt_id in prompt_ids}
    completed = set()
    seen_success = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            pair = (row.get("model_key"), row.get("prompt_id"))
            config = expected.get(pair)
            if config is None:
                raise ValueError(f"resume row {line_number} is outside the selected model/prompt set")
            expected_fields = {
                "benchmark_version": BENCHMARK_VERSION,
                "validator_version": VALIDATOR_VERSION,
                "prompt_suite_sha256": suite_sha256,
                "model_id": config["model_id"],
                "provider": config["provider"],
                "api_style": config["api_style"],
                "request_params": config["request_params"],
            }
            if any(row.get(key) != value for key, value in expected_fields.items()):
                raise ValueError(f"resume row {line_number} has a mismatched benchmark or model configuration")
            if "pricing" in row and row.get("pricing") != config.get("pricing"):
                raise ValueError(f"resume row {line_number} has a mismatched pricing rate card")
            if not row.get("provider_error"):
                if pair in seen_success:
                    raise ValueError(f"resume file has duplicate successful model/prompt row: {pair[0]}/{pair[1]}")
                seen_success.add(pair)
                completed.add(pair)
    except json.JSONDecodeError as exc:
        raise ValueError(f"resume file contains invalid JSON: {exc}") from exc
    return completed


def release_fast_failed_call_reservations(results_path, ledger, model_configs):
    """Release only unmatched reservations proven by a sub-50 ms, no-usage failure row."""
    rows = [json.loads(line) for line in Path(results_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    events = ledger._events()
    terminals = {event.get("reservation_id") for event in events
                 if event.get("event") in {"settle", "release"}}
    released_rows = {
        (event.get("model_key"), (event.get("evidence") or {}).get("prompt_id"),
         (event.get("evidence") or {}).get("result_created_at"))
        for event in events if event.get("event") == "release"
    }
    released_count = 0
    for model_key, config in model_configs.items():
        budget_key = config.get("budget_group", model_key)
        failed_rows = []
        for row in rows:
            if row.get("model_key") != model_key or not row.get("provider_error"):
                continue
            if (model_key, row.get("prompt_id"), row.get("created_at")) in released_rows:
                continue
            row_errors = [call for call in row.get("model_calls", []) if call.get("error")]
            already_released = (len(row_errors) == 1
                                and (row_errors[0]["error"].get("budget_guard") or {}).get("released") is True)
            if not already_released:
                failed_rows.append(row)
        pending = [event for event in events if event.get("event") == "reserve"
                   and event.get("reservation_id") not in terminals
                   and event.get("provider") == budget_key
                   and event.get("model_key", budget_key) == model_key]
        if len(pending) < len(failed_rows):
            raise ValueError(f"cannot safely match {len(pending)} pending reservations to "
                             f"{len(failed_rows)} provider-error rows for {model_key}")
        # An interrupted run can reserve a request and die before flushing its result row.
        # Keep such extra reservations pending; only pair the ordered prefix with known rows.
        for row, reservation in zip(failed_rows, pending):
            failed_calls = [call for call in row.get("model_calls", []) if call.get("error")]
            if len(failed_calls) != 1:
                raise ValueError(f"provider-error row {model_key}/{row.get('prompt_id')} "
                                 "does not identify exactly one failed call")
            call = failed_calls[0]
            telemetry = call.get("telemetry") or {}
            elapsed = row.get("elapsed_ms")
            usage = telemetry.get("token_usage") or {}
            usage_reported = any(isinstance(usage.get(field), int) and not isinstance(usage.get(field), bool)
                                 for field in ("prompt_tokens", "completion_tokens", "total_tokens"))
            if not isinstance(elapsed, int) or isinstance(elapsed, bool) or elapsed >= 50 or usage_reported:
                continue
            error = row.get("provider_error") or {}
            evidence = {
                "result_file": Path(results_path).name,
                "prompt_id": row.get("prompt_id"),
                "result_created_at": row.get("created_at"),
                "phase": error.get("phase"),
                "elapsed_ms": elapsed,
                "usage_reported": False,
                "provider_error": True,
                "provider_error_type": error.get("type"),
                "status_code": error.get("status_code"),
            }
            ledger.release(
                reservation.get("reservation_id"), provider=budget_key, model_key=model_key,
                reason="provider_call_failed_before_send", evidence=evidence)
            released_count += 1
    return released_count


def validate_paid_eval_suite(prompts: list[dict]) -> str | None:
    """Recognize only the frozen common47 set or its exact ordered delta25."""
    suite_dir = ROOT / "benchmarks" / "suites"
    try:
        full = [json.loads(line) for line in (suite_dir / "eval-v2.1.jsonl").read_text(encoding="utf-8").splitlines() if line]
        common = [json.loads(line) for line in (suite_dir / "eval-v2.1-common47.jsonl").read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError):
        return None
    if len(full) != 72 or len({row.get("id") for row in full}) != 72:
        return None
    common_ids = {row.get("id") for row in common}
    delta = [row for row in full if row.get("id") not in common_ids]
    if prompts == full:
        return "eval-v2.1"
    if prompts == common and len(common) == 47:
        return "eval-v2.1-common47"
    if prompts == delta and len(delta) == 25:
        return "eval-v2.1-delta25"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--suite", type=Path, help="Task suite JSONL to run instead of the legacy prompt file")
    parser.add_argument("--via-env", action="store_true", help="Replay generated candidates through PramaanEnvironment")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "results" / "paired.jsonl")
    parser.add_argument("--models", type=Path, default=ROOT / "benchmarks" / "models.json")
    parser.add_argument("--model", action="append", dest="model_keys",
                        help="Configured model key (repeat to evaluate several models); defaults to beam-hf")
    parser.add_argument("--ledger", type=Path,
                        default=ROOT / "benchmarks" / "results" / "provider-spend-ledger.jsonl",
                        help="Append-only provider spend ledger (never include credentials)")
    parser.add_argument("--cap-usd", type=float,
                        help="Optional lower per-model cap; may not exceed the configured cap")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Print full-suite worst-case spend bounds without making provider requests")
    parser.add_argument("--managed-paid-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--start", type=int, default=0, help="Zero-based prompt offset for a benchmark segment")
    parser.add_argument("--limit", type=int, help="Optional number of prompts to run from --start")
    parser.add_argument("--resume", action="store_true",
                        help="Append only missing pairs after validating the existing result file")
    args = parser.parse_args()
    if args.prompts and args.suite:
        parser.error("choose either --prompts or --suite")
    try:
        registry = load_model_registry(args.models, include_paid=True)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))
    selected = args.model_keys or ["beam-hf"]
    if not selected or any(key not in registry for key in selected):
        parser.error("select at least one configured model key")
    if len(selected) != len(set(selected)):
        parser.error("model keys must be unique per run")
    prompt_path = args.suite or args.prompts or (ROOT / "benchmarks" / "prompts.jsonl")
    prompt_bytes = prompt_path.read_bytes()
    suite_sha256 = hashlib.sha256(canonical_jsonl_bytes(prompt_bytes)).hexdigest()
    prompts = [json.loads(line) for line in prompt_bytes.decode("utf-8").splitlines() if line.strip()]
    if args.via_env and not args.suite:
        parser.error("--via-env requires --suite so task identity and verifier profile are explicit")
    all_prompt_ids = [prompt.get("id") for prompt in prompts]
    if not prompts or any(not prompt_id for prompt_id in all_prompt_ids):
        parser.error("prompt file must contain records with non-empty ids")
    if len(all_prompt_ids) != len(set(all_prompt_ids)):
        parser.error("prompt file contains duplicate ids")
    if args.start < 0:
        parser.error("--start cannot be negative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    paid_models = {key: registry[key] for key in selected if registry[key].get("budget_provider")}
    if paid_models and not args.managed_paid_run:
        parser.error("budgeted models must be launched through scripts/run_arms.py so prior-scope and per-model caps are enforced")
    if args.cap_usd is not None:
        if not paid_models or not math.isfinite(args.cap_usd) or args.cap_usd <= 0:
            parser.error("--cap-usd requires paid models and a positive finite amount")
        for key, config in paid_models.items():
            configured_cap = float(config["budget"].get("cap_usd", MODEL_SPEND_CAP_USD))
            if args.cap_usd > configured_cap:
                parser.error(f"--cap-usd cannot exceed {key}'s configured per-model cap of ${configured_cap:.2f}")
            config["budget"] = {**config["budget"], "cap_usd": args.cap_usd}
    if paid_models and (not args.suite or args.prompts):
        parser.error("guarded model runs require the frozen task suite via --suite")
    if paid_models:
        artifact = validate_paid_eval_suite(prompts)
        first_five_probe = (artifact == "eval-v2.1" and args.start == 0 and args.limit == 5
                            and args.managed_paid_run and not args.resume)
        if (any(prompt.get("suite_id") != "eval-v2.1" for prompt in prompts) or artifact is None
                or args.start != 0 or (args.limit is not None and not first_five_probe)):
            parser.error("guarded evaluations require one canonical eval-v2.1 suite; only the managed first-five probe may be segmented")
    if args.preflight_only and not paid_models:
        parser.error("--preflight-only requires one or more budgeted paid models")
    if paid_models and any(output_token_limit(registry[key]["request_params"]) != 4096 for key in selected):
        parser.error("all paid evaluation models must keep the frozen 4096-token output limit")
    budget_caps = {config.get("budget_group", key): config["budget"].get("cap_usd", MODEL_SPEND_CAP_USD)
                   for key, config in paid_models.items()}
    if args.preflight_only:
        if args.resume:
            parser.error("--preflight-only cannot be combined with --resume")
        ledger = CostLedger(args.ledger, caps=budget_caps)
        plan = preflight_selected_models(paid_models, {key: len(prompts) for key in paid_models}, ledger)
        plan["suite_path"] = str(prompt_path)
        plan["suite_sha256"] = suite_sha256
        plan["selected_model_ids"] = {key: registry[key]["model_id"] for key in paid_models}
        plan["output_limits"] = {key: output_token_limit(registry[key]["request_params"]) for key in paid_models}
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    prompts = prompts[args.start:args.start + args.limit if args.limit is not None else None]
    prompt_ids = [prompt.get("id") for prompt in prompts]
    if not prompts:
        parser.error("no prompts selected; check --start and --limit")
    if args.resume and not args.output.exists():
        parser.error("--resume requires an existing results file")
    if args.output.exists() and not args.resume:
        parser.error(f"refusing to overwrite existing results file: {args.output}; choose a new --output path or use --resume")
    try:
        completed = (load_resume_pairs(args.output, registry, all_prompt_ids, suite_sha256)
                     if args.resume else set())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    ledger = CostLedger(args.ledger, caps=budget_caps)
    if args.resume and paid_models:
        try:
            released = release_fast_failed_call_reservations(args.output, ledger, paid_models)
            if released:
                print(f"Released {released} reservation(s) for failed calls proven to stop before provider send.")
            completed = load_resume_pairs(args.output, registry, all_prompt_ids, suite_sha256)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
    preflight = None
    if paid_models:
        prompt_counts = {
            key: sum((key, prompt_id) not in completed for prompt_id in all_prompt_ids)
            for key in paid_models
        }
        preflight = preflight_selected_models(paid_models, prompt_counts, ledger)
        print(json.dumps(preflight, indent=2, sort_keys=True))
        blocked_groups = [key for key, group in preflight.get("budget_groups", {}).items()
                          if not group.get("fits")]
        # Managed paid arms run through the per-call cap guard and stop at task boundaries.
        if blocked_groups and not args.managed_paid_run:
            parser.error("conservative full-suite reservations exceed budget group(s): " + ", ".join(blocked_groups))
    missing_tokens = [registry[key]["api_key_env"] for key in selected
                      if not configured_token(registry[key]["api_key_env"])]
    if missing_tokens:
        parser.error("missing credentials for selected model(s): " + ", ".join(sorted(set(missing_tokens))))
    freeze_hash = None
    if paid_models:
        freeze_manifest_path = Path(str(args.output) + ".preflight.json")
        try:
            freeze_hash = save_or_validate_paid_freeze(
                freeze_manifest_path,
                paid_run_freeze(paid_models, suite_sha256),
                preflight,
                resume=args.resume,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a" if args.resume else "w", encoding="utf-8") as out:
        for model_key in selected:
            model_config = registry[model_key]
            remaining = [prompt for prompt in prompts if (model_key, prompt["id"]) not in completed]
            if not remaining:
                print(f"{model_key}: all selected prompt pairs already recorded")
                continue
            client = make_client(model_config, ledger=ledger)
            for prompt in remaining:
                if isinstance(client, GuardedCompletionClient):
                    system, first_prompt = initial_request_for_prompt(prompt)
                    if not client.can_reserve_complete_question(system, first_prompt):
                        print(f"Stopped before starting {model_key}/{prompt['id']}: less than three actual-request reservations remain.")
                        break
                try:
                    record = run_prompt(prompt, client=client, suite_sha256=suite_sha256,
                                        model_config=model_config, via_env=args.via_env)
                except BudgetExceeded as exc:
                    print(f"Stopped before completing {model_key}/{prompt['id']}: {exc}")
                    print("This model is capped; completed rows are durable. Proceeding to the next model.")
                    break
                if preflight is not None:
                    record["budget_preflight"] = preflight
                    record["evaluation_freeze_sha256"] = freeze_hash
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(f"{model_key}/{prompt['id']}: validity@1={record['validity_at_1']} "
                      f"validity@3={record['validity_at_3']} attempts={record['attempt_count']}/3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
