from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from .config import ModelConfig
from .prompts import support_action_prompt


class BaseModelRunner(ABC):
    """Abstract interface for running a model on a single task."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def generate(self, input_text: str, context: str | None = None) -> Dict[str, Any]:
        """Return a JSON-compatible dict as model output for the given task."""
        raise NotImplementedError


class RuleBasedModelRunner(BaseModelRunner):
    """Very simple heuristic baseline.

    This is intentionally naive, but it lets the pipeline run without external APIs.
    """

    def generate(self, input_text: str, context: str | None = None) -> Dict[str, Any]:
        text = input_text.lower()
        intent = "general_question"
        target_system = "general"
        priority = "medium"
        requires_human = True
        sla_hours = 24

        if "refund" in text or "charge" in text or "billing" in text or "invoice" in text:
            intent = "billing_issue"
            target_system = "billing"
        if "password" in text or "login" in text or "account" in text:
            intent = "account_issue"
            target_system = "account"
        if "slow" in text or "disconnect" in text or "latency" in text or "network" in text:
            intent = "technical_issue"
            target_system = "network"
        if "cancel" in text or "close my account" in text:
            intent = "cancellation_request"

        if "urgent" in text or "asap" in text or "immediately" in text or "not working" in text:
            priority = "high"
            sla_hours = 4
        elif "whenever" in text or "no rush" in text:
            priority = "low"
            sla_hours = 72

        if "just wanted to ask" in text or "curious" in text:
            requires_human = False

        return {
            "intent": intent,
            "priority": priority,
            "requires_human": requires_human,
            "target_system": target_system,
            "sla_hours": sla_hours,
        }


class OpenAIChatModelRunner(BaseModelRunner):
    """Calls an OpenAI chat completion model.

    Requires:
    - `openai` Python package installed.
    - `OPENAI_API_KEY` environment variable set.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        try:
            import openai  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "openai package is not installed. Install it to use OpenAIChatModelRunner."
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

        # For the new OpenAI client
        try:
            from openai import OpenAI  # type: ignore
            self.client = OpenAI(api_key=api_key)
            self._use_client = True
        except Exception:
            # Fallback for older openai package style
            import openai as _openai  # type: ignore
            _openai.api_key = api_key
            self._openai_legacy = _openai
            self._use_client = False

    def generate(self, input_text: str, context: str | None = None) -> Dict[str, Any]:
        prompt = support_action_prompt(input_text, context)

        if self._use_client:
            response = self.client.chat.completions.create(
                model=self.config.params["model"],
                messages=[
                    {"role": "system", "content": "You are a strict JSON-only assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
        else:
            response = self._openai_legacy.ChatCompletion.create(
                model=self.config.params["model"],
                messages=[
                    {"role": "system", "content": "You are a strict JSON-only assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            content = response.choices[0].message["content"]

        try:
            return json.loads(content)
        except Exception:
            # If the model returns invalid JSON, wrap it for downstream error handling.
            return {"_raw": content}


def create_model_runner(config: ModelConfig) -> BaseModelRunner:
    if config.kind == "rule_based":
        return RuleBasedModelRunner(config)
    if config.kind == "openai_chat":
        return OpenAIChatModelRunner(config)
    raise ValueError(f"Unknown model kind: {config.kind}")
