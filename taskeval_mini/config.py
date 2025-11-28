from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelConfig:
    name: str
    kind: str  # 'rule_based' or 'openai_chat'
    params: Dict[str, Any]


# You can extend this mapping with more models.
AVAILABLE_MODELS: Dict[str, ModelConfig] = {
    "rule_based": ModelConfig(
        name="rule_based",
        kind="rule_based",
        params={},
    ),
    # Example OpenAI chat model configuration.
    # Requires OPENAI_API_KEY in the environment and the openai package installed.
    "openai_gpt4o": ModelConfig(
        name="openai_gpt4o",
        kind="openai_chat",
        params={
            "model": "gpt-4o",
        },
    ),
}
