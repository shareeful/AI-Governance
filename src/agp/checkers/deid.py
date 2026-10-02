from __future__ import annotations

import re
from dataclasses import dataclass

from ..config import Config
from ..errors import ConfigurationError
from .backends import EncoderSpec, TokenClassifier


@dataclass(frozen=True)
class DeidentificationOutcome:
    text: str
    removed: frozenset[str]
    interpretable: bool


class DeIdentifier:
    def __init__(self, config: Config) -> None:
        self.spec = EncoderSpec.from_config(config, "checkers.deidentification")
        self.labeller = TokenClassifier(self.spec)
        self.mask_template = config.str_value("checkers.deidentification.mask_template")
        if "{label}" not in self.mask_template:
            raise ConfigurationError(
                "checkers.deidentification.mask_template must contain the '{label}' placeholder"
            )
        self.minimum_token_ratio = config.require_unit_interval(
            "checkers.deidentification.minimum_interpretable_token_ratio"
        )

    def extract(self, text: str) -> frozenset[str]:
        return frozenset(
            f"{label}:{surface.strip()}"
            for label, surface, _, _ in self.labeller.spans(text)
            if surface.strip()
        )

    def categories(self, text: str) -> frozenset[str]:
        return frozenset(label for label, _, _, _ in self.labeller.spans(text))

    def mask(self, text: str, retain: frozenset[str]) -> DeidentificationOutcome:
        spans = self.labeller.spans(text)
        removable = [
            span for span in spans if f"{span[0]}:{span[1].strip()}" not in retain
        ]
        if not removable:
            return DeidentificationOutcome(text=text, removed=frozenset(), interpretable=True)
        masked = text
        removed: set[str] = set()
        for label, surface, start, end in sorted(removable, key=lambda s: s[2], reverse=True):
            masked = masked[:start] + self.mask_template.format(label=label) + masked[end:]
            removed.add(f"{label}:{surface.strip()}")
        original_tokens = len(re.findall(r"\w+", text))
        masked_tokens = len(re.findall(r"\w+", re.sub(r"\[[A-Z_]+\]", " ", masked)))
        ratio = masked_tokens / original_tokens if original_tokens else 0.0
        return DeidentificationOutcome(
            text=masked,
            removed=frozenset(removed),
            interpretable=ratio >= self.minimum_token_ratio,
        )
