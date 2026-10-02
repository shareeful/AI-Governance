from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import Config
from ..errors import ConfigurationError, MissingResourceError


def resolve_device(config: Config) -> str:
    requested = config.str_value("runtime.device")
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(frozen=True)
class EncoderSpec:
    identifier: str
    max_length: int
    batch_size: int
    device: str

    @classmethod
    def from_config(cls, config: Config, prefix: str) -> "EncoderSpec":
        return cls(
            identifier=config.str_value(f"{prefix}.model"),
            max_length=config.int_value(f"{prefix}.max_length"),
            batch_size=config.int_value(f"{prefix}.batch_size"),
            device=resolve_device(config),
        )


@functools.lru_cache(maxsize=None)
def _load_sequence_classifier(identifier: str, device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(identifier)
        model = AutoModelForSequenceClassification.from_pretrained(identifier)
    except (OSError, ValueError) as exc:
        raise MissingResourceError(
            f"sequence-classification checkpoint '{identifier}' could not be resolved. "
            "Configure a local path or an available model identifier; the framework does not "
            "substitute an untrained model."
        ) from exc
    model.eval()
    model.to(device)
    return tokenizer, model


@functools.lru_cache(maxsize=None)
def _load_token_classifier(identifier: str, device: str) -> tuple[Any, Any]:
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(identifier)
        model = AutoModelForTokenClassification.from_pretrained(identifier)
    except (OSError, ValueError) as exc:
        raise MissingResourceError(
            f"token-classification checkpoint '{identifier}' could not be resolved. "
            "Configure a local path or an available model identifier."
        ) from exc
    model.eval()
    model.to(device)
    return tokenizer, model


class SequenceClassifier:
    def __init__(self, spec: EncoderSpec) -> None:
        self.spec = spec
        self.tokenizer, self.model = _load_sequence_classifier(spec.identifier, spec.device)
        self.label_to_index = {
            str(label).lower(): int(index)
            for label, index in self.model.config.label2id.items()
        }

    def index_for_label(self, label: str) -> int:
        key = label.lower()
        if key not in self.label_to_index:
            available = ", ".join(sorted(self.label_to_index))
            raise ConfigurationError(
                f"checkpoint '{self.spec.identifier}' does not expose a label named {label!r}; "
                f"available labels: {available}"
            )
        return self.label_to_index[key]

    def probabilities(self, texts: list[str], pairs: list[str] | None = None) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, self.model.config.num_labels), dtype=float)
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), self.spec.batch_size):
            chunk = texts[start : start + self.spec.batch_size]
            second = pairs[start : start + self.spec.batch_size] if pairs is not None else None
            encoded = self.tokenizer(
                chunk,
                second,
                truncation=True,
                padding=True,
                max_length=self.spec.max_length,
                return_tensors="pt",
            ).to(self.spec.device)
            with torch.no_grad():
                logits = self.model(**encoded).logits
            outputs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(outputs, axis=0)


class TokenClassifier:
    def __init__(self, spec: EncoderSpec) -> None:
        self.spec = spec
        self.tokenizer, self.model = _load_token_classifier(spec.identifier, spec.device)
        self.index_to_label = {int(k): str(v) for k, v in self.model.config.id2label.items()}

    def spans(self, text: str) -> list[tuple[str, str, int, int]]:
        import torch

        if not text.strip():
            return []
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.spec.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {k: v.to(self.spec.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
        predictions = torch.argmax(logits, dim=-1)[0].cpu().tolist()

        spans: list[tuple[str, str, int, int]] = []
        current_label: str | None = None
        current_start = 0
        current_end = 0
        for label_index, (start, end) in zip(predictions, offsets):
            if start == end:
                continue
            raw_label = self.index_to_label[label_index]
            if raw_label in {"O", "o"}:
                if current_label is not None:
                    spans.append((current_label, text[current_start:current_end], current_start, current_end))
                    current_label = None
                continue
            prefix, _, entity = raw_label.partition("-")
            entity = entity or prefix
            if prefix == "B" or current_label != entity:
                if current_label is not None:
                    spans.append((current_label, text[current_start:current_end], current_start, current_end))
                current_label = entity
                current_start = start
            current_end = end
        if current_label is not None:
            spans.append((current_label, text[current_start:current_end], current_start, current_end))
        return spans
