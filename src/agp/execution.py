from __future__ import annotations

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np

from .errors import InterfaceError


class Verdict(str, Enum):
    CERTIFY = "certify"
    REJECT = "reject"


class Decision(str, Enum):
    ADMIT = "admit"
    WITHHOLD = "withhold"


class RiskBand(str, Enum):
    LOW = "low"
    HIGH = "high"


class OversightRegime(str, Enum):
    HITL = "human_in_the_loop"
    HOTL = "human_on_the_loop"


def digest(payload: Any) -> str:
    if isinstance(payload, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(payload).tobytes()).hexdigest()
    if isinstance(payload, bytes):
        return hashlib.sha256(payload).hexdigest()
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EvidenceItem:
    identifier: str
    content: Any
    provenance: str | None
    modality: str

    @property
    def has_provenance(self) -> bool:
        return isinstance(self.provenance, str) and len(self.provenance.strip()) > 0


@dataclass(frozen=True)
class Reason:
    identifier: str
    statement: str
    support: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Claim:
    identifier: str
    statement: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReasoningTrace:
    reasons: tuple[Reason, ...]
    recorded_checks: tuple[str, ...]
    commit_index: int
    text: str

    def without(self, reason_id: str) -> "ReasoningTrace":
        remaining = tuple(r for r in self.reasons if r.identifier != reason_id)
        if len(remaining) == len(self.reasons):
            raise InterfaceError(f"reason '{reason_id}' is not present in the trace")
        return replace(self, reasons=remaining, text="\n".join(r.statement for r in remaining))

    def checks_before_commit(self) -> frozenset[str]:
        return frozenset(self.recorded_checks[: self.commit_index])


@dataclass(frozen=True)
class Response:
    claims: tuple[Claim, ...]
    text: str


@dataclass(frozen=True)
class Input:
    identifier: str
    request: str
    features: Any
    evidence: tuple[EvidenceItem, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    requested_action: str | None = None

    def with_attribute(self, attribute: str, value: Any) -> "Input":
        raise InterfaceError(
            "counterfactual substitution is delegated to the governed system; "
            "implement GovernedSystem.substitute_attribute"
        )


@dataclass(frozen=True)
class Execution:
    identifier: str
    inputs: Input
    trace: ReasoningTrace
    action: str
    response: Response
    protected_attributes: Mapping[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @staticmethod
    def new_identifier() -> str:
        return str(uuid.uuid4())

    def input_digest(self) -> str:
        return digest(
            {
                "identifier": self.inputs.identifier,
                "request": self.inputs.request,
                "features": digest(self.inputs.features),
                "evidence": [e.identifier for e in self.inputs.evidence],
            }
        )

    def trace_digest(self) -> str:
        return digest(
            {
                "reasons": [r.identifier for r in self.trace.reasons],
                "checks": list(self.trace.recorded_checks),
                "commit_index": self.trace.commit_index,
            }
        )

    def response_digest(self) -> str:
        return digest([c.statement for c in self.response.claims])


class GovernedSystem(ABC):
    name: str
    modality: str

    @abstractmethod
    def execute(self, request: Input) -> Execution:
        raise NotImplementedError

    @abstractmethod
    def action_for_input(self, request: Input) -> str:
        raise NotImplementedError

    @abstractmethod
    def action_for_trace(self, request: Input, trace: ReasoningTrace) -> str:
        raise NotImplementedError

    @abstractmethod
    def substitute_attribute(self, request: Input, attribute: str, value: Any) -> Input:
        raise NotImplementedError

    @abstractmethod
    def protected_attribute_domain(self, attribute: str) -> Sequence[Any]:
        raise NotImplementedError

    @abstractmethod
    def attribution(self, request: Input) -> Mapping[str, float]:
        raise NotImplementedError

    @abstractmethod
    def decision_scores(self, request: Input) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def perturbations(self, request: Input, count: int, rng: np.random.Generator) -> list[Input]:
        raise NotImplementedError

    @abstractmethod
    def paraphrases(self, request: Input, count: int, rng: np.random.Generator) -> list[Input]:
        raise NotImplementedError

    @abstractmethod
    def required_checks(self, action: str) -> frozenset[str]:
        raise NotImplementedError

    @abstractmethod
    def minimum_necessary_personal_data(self) -> frozenset[str]:
        raise NotImplementedError
