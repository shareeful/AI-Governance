from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from ..errors import ConfigurationError


class ClauseId(str, Enum):
    NON_DISCRIMINATION = "non_discrimination"
    DATA_MINIMISATION = "data_minimisation"
    REASONING_FAITHFULNESS = "reasoning_faithfulness"
    PRECONDITION_COMPLIANCE = "precondition_compliance"
    FACTUAL_GROUNDEDNESS = "factual_groundedness"


class DecisionKind(str, Enum):
    EXACT = "exact"
    LEARNED = "learned"


class ExecutionComponent(str, Enum):
    INPUT = "x"
    TRACE = "r"
    ACTION = "a"
    RESPONSE = "y"


class FailureAction(str, Enum):
    WITHHOLD = "withhold"
    REFER = "refer"
    LOG = "log"


EXACT_CLAUSES: frozenset[ClauseId] = frozenset(
    {
        ClauseId.NON_DISCRIMINATION,
        ClauseId.DATA_MINIMISATION,
        ClauseId.PRECONDITION_COMPLIANCE,
    }
)

LEARNED_CLAUSES: frozenset[ClauseId] = frozenset(
    {
        ClauseId.REASONING_FAITHFULNESS,
        ClauseId.FACTUAL_GROUNDEDNESS,
    }
)


@dataclass(frozen=True)
class ClauseRecord:
    clause: ClauseId
    regulatory_basis: str
    measure: str
    decision_kind: DecisionKind
    threshold: float | None
    checker: str
    read_access: tuple[ExecutionComponent, ...]
    failure_action: tuple[FailureAction, ...]
    requires_reexecution: bool
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.decision_kind is DecisionKind.EXACT and self.threshold is None:
            raise ConfigurationError(
                f"clause '{self.clause.value}' is exactly decided and requires a fixed threshold"
            )
        if self.decision_kind is DecisionKind.EXACT and self.threshold != 1.0:
            raise ConfigurationError(
                f"clause '{self.clause.value}' is exactly decided; its threshold must be 1.0 "
                f"per Table 1 of the specification, got {self.threshold}"
            )
        if not self.failure_action:
            raise ConfigurationError(
                f"clause '{self.clause.value}' must declare at least one failure action"
            )
        if not self.read_access:
            raise ConfigurationError(
                f"clause '{self.clause.value}' must declare the execution components it reads"
            )

    @property
    def is_calibrated(self) -> bool:
        return self.decision_kind is DecisionKind.LEARNED

    def with_threshold(self, threshold: float) -> "ClauseRecord":
        if self.decision_kind is DecisionKind.EXACT:
            raise ConfigurationError(
                f"clause '{self.clause.value}' is exactly decided and cannot be recalibrated"
            )
        return ClauseRecord(
            clause=self.clause,
            regulatory_basis=self.regulatory_basis,
            measure=self.measure,
            decision_kind=self.decision_kind,
            threshold=threshold,
            checker=self.checker,
            read_access=self.read_access,
            failure_action=self.failure_action,
            requires_reexecution=self.requires_reexecution,
            parameters=self.parameters,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause": self.clause.value,
            "regulatory_basis": self.regulatory_basis,
            "measure": self.measure,
            "decision_kind": self.decision_kind.value,
            "threshold": self.threshold,
            "checker": self.checker,
            "read_access": [c.value for c in self.read_access],
            "failure_action": [f.value for f in self.failure_action],
            "requires_reexecution": self.requires_reexecution,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True)
class ClausePolicy:
    records: Mapping[ClauseId, ClauseRecord]

    def __post_init__(self) -> None:
        missing = set(ClauseId) - set(self.records)
        if missing:
            raise ConfigurationError(
                "the policy model requires all five clauses; missing: "
                + ", ".join(sorted(c.value for c in missing))
            )

    def __getitem__(self, clause: ClauseId) -> ClauseRecord:
        return self.records[clause]

    def calibrated(self) -> tuple[ClauseRecord, ...]:
        return tuple(r for r in self.records.values() if r.is_calibrated)

    def exact(self) -> tuple[ClauseRecord, ...]:
        return tuple(r for r in self.records.values() if not r.is_calibrated)

    def with_thresholds(self, thresholds: Mapping[ClauseId, float]) -> "ClausePolicy":
        updated = dict(self.records)
        for clause, threshold in thresholds.items():
            updated[clause] = updated[clause].with_threshold(threshold)
        return ClausePolicy(records=updated)

    def to_dict(self) -> dict[str, Any]:
        return {clause.value: record.to_dict() for clause, record in self.records.items()}
