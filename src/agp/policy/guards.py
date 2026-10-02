from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..execution import Verdict
from .clause import ClauseId, ClauseRecord, DecisionKind


@dataclass(frozen=True)
class GuardEntry:
    guard: str
    clauses: tuple[ClauseId, ...]
    measures: Mapping[str, float]
    thresholds: Mapping[str, float]
    verdict: Verdict
    evidence_digest: str
    detail: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "guard": self.guard,
            "clause": [c.value for c in self.clauses],
            "measure": dict(self.measures),
            "threshold": dict(self.thresholds),
            "verdict": self.verdict.value,
            "evidence": self.evidence_digest,
            "detail": dict(self.detail),
        }


def decide(record: ClauseRecord, measure_value: float) -> Verdict:
    if record.threshold is None:
        raise ValueError(
            f"clause '{record.clause.value}' has no threshold; "
            "learned clauses must be calibrated in Phase 2 before enforcement"
        )
    if record.decision_kind is DecisionKind.EXACT:
        return Verdict.CERTIFY if measure_value >= record.threshold else Verdict.REJECT
    return Verdict.CERTIFY if measure_value >= record.threshold else Verdict.REJECT
