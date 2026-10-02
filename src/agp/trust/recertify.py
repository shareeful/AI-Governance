from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ..errors import GovernanceError
from .ledger import LedgerRecord
from .monitor import DriftSignal


@dataclass(frozen=True)
class RecertificationCandidate:
    record: LedgerRecord
    admitted: bool
    attested: bool


@dataclass(frozen=True)
class ScreeningOutcome:
    approved: tuple[LedgerRecord, ...]
    rejected: tuple[tuple[LedgerRecord, str], ...]


ValidationHook = Callable[[LedgerRecord], tuple[bool, str]]


def screen_candidates(
    candidates: Sequence[RecertificationCandidate],
    contamination_hook: ValidationHook,
) -> ScreeningOutcome:
    if not candidates:
        raise GovernanceError(
            "re-certification requires candidate records drawn from the signed runtime log"
        )
    approved: list[LedgerRecord] = []
    rejected: list[tuple[LedgerRecord, str]] = []
    for candidate in candidates:
        if not candidate.admitted:
            rejected.append((candidate.record, "withheld_execution"))
            continue
        if not candidate.attested:
            rejected.append((candidate.record, "not_attested_execution"))
            continue
        accepted, reason = contamination_hook(candidate.record)
        if not accepted:
            rejected.append((candidate.record, reason))
            continue
        approved.append(candidate.record)
    if not approved:
        raise GovernanceError(
            "every re-certification candidate was screened out; the certified baselines cannot "
            "be refreshed from contaminated or withheld evidence"
        )
    return ScreeningOutcome(approved=tuple(approved), rejected=tuple(rejected))


@dataclass(frozen=True)
class RecertificationTrigger:
    signal: DriftSignal
    executions_observed: int

    @property
    def required(self) -> bool:
        return self.signal.triggered
