from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Mapping

from ..config import Config
from ..errors import ConfigurationError, MissingResourceError
from ..execution import Execution, OversightRegime, RiskBand


class ReviewOutcome(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING = "pending"


@dataclass(frozen=True)
class OversightDecision:
    regime: OversightRegime
    reviewer: str | None
    outcome: ReviewOutcome
    referred: bool

    @property
    def rejected(self) -> bool:
        return self.outcome is ReviewOutcome.REJECTED

    @property
    def held(self) -> bool:
        return self.outcome is ReviewOutcome.PENDING

    def to_dict(self) -> dict[str, object]:
        return {
            "regime": self.regime.value,
            "reviewer": self.reviewer,
            "decision": self.outcome.value,
            "referred": self.referred,
        }


class Reviewer(ABC):
    identifier: str

    @abstractmethod
    def review(
        self, execution: Execution, band: RiskBand, evidence: Mapping[str, object]
    ) -> ReviewOutcome:
        raise NotImplementedError


class QueueingReviewer(Reviewer):
    def __init__(self, identifier: str, queue_path: Path) -> None:
        self.identifier = identifier
        self.queue_path = queue_path

    def review(
        self, execution: Execution, band: RiskBand, evidence: Mapping[str, object]
    ) -> ReviewOutcome:
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        with self.queue_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "exec_id": execution.identifier,
                        "timestamp": execution.timestamp,
                        "band": band.value,
                        "action": execution.action,
                        "certified_trace": [r.statement for r in execution.trace.reasons],
                        "evidence": dict(evidence),
                    },
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )
        return ReviewOutcome.PENDING


class RecordedDecisionReviewer(Reviewer):
    def __init__(self, identifier: str, decision_path: Path, queue_path: Path) -> None:
        self.identifier = identifier
        self.decision_path = decision_path
        self.queue = QueueingReviewer(identifier, queue_path)
        if not decision_path.is_file():
            raise MissingResourceError(
                f"no reviewer decision log at {decision_path}; this reviewer replays decisions "
                "a human has actually recorded and does not approve executions on their behalf"
            )
        self.decisions = self._load(decision_path)

    @staticmethod
    def _load(path: Path) -> dict[str, ReviewOutcome]:
        decisions: dict[str, ReviewOutcome] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if "exec_id" not in row or "decision" not in row:
                    raise ConfigurationError(
                        f"reviewer decision log {path} line {line_number} must carry "
                        "'exec_id' and 'decision'"
                    )
                decisions[str(row["exec_id"])] = ReviewOutcome(str(row["decision"]))
        return decisions

    def review(
        self, execution: Execution, band: RiskBand, evidence: Mapping[str, object]
    ) -> ReviewOutcome:
        if execution.identifier in self.decisions:
            return self.decisions[execution.identifier]
        return self.queue.review(execution, band, evidence)


def build_reviewer(config: Config, queue_path: Path) -> Reviewer:
    identifier = config.str_value("oversight.reviewer_identifier")
    mode = config.str_value("oversight.reviewer_mode")
    if mode == "queue":
        return QueueingReviewer(identifier, queue_path)
    if mode == "recorded_decisions":
        return RecordedDecisionReviewer(
            identifier=identifier,
            decision_path=config.path("oversight.reviewer_decision_log"),
            queue_path=queue_path,
        )
    raise ConfigurationError(
        f"oversight.reviewer_mode must be 'queue' or 'recorded_decisions'; got {mode!r}"
    )


class HumanOversight:
    def __init__(self, config: Config, reviewer: Reviewer) -> None:
        self.reviewer = reviewer
        high = config.str_value("oversight.high_band_regime")
        low = config.str_value("oversight.low_band_regime")
        valid = {r.value for r in OversightRegime}
        for label, value in (("high_band_regime", high), ("low_band_regime", low)):
            if value not in valid:
                raise ConfigurationError(
                    f"oversight.{label} must be one of {sorted(valid)}; got {value!r}"
                )
        self.high_band_regime = OversightRegime(high)
        self.low_band_regime = OversightRegime(low)

    def regime_for(self, band: RiskBand) -> OversightRegime:
        return self.high_band_regime if band is RiskBand.HIGH else self.low_band_regime

    def route(
        self,
        execution: Execution,
        band: RiskBand,
        evidence: Mapping[str, object],
    ) -> OversightDecision:
        regime = self.regime_for(band)
        if regime is OversightRegime.HITL:
            outcome = self.reviewer.review(execution, band, evidence)
            return OversightDecision(
                regime=regime,
                reviewer=self.reviewer.identifier,
                outcome=outcome,
                referred=True,
            )
        return OversightDecision(
            regime=regime,
            reviewer=self.reviewer.identifier,
            outcome=ReviewOutcome.APPROVED,
            referred=False,
        )


def missed_referral_rate(referred: list[bool], true_high_band: list[bool]) -> float:
    if len(referred) != len(true_high_band):
        raise ValueError("referral and true-band vectors must have identical length")
    high = [index for index, flag in enumerate(true_high_band) if flag]
    if not high:
        raise ValueError(
            "the missed-referral rate is undefined when no execution carries a high true risk band"
        )
    missed = sum(1 for index in high if not referred[index])
    return missed / len(high)
