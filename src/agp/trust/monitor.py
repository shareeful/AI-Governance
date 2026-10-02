from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from scipy.stats import ks_2samp

from ..config import Config
from ..errors import ConfigurationError
from ..execution import Decision
from .ledger import LedgerRecord


@dataclass(frozen=True)
class DriftSignal:
    reject_rate: float
    not_attested_rate: float
    input_shift_statistic: float
    reject_rate_exceeded: bool
    not_attested_rate_exceeded: bool
    input_shift_exceeded: bool

    @property
    def triggered(self) -> bool:
        return (
            self.reject_rate_exceeded
            or self.not_attested_rate_exceeded
            or self.input_shift_exceeded
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "reject_rate": self.reject_rate,
            "not_attested_rate": self.not_attested_rate,
            "input_shift_statistic": self.input_shift_statistic,
            "reject_rate_exceeded": self.reject_rate_exceeded,
            "not_attested_rate_exceeded": self.not_attested_rate_exceeded,
            "input_shift_exceeded": self.input_shift_exceeded,
            "triggered": self.triggered,
        }


class PolicyMonitor:
    def __init__(self, config: Config, calibration_embedding: np.ndarray) -> None:
        self.window = config.int_value("trust.monitor.window")
        if self.window < 2:
            raise ConfigurationError("trust.monitor.window must span at least two executions")
        self.reject_rate_bound = config.require_unit_interval("trust.monitor.reject_rate_bound")
        self.not_attested_rate_bound = config.require_unit_interval(
            "trust.monitor.not_attested_rate_bound"
        )
        self.input_shift_bound = config.require_unit_interval("trust.monitor.input_shift_bound")
        calibration_embedding = np.asarray(calibration_embedding, dtype=float)
        if calibration_embedding.ndim != 1 or calibration_embedding.size == 0:
            raise ConfigurationError(
                "the drift monitor requires a non-empty one-dimensional embedding of the "
                "calibration split"
            )
        self.calibration_embedding = calibration_embedding
        self._decisions: deque[Decision] = deque(maxlen=self.window)
        self._attested: deque[bool] = deque(maxlen=self.window)
        self._embeddings: deque[float] = deque(maxlen=self.window)

    def observe(self, decision: Decision, attested: bool, embedding: float) -> DriftSignal:
        self._decisions.append(decision)
        self._attested.append(attested)
        self._embeddings.append(float(embedding))
        return self.signal()

    def signal(self) -> DriftSignal:
        if not self._decisions:
            raise ConfigurationError("the monitor has observed no execution")
        reject_rate = float(
            np.mean([d is Decision.WITHHOLD for d in self._decisions])
        )
        not_attested_rate = float(np.mean([not a for a in self._attested]))
        if len(self._embeddings) >= 2:
            shift = float(
                ks_2samp(
                    np.array(self._embeddings, dtype=float),
                    self.calibration_embedding,
                    alternative="two-sided",
                    method="asymp",
                ).statistic
            )
        else:
            shift = 0.0
        return DriftSignal(
            reject_rate=reject_rate,
            not_attested_rate=not_attested_rate,
            input_shift_statistic=shift,
            reject_rate_exceeded=reject_rate > self.reject_rate_bound,
            not_attested_rate_exceeded=not_attested_rate > self.not_attested_rate_bound,
            input_shift_exceeded=shift > self.input_shift_bound,
        )

    def reset(self) -> None:
        self._decisions.clear()
        self._attested.clear()
        self._embeddings.clear()


def replay(records: Iterable[LedgerRecord]) -> list[Mapping[str, object]]:
    return [record.to_dict() for record in records]


def clause_failure_rates(records: Iterable[LedgerRecord]) -> dict[str, float]:
    counts: dict[str, int] = {}
    totals: dict[str, int] = {}
    for record in records:
        for entry in record.body.get("guards", []):
            for clause in entry.get("clause", []):
                totals[clause] = totals.get(clause, 0) + 1
                if entry.get("verdict") == "reject":
                    counts[clause] = counts.get(clause, 0) + 1
    return {
        clause: counts.get(clause, 0) / total
        for clause, total in totals.items()
        if total > 0
    }
