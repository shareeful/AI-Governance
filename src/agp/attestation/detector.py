from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import ks_2samp

from ..calibration.conformal import conformal_p_value, holm_adjust
from ..calibration.ltt import (
    CalibrationResult,
    CalibrationTargets,
    candidate_thresholds,
    learn_then_test_upper,
)
from ..config import Config
from ..errors import AttestationError, CalibrationError
from .fingerprint import BehaviouralFingerprint
from .signals import SCALAR_SIGNALS, VECTOR_SIGNAL, SignalVector

_LOG_FLOOR = 1e-300


def jensen_shannon_divergence(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise AttestationError("Jensen-Shannon divergence requires vectors of identical length")
    left_sum = left.sum()
    right_sum = right.sum()
    if left_sum <= 0.0 or right_sum <= 0.0:
        raise AttestationError("Jensen-Shannon divergence requires positive attribution mass")
    p = left / left_sum
    q = right / right_sum
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


@dataclass(frozen=True)
class AttestationVerdict:
    attested: bool
    statistic: float
    threshold: float
    p_values: Mapping[str, float]
    adjusted_p_values: Mapping[str, float]
    window_size: int


@dataclass(frozen=True)
class WindowStatistic:
    value: float
    p_values: Mapping[str, float]
    adjusted: Mapping[str, float]


class AttestationDetector:
    def __init__(
        self,
        fingerprint: BehaviouralFingerprint,
        window: int,
        enabled_signals: Sequence[str],
    ) -> None:
        if window < 2:
            raise AttestationError("the attestation window must span at least two executions")
        known = set(SCALAR_SIGNALS) | {VECTOR_SIGNAL}
        unknown = set(enabled_signals) - known
        if unknown:
            raise AttestationError(
                "unknown attestation signals: " + ", ".join(sorted(unknown))
            )
        if not enabled_signals:
            raise AttestationError("at least one attestation signal must be enabled")
        self.fingerprint = fingerprint
        self.window = window
        self.enabled_signals = tuple(enabled_signals)
        self._buffer: deque[SignalVector] = deque(maxlen=window)
        self._null: Mapping[str, np.ndarray] | None = None
        self._threshold: float | None = None

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise CalibrationError(
                "the attestation detector is not calibrated; run Task 2.2 before enforcement"
            )
        return self._threshold

    def is_calibrated(self) -> bool:
        return self._threshold is not None and self._null is not None

    def raw_statistics(self, vectors: Sequence[SignalVector]) -> dict[str, float]:
        statistics: dict[str, float] = {}
        for name in self.enabled_signals:
            if name == VECTOR_SIGNAL:
                observed = np.vstack([v.feature_reliance for v in vectors]).mean(axis=0)
                statistics[name] = jensen_shannon_divergence(
                    observed, self.fingerprint.reference_feature_distribution
                )
            else:
                observed = np.array([v.scalar(name) for v in vectors], dtype=float)
                statistics[name] = float(
                    ks_2samp(
                        observed,
                        self.fingerprint.scalar_samples[name],
                        alternative="two-sided",
                        method="asymp",
                    ).statistic
                )
        return statistics

    def build_null_distribution(
        self,
        certification_vectors: Sequence[SignalVector],
        draws: int,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        if len(certification_vectors) < self.window:
            raise AttestationError(
                f"the certification workload holds {len(certification_vectors)} executions but "
                f"the detector window spans {self.window}; enlarge the workload or shorten the "
                "window in configuration"
            )
        samples: dict[str, list[float]] = {name: [] for name in self.enabled_signals}
        indices = np.arange(len(certification_vectors))
        for _ in range(draws):
            selected = rng.choice(indices, size=self.window, replace=False)
            window_vectors = [certification_vectors[int(i)] for i in selected]
            for name, value in self.raw_statistics(window_vectors).items():
                samples[name].append(value)
        null = {name: np.array(values, dtype=float) for name, values in samples.items()}
        self._null = null
        return null

    def window_statistic(self, vectors: Sequence[SignalVector]) -> WindowStatistic:
        if self._null is None:
            raise CalibrationError(
                "the conformal null distribution is absent; build it from the certification "
                "workload before computing the detector statistic"
            )
        raw = self.raw_statistics(vectors)
        names = tuple(raw)
        p_values = np.array(
            [conformal_p_value(raw[name], self._null[name]) for name in names], dtype=float
        )
        adjusted = holm_adjust(p_values)
        minimum = float(np.min(adjusted))
        statistic = -math.log(max(minimum, _LOG_FLOOR))
        return WindowStatistic(
            value=statistic,
            p_values={name: float(p) for name, p in zip(names, p_values)},
            adjusted={name: float(p) for name, p in zip(names, adjusted)},
        )

    def calibrate(
        self,
        certification_vectors: Sequence[SignalVector],
        config: Config,
        rng: np.random.Generator,
    ) -> CalibrationResult:
        draws = config.int_value("attestation.null_draws")
        calibration_windows = config.int_value("attestation.calibration_windows")
        self.build_null_distribution(certification_vectors, draws, rng)

        statistics: list[float] = []
        indices = np.arange(len(certification_vectors))
        for _ in range(calibration_windows):
            selected = rng.choice(indices, size=self.window, replace=False)
            window_vectors = [certification_vectors[int(i)] for i in selected]
            statistics.append(self.window_statistic(window_vectors).value)

        targets = CalibrationTargets.from_config(config, "attestation.calibration")
        thresholds = candidate_thresholds(config, "attestation.calibration")
        result = learn_then_test_upper(
            statistics=statistics,
            thresholds=thresholds.tolist(),
            targets=targets,
        )
        self._threshold = result.threshold
        return result

    def observe(self, vector: SignalVector) -> AttestationVerdict:
        self._buffer.append(vector)
        if len(self._buffer) < self.window:
            return AttestationVerdict(
                attested=True,
                statistic=0.0,
                threshold=self.threshold,
                p_values={},
                adjusted_p_values={},
                window_size=len(self._buffer),
            )
        statistic = self.window_statistic(list(self._buffer))
        return AttestationVerdict(
            attested=statistic.value <= self.threshold,
            statistic=statistic.value,
            threshold=self.threshold,
            p_values=statistic.p_values,
            adjusted_p_values=statistic.adjusted,
            window_size=len(self._buffer),
        )

    def score_stream(self, vectors: Sequence[SignalVector]) -> np.ndarray:
        if len(vectors) < self.window:
            raise AttestationError(
                f"the stream holds {len(vectors)} executions but the detector window spans "
                f"{self.window}; a sliding-window statistic is undefined before the window fills"
            )
        self.reset()
        scores: list[float] = []
        for index, vector in enumerate(vectors):
            verdict = self.observe(vector)
            if index >= self.window - 1:
                scores.append(verdict.statistic)
        self.reset()
        return np.array(scores, dtype=float)

    def reset(self) -> None:
        self._buffer.clear()

    def without_signal(self, signal: str) -> "AttestationDetector":
        remaining = tuple(s for s in self.enabled_signals if s != signal)
        if not remaining:
            raise AttestationError(
                f"removing signal '{signal}' would leave the detector with no signals"
            )
        return AttestationDetector(
            fingerprint=self.fingerprint,
            window=self.window,
            enabled_signals=remaining,
        )
