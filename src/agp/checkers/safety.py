from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calibration.ltt import CalibrationResult, CalibrationTargets, candidate_thresholds, learn_then_test
from ..config import Config
from ..errors import CalibrationError
from .backends import EncoderSpec, SequenceClassifier


@dataclass(frozen=True)
class SafetyVerdict:
    unsafe: bool
    score: float


class SafetyChecker:
    def __init__(self, config: Config) -> None:
        self.spec = EncoderSpec.from_config(config, "checkers.safety")
        self.classifier = SequenceClassifier(self.spec)
        self.unsafe_index = self.classifier.index_for_label(
            config.str_value("checkers.safety.unsafe_label")
        )
        self._threshold: float | None = None

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise CalibrationError(
                "the safety checker has no decision point; calibrate it before enforcement"
            )
        return self._threshold

    def score_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=float)
        return self.classifier.probabilities(texts)[:, self.unsafe_index]

    def calibrate(self, texts: list[str], is_unsafe: list[bool], config: Config) -> CalibrationResult:
        scores = self.score_many(texts)
        targets = CalibrationTargets.from_config(config, "checkers.safety.calibration")
        thresholds = candidate_thresholds(config, "checkers.safety.calibration")
        result = learn_then_test(
            measures=(1.0 - scores).tolist(),
            violates=is_unsafe,
            thresholds=thresholds.tolist(),
            targets=targets,
        )
        self._threshold = 1.0 - result.threshold
        return result

    def check(self, text: str) -> SafetyVerdict:
        score = float(self.score_many([text])[0])
        return SafetyVerdict(unsafe=score > self.threshold, score=score)
