from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calibration.ltt import CalibrationResult, CalibrationTargets, candidate_thresholds, learn_then_test
from ..config import Config
from ..errors import CalibrationError
from .backends import EncoderSpec, SequenceClassifier


@dataclass(frozen=True)
class InjectionVerdict:
    is_attack: bool
    score: float
    sanitised: str | None


class InjectionJailbreakChecker:
    def __init__(self, config: Config) -> None:
        self.spec = EncoderSpec.from_config(config, "checkers.injection")
        self.classifier = SequenceClassifier(self.spec)
        self.attack_index = self.classifier.index_for_label(
            config.str_value("checkers.injection.attack_label")
        )
        self.sanitise = config.bool_value("checkers.injection.sanitise")
        self._threshold: float | None = None

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise CalibrationError(
                "the injection checker has no decision point; calibrate it on the held-out "
                "attack set before enforcement"
            )
        return self._threshold

    def score(self, text: str) -> float:
        probabilities = self.classifier.probabilities([text])
        return float(probabilities[0, self.attack_index])

    def score_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=float)
        return self.classifier.probabilities(texts)[:, self.attack_index]

    def calibrate(
        self,
        texts: list[str],
        is_attack: list[bool],
        config: Config,
    ) -> CalibrationResult:
        scores = self.score_many(texts)
        compliance = 1.0 - scores
        targets = CalibrationTargets.from_config(config, "checkers.injection.calibration")
        thresholds = candidate_thresholds(config, "checkers.injection.calibration")
        result = learn_then_test(
            measures=compliance.tolist(),
            violates=is_attack,
            thresholds=thresholds.tolist(),
            targets=targets,
        )
        self._threshold = 1.0 - result.threshold
        return result

    def check(self, text: str) -> InjectionVerdict:
        score = self.score(text)
        if score <= self.threshold:
            return InjectionVerdict(is_attack=False, score=score, sanitised=text)
        if not self.sanitise:
            return InjectionVerdict(is_attack=True, score=score, sanitised=None)
        stripped = self._strip_above_decision_point(text)
        if stripped is None:
            return InjectionVerdict(is_attack=True, score=score, sanitised=None)
        return InjectionVerdict(is_attack=False, score=self.score(stripped), sanitised=stripped)

    def _strip_above_decision_point(self, text: str) -> str | None:
        segments = [segment for segment in text.splitlines() if segment.strip()]
        if len(segments) <= 1:
            return None
        segment_scores = self.score_many(segments)
        retained = [
            segment
            for segment, score in zip(segments, segment_scores)
            if score <= self.threshold
        ]
        if not retained:
            return None
        candidate = "\n".join(retained)
        if self.score(candidate) > self.threshold:
            return None
        return candidate
