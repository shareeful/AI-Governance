from __future__ import annotations

import numpy as np

from ..calibration.ltt import CalibrationResult, CalibrationTargets, candidate_thresholds, learn_then_test
from ..config import Config
from ..errors import CalibrationError
from .backends import EncoderSpec, SequenceClassifier


class EntailmentChecker:
    def __init__(self, config: Config) -> None:
        self.spec = EncoderSpec.from_config(config, "checkers.entailment")
        self.classifier = SequenceClassifier(self.spec)
        self.entailment_index = self.classifier.index_for_label(
            config.str_value("checkers.entailment.entailment_label")
        )
        self._threshold: float | None = None

    @property
    def threshold(self) -> float:
        if self._threshold is None:
            raise CalibrationError(
                "the entailment checker has no decision point; calibrate it before enforcement"
            )
        return self._threshold

    def entailment_probability(self, premise: str, hypothesis: str) -> float:
        probabilities = self.classifier.probabilities([premise], [hypothesis])
        return float(probabilities[0, self.entailment_index])

    def entailment_probabilities(self, premises: list[str], hypotheses: list[str]) -> np.ndarray:
        if not premises:
            return np.zeros(0, dtype=float)
        return self.classifier.probabilities(premises, hypotheses)[:, self.entailment_index]

    def calibrate(
        self,
        premises: list[str],
        hypotheses: list[str],
        is_unsupported: list[bool],
        config: Config,
    ) -> CalibrationResult:
        probabilities = self.entailment_probabilities(premises, hypotheses)
        targets = CalibrationTargets.from_config(config, "checkers.entailment.calibration")
        thresholds = candidate_thresholds(config, "checkers.entailment.calibration")
        result = learn_then_test(
            measures=probabilities.tolist(),
            violates=is_unsupported,
            thresholds=thresholds.tolist(),
            targets=targets,
        )
        self._threshold = result.threshold
        return result

    def entails(self, premise: str, hypothesis: str) -> bool:
        return self.entailment_probability(premise, hypothesis) >= self.threshold
