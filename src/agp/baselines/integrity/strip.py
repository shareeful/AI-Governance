from __future__ import annotations

from typing import Sequence

import numpy as np

from ...attestation.signals import predictive_entropy
from ...config import Config
from ...execution import Input
from .base import AccessLevel, DetectorProfile, IntegrityDetector


class StripDetector(IntegrityDetector):
    profile = DetectorProfile(
        name="STRIP", access=(AccessLevel.BLACK_BOX,), runtime="per-execution"
    )

    def __init__(self, config: Config) -> None:
        self.perturbation_count = config.int_value("baselines.integrity.strip.perturbation_count")
        self.blend_weight = config.require_unit_interval("baselines.integrity.strip.blend_weight")
        self._pool: list[np.ndarray] = []

    def fit(self, reference_system, requests: Sequence[Input], rng: np.random.Generator) -> None:
        self._pool = [np.asarray(request.features, dtype=np.float32) for request in requests]
        if not self._pool:
            raise ValueError("STRIP requires a non-empty superimposition pool")

    def score(self, system, requests: Sequence[Input], rng: np.random.Generator) -> np.ndarray:
        from dataclasses import replace

        scores: list[float] = []
        for request in requests:
            base = np.asarray(request.features, dtype=np.float32)
            entropies: list[float] = []
            indices = rng.integers(0, len(self._pool), size=self.perturbation_count)
            for index in indices:
                donor = self._pool[int(index)]
                if donor.shape != base.shape:
                    continue
                blended = (
                    (1.0 - self.blend_weight) * base + self.blend_weight * donor
                ).astype(np.float32)
                entropies.append(
                    predictive_entropy(system.decision_scores(replace(request, features=blended)))
                )
            if not entropies:
                raise ValueError(
                    "no donor in the STRIP pool matched the input shape; the pool must be drawn "
                    "from the same system"
                )
            scores.append(-float(np.mean(entropies)))
        return np.asarray(scores, dtype=float)
