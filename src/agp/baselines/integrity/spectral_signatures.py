from __future__ import annotations

from typing import Sequence

import numpy as np

from ...config import Config
from ...execution import Input
from .base import AccessLevel, DetectorProfile, IntegrityDetector, hidden_representations


class SpectralSignatureDetector(IntegrityDetector):
    profile = DetectorProfile(
        name="Spectral signatures",
        access=(AccessLevel.WHITE_BOX, AccessLevel.OFFLINE_BATCH),
        runtime="offline",
    )

    def __init__(self, config: Config) -> None:
        self.upper_quantile = config.require_unit_interval(
            "baselines.integrity.spectral_signatures.outlier_quantile"
        )
        self._reference: float | None = None

    def fit(self, reference_system, requests: Sequence[Input], rng: np.random.Generator) -> None:
        self._reference = float(np.mean(self._outlier_scores(reference_system, requests)))

    def score(self, system, requests: Sequence[Input], rng: np.random.Generator) -> np.ndarray:
        scores = self._outlier_scores(system, requests)
        baseline = self._reference if self._reference is not None else 0.0
        return scores - baseline

    def _outlier_scores(self, system, requests: Sequence[Input]) -> np.ndarray:
        activations = hidden_representations(system, requests)
        centred = activations - activations.mean(axis=0, keepdims=True)
        _, _, right = np.linalg.svd(centred, full_matrices=False)
        projection = centred @ right[0]
        return np.square(projection)
