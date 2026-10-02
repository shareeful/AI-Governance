from __future__ import annotations

from typing import Sequence

import numpy as np

from ...config import Config
from ...execution import Input
from .base import AccessLevel, DetectorProfile, IntegrityDetector, hidden_representations


class ActivationClusteringDetector(IntegrityDetector):
    profile = DetectorProfile(
        name="Activation clustering",
        access=(AccessLevel.WHITE_BOX, AccessLevel.OFFLINE_BATCH),
        runtime="offline",
    )

    def __init__(self, config: Config) -> None:
        self.components = config.int_value("baselines.integrity.activation_clustering.components")
        self.clusters = config.int_value("baselines.integrity.activation_clustering.clusters")
        self.restarts = config.int_value("baselines.integrity.activation_clustering.restarts")
        self.random_state = config.int_value(
            "baselines.integrity.activation_clustering.random_state"
        )
        self._reference: float | None = None

    def fit(self, reference_system, requests: Sequence[Input], rng: np.random.Generator) -> None:
        self._reference = self._silhouette_gap(reference_system, requests)

    def score(self, system, requests: Sequence[Input], rng: np.random.Generator) -> np.ndarray:
        gap = self._silhouette_gap(system, requests)
        baseline = self._reference if self._reference is not None else 0.0
        return np.full(len(requests), gap - baseline, dtype=float)

    def _silhouette_gap(self, system, requests: Sequence[Input]) -> float:
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        activations = hidden_representations(system, requests)
        components = min(self.components, activations.shape[1], activations.shape[0])
        reduced = PCA(n_components=components).fit_transform(activations)
        model = KMeans(
            n_clusters=self.clusters, n_init=self.restarts, random_state=self.random_state
        ).fit(reduced)
        sizes = np.bincount(model.labels_, minlength=self.clusters).astype(float)
        sizes = sizes / sizes.sum()
        return float(np.max(sizes) - np.min(sizes))
