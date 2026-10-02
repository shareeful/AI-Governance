from __future__ import annotations

from typing import Sequence

import numpy as np

from ...config import Config
from ...errors import ConfigurationError
from ...execution import Input
from .base import AccessLevel, DetectorProfile, IntegrityDetector


def rbf_kernel(left: np.ndarray, right: np.ndarray, bandwidth: float) -> np.ndarray:
    distances = (
        np.sum(left**2, axis=1)[:, None]
        + np.sum(right**2, axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.exp(-np.maximum(distances, 0.0) / (2.0 * bandwidth**2))


def maximum_mean_discrepancy(left: np.ndarray, right: np.ndarray, bandwidth: float) -> float:
    kxx = rbf_kernel(left, left, bandwidth)
    kyy = rbf_kernel(right, right, bandwidth)
    kxy = rbf_kernel(left, right, bandwidth)
    n = left.shape[0]
    m = right.shape[0]
    if n < 2 or m < 2:
        raise ConfigurationError("the MMD statistic requires at least two samples per sample set")
    term_xx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    term_yy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    return float(term_xx + term_yy - 2.0 * kxy.mean())


def median_bandwidth(sample: np.ndarray) -> float:
    if sample.shape[0] < 2:
        raise ConfigurationError("the median heuristic requires at least two samples")
    distances = np.sqrt(
        np.maximum(
            np.sum(sample**2, axis=1)[:, None]
            + np.sum(sample**2, axis=1)[None, :]
            - 2.0 * sample @ sample.T,
            0.0,
        )
    )
    upper = distances[np.triu_indices_from(distances, k=1)]
    median = float(np.median(upper))
    return median if median > 0.0 else 1.0


class MmdDriftDetector(IntegrityDetector):
    profile = DetectorProfile(
        name="MMD drift detector", access=(AccessLevel.BLACK_BOX,), runtime="streaming"
    )

    def __init__(self, config: Config) -> None:
        self.window = config.int_value("baselines.integrity.mmd.window")
        self._reference: np.ndarray | None = None
        self._bandwidth: float | None = None

    def fit(self, reference_system, requests: Sequence[Input], rng: np.random.Generator) -> None:
        self._reference = np.vstack(
            [reference_system.decision_scores(request) for request in requests]
        )
        self._bandwidth = median_bandwidth(self._reference)

    def score(self, system, requests: Sequence[Input], rng: np.random.Generator) -> np.ndarray:
        if self._reference is None or self._bandwidth is None:
            raise ConfigurationError("the MMD detector must be fitted on the certified model")
        outputs = np.vstack([system.decision_scores(request) for request in requests])
        scores = np.zeros(len(requests), dtype=float)
        for index in range(len(requests)):
            start = max(0, index - self.window + 1)
            window = outputs[start : index + 1]
            if window.shape[0] < 2:
                scores[index] = 0.0
                continue
            scores[index] = maximum_mean_discrepancy(window, self._reference, self._bandwidth)
        return scores
