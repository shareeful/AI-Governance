from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm

from ..errors import GovernanceError


@dataclass(frozen=True)
class Interval:
    point: float
    lower: float
    upper: float
    confidence: float
    resamples: int
    method: str

    def to_dict(self) -> dict[str, float | str | int]:
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "resamples": self.resamples,
            "method": self.method,
        }

    def format(self, digits: int = 3) -> str:
        return (
            f"{self.point:.{digits}f} "
            f"[{self.lower:.{digits}f}, {self.upper:.{digits}f}]"
        )


def bca_interval(
    sample: Sequence[float],
    statistic: Callable[[np.ndarray], float],
    resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> Interval:
    values = np.asarray(sample, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise GovernanceError("the bootstrap requires a non-empty one-dimensional sample")
    if not 0.0 < confidence < 1.0:
        raise GovernanceError("the bootstrap confidence level must lie within (0, 1)")
    if resamples < 2:
        raise GovernanceError("the bootstrap requires at least two resamples")

    observed = float(statistic(values))
    n = values.size
    indices = rng.integers(0, n, size=(resamples, n))
    replicates = np.array([statistic(values[row]) for row in indices], dtype=float)

    proportion_below = float(np.mean(replicates < observed))
    if proportion_below <= 0.0 or proportion_below >= 1.0:
        bias_correction = 0.0
    else:
        bias_correction = float(norm.ppf(proportion_below))

    jackknife = np.array(
        [statistic(np.delete(values, index)) for index in range(n)], dtype=float
    )
    jackknife_mean = float(np.mean(jackknife))
    deviations = jackknife_mean - jackknife
    denominator = 6.0 * (np.sum(deviations**2) ** 1.5)
    acceleration = (
        float(np.sum(deviations**3) / denominator) if denominator != 0.0 else 0.0
    )

    alpha = 1.0 - confidence
    lower_z = norm.ppf(alpha / 2.0)
    upper_z = norm.ppf(1.0 - alpha / 2.0)

    def _adjust(z: float) -> float:
        numerator = bias_correction + z
        denom = 1.0 - acceleration * numerator
        if denom == 0.0:
            return float(norm.cdf(bias_correction + numerator))
        return float(norm.cdf(bias_correction + numerator / denom))

    lower_percentile = 100.0 * _adjust(lower_z)
    upper_percentile = 100.0 * _adjust(upper_z)
    lower_percentile = float(np.clip(lower_percentile, 0.0, 100.0))
    upper_percentile = float(np.clip(upper_percentile, 0.0, 100.0))
    if lower_percentile > upper_percentile:
        lower_percentile, upper_percentile = upper_percentile, lower_percentile

    return Interval(
        point=observed,
        lower=float(np.percentile(replicates, lower_percentile)),
        upper=float(np.percentile(replicates, upper_percentile)),
        confidence=confidence,
        resamples=resamples,
        method="bca",
    )


def mean_interval(
    sample: Sequence[float],
    resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> Interval:
    return bca_interval(sample, lambda a: float(np.mean(a)), resamples, confidence, rng)


def paired_difference_interval(
    left: Sequence[float],
    right: Sequence[float],
    resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> Interval:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise GovernanceError("paired samples must have identical shape")
    return mean_interval(left_array - right_array, resamples, confidence, rng)
