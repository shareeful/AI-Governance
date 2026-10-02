from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

from ..errors import GovernanceError


@dataclass(frozen=True)
class PairedComparison:
    label: str
    n: int
    statistic: float
    p_value: float
    adjusted_p_value: float
    mean_difference: float
    significant: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "n": self.n,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "mean_difference": self.mean_difference,
            "significant": self.significant,
        }


def wilcoxon_signed_rank(left: Sequence[float], right: Sequence[float]) -> tuple[float, float]:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise GovernanceError("the paired test requires samples of identical shape")
    if left_array.size < 1:
        raise GovernanceError("the paired test requires at least one pair")
    differences = left_array - right_array
    if np.allclose(differences, 0.0):
        return 0.0, 1.0
    result = wilcoxon(left_array, right_array, alternative="two-sided", zero_method="wilcox")
    return float(result.statistic), float(result.pvalue)


def holm_bonferroni(p_values: Mapping[str, float], family_wise_alpha: float) -> dict[str, float]:
    if not p_values:
        raise GovernanceError("the Holm-Bonferroni correction requires at least one p-value")
    if not 0.0 < family_wise_alpha < 1.0:
        raise GovernanceError("the family-wise level must lie within (0, 1)")
    items = sorted(p_values.items(), key=lambda pair: pair[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (label, value) in enumerate(items):
        scaled = min(1.0, (m - index) * value)
        running = max(running, scaled)
        adjusted[label] = running
    return adjusted


def compare_paired(
    samples: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    family_wise_alpha: float,
) -> dict[str, PairedComparison]:
    raw: dict[str, tuple[int, float, float, float]] = {}
    for label, (left, right) in samples.items():
        statistic, p_value = wilcoxon_signed_rank(left, right)
        difference = float(np.mean(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))
        raw[label] = (len(list(left)), statistic, p_value, difference)
    adjusted = holm_bonferroni({k: v[2] for k, v in raw.items()}, family_wise_alpha)
    return {
        label: PairedComparison(
            label=label,
            n=values[0],
            statistic=values[1],
            p_value=values[2],
            adjusted_p_value=adjusted[label],
            mean_difference=values[3],
            significant=adjusted[label] <= family_wise_alpha,
        )
        for label, values in raw.items()
    }
