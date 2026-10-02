from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..config import Config
from ..errors import CalibrationError, ConfigurationError, NoAdmissibleThresholdError
from .bounds import bentkus_p_value, hoeffding_bentkus_p_value, hoeffding_p_value

_BOUNDS: dict[str, Callable[[float, int, float], float]] = {
    "hoeffding": hoeffding_p_value,
    "bentkus": bentkus_p_value,
    "hoeffding_bentkus": hoeffding_bentkus_p_value,
}


@dataclass(frozen=True)
class CalibrationTargets:
    alpha: float
    delta: float
    bound: str
    correction: str

    @classmethod
    def from_config(cls, config: Config, prefix: str = "calibration") -> "CalibrationTargets":
        alpha = config.require_probability(f"{prefix}.alpha")
        delta = config.require_probability(f"{prefix}.delta")
        bound = config.str_value(f"{prefix}.bound")
        correction = config.str_value(f"{prefix}.correction")
        if bound not in _BOUNDS:
            raise ConfigurationError(
                f"{prefix}.bound must be one of {sorted(_BOUNDS)}; got {bound!r}"
            )
        if correction not in {"fixed_sequence", "bonferroni"}:
            raise ConfigurationError(
                f"{prefix}.correction must be 'fixed_sequence' or 'bonferroni'; got {correction!r}"
            )
        return cls(alpha=alpha, delta=delta, bound=bound, correction=correction)


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float
    empirical_risk: float
    p_value: float
    targets: CalibrationTargets
    calibration_size: int
    tested_thresholds: tuple[float, ...]
    risk_curve: tuple[float, ...]
    p_value_curve: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "empirical_risk": self.empirical_risk,
            "p_value": self.p_value,
            "alpha": self.targets.alpha,
            "delta": self.targets.delta,
            "bound": self.targets.bound,
            "correction": self.targets.correction,
            "calibration_size": self.calibration_size,
            "tested_thresholds": list(self.tested_thresholds),
            "risk_curve": list(self.risk_curve),
            "p_value_curve": list(self.p_value_curve),
        }


def candidate_thresholds(config: Config, prefix: str = "calibration") -> np.ndarray:
    start = config.float_value(f"{prefix}.candidate_thresholds.start")
    stop = config.float_value(f"{prefix}.candidate_thresholds.stop")
    count = config.int_value(f"{prefix}.candidate_thresholds.count")
    if count < 2:
        raise ConfigurationError(f"{prefix}.candidate_thresholds.count must be at least 2")
    if not start < stop:
        raise ConfigurationError(
            f"{prefix}.candidate_thresholds.start must be strictly below stop"
        )
    return np.linspace(start, stop, count)


def learn_then_test(
    measures: Sequence[float],
    violates: Sequence[bool],
    thresholds: Sequence[float],
    targets: CalibrationTargets,
) -> CalibrationResult:
    measure_array = np.asarray(measures, dtype=float)
    violation_array = np.asarray(violates, dtype=bool)
    if measure_array.ndim != 1:
        raise CalibrationError("measures must be one-dimensional")
    if measure_array.shape != violation_array.shape:
        raise CalibrationError("measures and violation labels must have identical shape")
    n = measure_array.size
    if n == 0:
        raise CalibrationError("the calibration set is empty")
    if not violation_array.any():
        raise CalibrationError(
            "the calibration set contains no violating execution; the risk of wrongful "
            "certification cannot be estimated from compliant executions alone"
        )

    ordered = np.sort(np.asarray(thresholds, dtype=float))
    bound_fn = _BOUNDS[targets.bound]
    effective_delta = (
        targets.delta / ordered.size if targets.correction == "bonferroni" else targets.delta
    )

    tested: list[float] = []
    risks: list[float] = []
    p_values: list[float] = []

    for threshold in ordered:
        certified_violation = (measure_array >= threshold) & violation_array
        empirical_risk = float(np.mean(certified_violation))
        p_value = float(bound_fn(empirical_risk, n, targets.alpha))
        tested.append(float(threshold))
        risks.append(empirical_risk)
        p_values.append(p_value)
        if p_value <= effective_delta:
            return CalibrationResult(
                threshold=float(threshold),
                empirical_risk=empirical_risk,
                p_value=p_value,
                targets=targets,
                calibration_size=n,
                tested_thresholds=tuple(tested),
                risk_curve=tuple(risks),
                p_value_curve=tuple(p_values),
            )

    raise NoAdmissibleThresholdError(
        "no candidate threshold controls the risk of wrongful certification at "
        f"alpha={targets.alpha} with confidence {1.0 - targets.delta}; "
        f"the most permissive attainable empirical risk was {min(risks):.6f} over "
        f"{n} calibration executions"
    )


def learn_then_test_upper(
    statistics: Sequence[float],
    thresholds: Sequence[float],
    targets: CalibrationTargets,
) -> CalibrationResult:
    statistic_array = np.asarray(statistics, dtype=float)
    if statistic_array.ndim != 1 or statistic_array.size == 0:
        raise CalibrationError("statistics must be a non-empty one-dimensional array")
    n = statistic_array.size
    bound_fn = _BOUNDS[targets.bound]
    ordered = np.sort(np.asarray(thresholds, dtype=float))[::-1]
    effective_delta = (
        targets.delta / ordered.size if targets.correction == "bonferroni" else targets.delta
    )

    tested: list[float] = []
    risks: list[float] = []
    p_values: list[float] = []

    for threshold in ordered:
        empirical_risk = float(np.mean(statistic_array > threshold))
        p_value = float(bound_fn(empirical_risk, n, targets.alpha))
        tested.append(float(threshold))
        risks.append(empirical_risk)
        p_values.append(p_value)
        if p_value <= effective_delta:
            return CalibrationResult(
                threshold=float(threshold),
                empirical_risk=empirical_risk,
                p_value=p_value,
                targets=targets,
                calibration_size=n,
                tested_thresholds=tuple(tested),
                risk_curve=tuple(risks),
                p_value_curve=tuple(p_values),
            )

    raise NoAdmissibleThresholdError(
        "no detector threshold holds the false-alarm rate on the unmanipulated model at or "
        f"below alpha={targets.alpha} with confidence {1.0 - targets.delta} over {n} "
        "certification windows"
    )
