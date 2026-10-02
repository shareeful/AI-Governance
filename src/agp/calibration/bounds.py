from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom


def _kl_bernoulli(a: float, b: float) -> float:
    eps = 1e-12
    a = min(max(a, eps), 1.0 - eps)
    b = min(max(b, eps), 1.0 - eps)
    return a * math.log(a / b) + (1.0 - a) * math.log((1.0 - a) / (1.0 - b))


def hoeffding_p_value(empirical_risk: float, n: int, alpha: float) -> float:
    if n <= 0:
        raise ValueError("sample size must be positive")
    if empirical_risk >= alpha:
        return 1.0
    return float(math.exp(-n * _kl_bernoulli(empirical_risk, alpha)))


def bentkus_p_value(empirical_risk: float, n: int, alpha: float) -> float:
    if n <= 0:
        raise ValueError("sample size must be positive")
    if empirical_risk >= alpha:
        return 1.0
    successes = math.ceil(n * empirical_risk)
    return float(min(1.0, math.e * binom.cdf(successes, n, alpha)))


def hoeffding_bentkus_p_value(empirical_risk: float, n: int, alpha: float) -> float:
    return float(
        min(
            1.0,
            hoeffding_p_value(empirical_risk, n, alpha),
            bentkus_p_value(empirical_risk, n, alpha),
        )
    )
