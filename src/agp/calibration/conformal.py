from __future__ import annotations

import numpy as np

from ..errors import CalibrationError


def conformal_p_value(observed: float, calibration: np.ndarray) -> float:
    calibration = np.asarray(calibration, dtype=float)
    if calibration.ndim != 1 or calibration.size == 0:
        raise CalibrationError("the conformal calibration sample must be non-empty")
    exceedances = int(np.sum(calibration >= observed))
    return (1.0 + exceedances) / (calibration.size + 1.0)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or p_values.size == 0:
        raise CalibrationError("Holm correction requires a non-empty vector of p-values")
    m = p_values.size
    order = np.argsort(p_values, kind="stable")
    ordered = p_values[order]
    scaled = ordered * (m - np.arange(m))
    stepped = np.maximum.accumulate(scaled)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.minimum(stepped, 1.0)
    return adjusted


def holm_minimum(p_values: np.ndarray) -> float:
    return float(np.min(holm_adjust(p_values)))
