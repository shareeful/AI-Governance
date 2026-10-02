from __future__ import annotations

import numpy as np
import pytest

from agp.calibration.bounds import (
    bentkus_p_value,
    hoeffding_bentkus_p_value,
    hoeffding_p_value,
)
from agp.calibration.conformal import conformal_p_value, holm_adjust, holm_minimum
from agp.calibration.ltt import CalibrationTargets, learn_then_test, learn_then_test_upper
from agp.errors import CalibrationError, NoAdmissibleThresholdError

TARGETS = CalibrationTargets(
    alpha=0.05, delta=0.10, bound="hoeffding_bentkus", correction="fixed_sequence"
)


def test_p_value_is_one_when_risk_meets_target():
    assert hoeffding_p_value(0.05, 100, 0.05) == 1.0
    assert bentkus_p_value(0.20, 100, 0.05) == 1.0


def test_p_value_decreases_with_sample_size():
    small = hoeffding_bentkus_p_value(0.01, 50, 0.05)
    large = hoeffding_bentkus_p_value(0.01, 5000, 0.05)
    assert large < small


def test_learn_then_test_returns_permissive_admissible_threshold():
    rng = np.random.default_rng(0)
    measures = np.concatenate([rng.uniform(0.0, 0.4, 500), rng.uniform(0.6, 1.0, 500)])
    violates = np.concatenate([np.ones(500, dtype=bool), np.zeros(500, dtype=bool)])
    thresholds = np.linspace(0.0, 1.0, 101)
    result = learn_then_test(measures, violates, thresholds, TARGETS)

    assert result.empirical_risk <= TARGETS.alpha
    assert result.p_value <= TARGETS.delta
    assert result.threshold == result.tested_thresholds[-1]
    assert all(p > TARGETS.delta for p in result.p_value_curve[:-1])


def test_learn_then_test_admits_no_more_permissive_threshold():
    rng = np.random.default_rng(0)
    measures = np.concatenate([rng.uniform(0.0, 0.4, 500), rng.uniform(0.6, 1.0, 500)])
    violates = np.concatenate([np.ones(500, dtype=bool), np.zeros(500, dtype=bool)])
    thresholds = np.linspace(0.0, 1.0, 101)
    result = learn_then_test(measures, violates, thresholds, TARGETS)

    more_permissive = thresholds[thresholds < result.threshold]
    for threshold in more_permissive:
        risk = float(np.mean((measures >= threshold) & violates))
        assert hoeffding_bentkus_p_value(risk, measures.size, TARGETS.alpha) > TARGETS.delta


def test_learn_then_test_rejects_when_no_threshold_controls_risk():
    measures = np.ones(200)
    violates = np.ones(200, dtype=bool)
    with pytest.raises(NoAdmissibleThresholdError):
        learn_then_test(measures, violates, np.linspace(0.0, 1.0, 11), TARGETS)


def test_learn_then_test_requires_a_violating_example():
    with pytest.raises(CalibrationError):
        learn_then_test(np.zeros(10), np.zeros(10, dtype=bool), [0.5], TARGETS)


def test_detector_threshold_bounds_false_alarm_rate():
    rng = np.random.default_rng(1)
    statistics = rng.exponential(scale=1.0, size=2000)
    targets = CalibrationTargets(
        alpha=0.01, delta=0.10, bound="hoeffding_bentkus", correction="fixed_sequence"
    )
    result = learn_then_test_upper(statistics, np.linspace(0.0, 12.0, 241), targets)
    assert float(np.mean(statistics > result.threshold)) <= targets.alpha


def test_conformal_p_value_is_bounded_and_monotone():
    calibration = np.linspace(0.0, 1.0, 100)
    assert conformal_p_value(-1.0, calibration) == pytest.approx(1.0)
    assert conformal_p_value(2.0, calibration) == pytest.approx(1.0 / 101.0)
    assert conformal_p_value(0.2, calibration) > conformal_p_value(0.8, calibration)


def test_holm_adjustment_matches_step_down_definition():
    adjusted = holm_adjust(np.array([0.01, 0.02, 0.03, 0.04]))
    assert adjusted[0] == pytest.approx(0.04)
    assert adjusted[1] == pytest.approx(0.06)
    assert adjusted[3] == pytest.approx(0.06)
    assert np.all(np.diff(np.sort(adjusted)) >= -1e-12)


def test_holm_minimum_scales_the_smallest_p_value():
    assert holm_minimum(np.array([0.001, 0.5, 0.6, 0.7])) == pytest.approx(0.004)
