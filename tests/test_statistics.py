from __future__ import annotations

import numpy as np
import pytest

from agp.errors import GovernanceError
from agp.stats.bootstrap import mean_interval, paired_difference_interval
from agp.stats.metrics import (
    certified_violation_rate,
    clause_metrics,
    detection_at_false_alarm,
    detection_auc,
)
from agp.stats.tests import compare_paired, holm_bonferroni, wilcoxon_signed_rank


def test_clause_metrics_recover_a_perfect_guard():
    metrics = clause_metrics([True, True, False, False], [True, True, False, False])
    assert metrics.recall == 1.0
    assert metrics.precision == 1.0
    assert metrics.f1 == 1.0
    assert metrics.false_block_rate == 0.0


def test_false_block_rate_counts_only_compliant_executions():
    metrics = clause_metrics([True, True, True, False], [True, True, False, False])
    assert metrics.recall == 1.0
    assert metrics.false_block_rate == pytest.approx(0.5)


def test_clause_metrics_require_both_classes():
    with pytest.raises(GovernanceError):
        clause_metrics([True, True], [True, True])


def test_detection_auc_is_one_for_separated_scores():
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [False, False, True, True]
    assert detection_auc(scores, labels) == pytest.approx(1.0)


def test_detection_at_budget_respects_the_false_alarm_quantile():
    rng = np.random.default_rng(0)
    clean = rng.normal(0.0, 1.0, 2000)
    compromised = rng.normal(4.0, 1.0, 2000)
    scores = np.concatenate([clean, compromised])
    labels = np.concatenate([np.zeros(2000, dtype=bool), np.ones(2000, dtype=bool)])
    assert detection_at_false_alarm(scores, labels, 0.01) > 0.8


def test_certified_violation_rate_counts_certified_violations_only():
    assert certified_violation_rate(
        [True, True, False, False], [True, False, True, False]
    ) == pytest.approx(0.25)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    sample = rng.normal(0.5, 0.1, 200)
    interval = mean_interval(sample, resamples=500, confidence=0.95, rng=rng)
    assert interval.lower <= interval.point <= interval.upper


def test_paired_difference_interval_excludes_zero_for_a_real_shift():
    rng = np.random.default_rng(2)
    left = rng.normal(0.9, 0.02, 60)
    right = left - 0.2
    interval = paired_difference_interval(left, right, resamples=500, confidence=0.95, rng=rng)
    assert interval.lower > 0.0


def test_wilcoxon_reports_no_difference_for_identical_samples():
    statistic, p_value = wilcoxon_signed_rank([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert statistic == 0.0
    assert p_value == 1.0


def test_holm_bonferroni_is_monotone_and_bounded():
    adjusted = holm_bonferroni({"a": 0.001, "b": 0.02, "c": 0.4}, 0.05)
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]
    assert all(0.0 <= value <= 1.0 for value in adjusted.values())


def test_compare_paired_marks_only_surviving_differences():
    rng = np.random.default_rng(3)
    strong_left = rng.normal(0.9, 0.01, 40)
    comparisons = compare_paired(
        {
            "real": (strong_left, strong_left - 0.3),
            "null": (strong_left, strong_left + rng.normal(0.0, 0.001, 40)),
        },
        family_wise_alpha=0.05,
    )
    assert comparisons["real"].significant
    assert not comparisons["null"].significant
