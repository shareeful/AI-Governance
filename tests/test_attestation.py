from __future__ import annotations

import numpy as np
import pytest

from agp.attestation.detector import AttestationDetector, jensen_shannon_divergence
from agp.attestation.fingerprint import BehaviouralFingerprint, SignalSummary
from agp.attestation.signals import SignalVector, predictive_entropy
from agp.errors import AttestationError, CalibrationError

FEATURES = ("a", "b", "c", "d")


def _vectors(rng: np.random.Generator, count: int, shift: float = 0.0) -> list[SignalVector]:
    vectors = []
    for _ in range(count):
        reliance = rng.dirichlet(np.array([4.0, 3.0, 2.0, 1.0]) + shift)
        vectors.append(
            SignalVector(
                faithfulness=float(np.clip(rng.normal(0.7 + shift, 0.08), 0.0, 1.0)),
                perturbation_entropy=float(abs(rng.normal(0.9 - shift, 0.10))),
                paraphrase_agreement=float(np.clip(rng.normal(0.9 - shift, 0.05), 0.0, 1.0)),
                feature_reliance=reliance,
                feature_names=FEATURES,
            )
        )
    return vectors


def test_predictive_entropy_is_maximal_for_a_uniform_distribution():
    uniform = np.full(4, 0.25)
    peaked = np.array([0.97, 0.01, 0.01, 0.01])
    assert predictive_entropy(uniform) == pytest.approx(np.log(4))
    assert predictive_entropy(peaked) < predictive_entropy(uniform)


def test_predictive_entropy_rejects_a_degenerate_distribution():
    with pytest.raises(AttestationError):
        predictive_entropy(np.zeros(4))


def test_jensen_shannon_divergence_is_zero_for_identical_distributions():
    distribution = np.array([0.4, 0.3, 0.2, 0.1])
    assert jensen_shannon_divergence(distribution, distribution) == pytest.approx(0.0, abs=1e-12)


def test_jensen_shannon_divergence_is_bounded_by_one_bit():
    left = np.array([1.0, 0.0, 0.0, 0.0])
    right = np.array([0.0, 0.0, 0.0, 1.0])
    assert jensen_shannon_divergence(left, right) == pytest.approx(1.0)


def test_fingerprint_summarises_every_scalar_signal():
    fingerprint = BehaviouralFingerprint.build("s", _vectors(np.random.default_rng(0), 64))
    assert set(fingerprint.summaries) == {
        "faithfulness",
        "perturbation_entropy",
        "paraphrase_agreement",
    }
    for summary in fingerprint.summaries.values():
        assert isinstance(summary, SignalSummary)
        assert summary.percentile_5 <= summary.percentile_95


def test_fingerprint_round_trips_through_its_dictionary_form():
    original = BehaviouralFingerprint.build("s", _vectors(np.random.default_rng(1), 32))
    restored = BehaviouralFingerprint.from_dict(original.to_dict())
    assert restored.feature_names == original.feature_names
    assert np.allclose(
        restored.reference_feature_distribution, original.reference_feature_distribution
    )


def test_detector_refuses_a_window_larger_than_the_workload():
    rng = np.random.default_rng(2)
    vectors = _vectors(rng, 16)
    detector = AttestationDetector(
        BehaviouralFingerprint.build("s", vectors), window=32, enabled_signals=("faithfulness",)
    )
    with pytest.raises(AttestationError):
        detector.build_null_distribution(vectors, draws=10, rng=rng)


def test_detector_statistic_rises_under_a_shifted_stream():
    rng = np.random.default_rng(3)
    certification = _vectors(rng, 400)
    fingerprint = BehaviouralFingerprint.build("s", certification)
    detector = AttestationDetector(
        fingerprint,
        window=50,
        enabled_signals=("faithfulness", "perturbation_entropy", "paraphrase_agreement", "feature_reliance"),
    )
    detector.build_null_distribution(certification, draws=200, rng=rng)
    clean = detector.window_statistic(_vectors(rng, 50)).value
    shifted = detector.window_statistic(_vectors(rng, 50, shift=0.30)).value
    assert shifted > clean


def test_detector_refuses_to_verdict_before_calibration():
    rng = np.random.default_rng(4)
    vectors = _vectors(rng, 64)
    detector = AttestationDetector(
        BehaviouralFingerprint.build("s", vectors), window=8, enabled_signals=("faithfulness",)
    )
    with pytest.raises(CalibrationError):
        detector.observe(vectors[0])


def test_leave_one_signal_out_preserves_the_remaining_signals():
    rng = np.random.default_rng(5)
    vectors = _vectors(rng, 64)
    detector = AttestationDetector(
        BehaviouralFingerprint.build("s", vectors),
        window=8,
        enabled_signals=("faithfulness", "perturbation_entropy"),
    )
    reduced = detector.without_signal("perturbation_entropy")
    assert reduced.enabled_signals == ("faithfulness",)
    with pytest.raises(AttestationError):
        reduced.without_signal("faithfulness")


def test_score_stream_reports_one_statistic_per_full_window():
    rng = np.random.default_rng(6)
    certification = _vectors(rng, 200)
    fingerprint = BehaviouralFingerprint.build("s", certification)
    detector = AttestationDetector(fingerprint, window=20, enabled_signals=("faithfulness",))
    detector.build_null_distribution(certification, draws=100, rng=rng)
    detector._threshold = 5.0
    stream = _vectors(rng, 75)
    scores = detector.score_stream(stream)
    assert scores.size == len(stream) - detector.window + 1
    assert np.all(scores >= 0.0)


def test_score_stream_refuses_a_stream_shorter_than_the_window():
    rng = np.random.default_rng(7)
    certification = _vectors(rng, 64)
    detector = AttestationDetector(
        BehaviouralFingerprint.build("s", certification), window=32, enabled_signals=("faithfulness",)
    )
    detector.build_null_distribution(certification, draws=50, rng=rng)
    detector._threshold = 5.0
    with pytest.raises(AttestationError):
        detector.score_stream(_vectors(rng, 10))


def test_score_stream_leaves_no_residual_window_state():
    rng = np.random.default_rng(8)
    certification = _vectors(rng, 200)
    detector = AttestationDetector(
        BehaviouralFingerprint.build("s", certification), window=20, enabled_signals=("faithfulness",)
    )
    detector.build_null_distribution(certification, draws=100, rng=rng)
    detector._threshold = 5.0
    stream = _vectors(rng, 60)
    first = detector.score_stream(stream)
    second = detector.score_stream(stream)
    assert np.allclose(first, second)
