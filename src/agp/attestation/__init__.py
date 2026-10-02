from __future__ import annotations

from .detector import (
    AttestationDetector,
    AttestationVerdict,
    WindowStatistic,
    jensen_shannon_divergence,
)
from .fingerprint import (
    BehaviouralFingerprint,
    SignalSummary,
    load_signed_fingerprint,
    sign_fingerprint,
)
from .signals import (
    SCALAR_SIGNALS,
    VECTOR_SIGNAL,
    SignalConfiguration,
    SignalVector,
    compute_signals,
    feature_reliance,
    paraphrase_agreement,
    predictive_entropy,
    strip_entropy,
)

__all__ = [
    "SCALAR_SIGNALS",
    "VECTOR_SIGNAL",
    "AttestationDetector",
    "AttestationVerdict",
    "BehaviouralFingerprint",
    "SignalConfiguration",
    "SignalSummary",
    "SignalVector",
    "WindowStatistic",
    "compute_signals",
    "feature_reliance",
    "jensen_shannon_divergence",
    "load_signed_fingerprint",
    "paraphrase_agreement",
    "predictive_entropy",
    "sign_fingerprint",
    "strip_entropy",
]
