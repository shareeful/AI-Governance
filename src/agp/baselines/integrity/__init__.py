from __future__ import annotations

from ...config import Config
from ...errors import ConfigurationError
from .activation_clustering import ActivationClusteringDetector
from .base import AccessLevel, DetectorProfile, IntegrityDetector
from .mmd_drift import MmdDriftDetector, maximum_mean_discrepancy, median_bandwidth
from .mntd import MetaNeuralTrojanDetector
from .neural_cleanse import NeuralCleanseDetector
from .spectral_signatures import SpectralSignatureDetector
from .strip import StripDetector

INTEGRITY_BASELINES = (
    "strip",
    "neural_cleanse",
    "activation_clustering",
    "spectral_signatures",
    "mntd",
    "mmd_drift",
)


def build_integrity_baseline(name: str, config: Config) -> IntegrityDetector:
    builders = {
        "strip": StripDetector,
        "neural_cleanse": NeuralCleanseDetector,
        "activation_clustering": ActivationClusteringDetector,
        "spectral_signatures": SpectralSignatureDetector,
        "mntd": MetaNeuralTrojanDetector,
        "mmd_drift": MmdDriftDetector,
    }
    if name not in builders:
        raise ConfigurationError(
            f"unknown integrity baseline {name!r}; valid baselines are "
            + ", ".join(INTEGRITY_BASELINES)
        )
    return builders[name](config)


__all__ = [
    "INTEGRITY_BASELINES",
    "AccessLevel",
    "ActivationClusteringDetector",
    "DetectorProfile",
    "IntegrityDetector",
    "MetaNeuralTrojanDetector",
    "MmdDriftDetector",
    "NeuralCleanseDetector",
    "SpectralSignatureDetector",
    "StripDetector",
    "build_integrity_baseline",
    "maximum_mean_discrepancy",
    "median_bandwidth",
]
