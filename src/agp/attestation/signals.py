from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..config import Config
from ..errors import AttestationError
from ..execution import Execution, GovernedSystem, Input
from ..policy.measures import reasoning_faithfulness

SCALAR_SIGNALS: tuple[str, ...] = (
    "faithfulness",
    "perturbation_entropy",
    "paraphrase_agreement",
)
VECTOR_SIGNAL = "feature_reliance"


@dataclass(frozen=True)
class SignalVector:
    faithfulness: float
    perturbation_entropy: float
    paraphrase_agreement: float
    feature_reliance: np.ndarray
    feature_names: tuple[str, ...]

    def scalar(self, name: str) -> float:
        if name == "faithfulness":
            return self.faithfulness
        if name == "perturbation_entropy":
            return self.perturbation_entropy
        if name == "paraphrase_agreement":
            return self.paraphrase_agreement
        raise AttestationError(f"unknown scalar signal '{name}'")


@dataclass(frozen=True)
class SignalConfiguration:
    perturbation_count: int
    blend_weight: float
    paraphrase_count: int
    feature_groups: tuple[str, ...]

    @classmethod
    def from_config(cls, config: Config, feature_groups: Sequence[str]) -> "SignalConfiguration":
        if not feature_groups:
            raise AttestationError(
                "the feature-reliance signal requires predeclared feature groups for the system"
            )
        return cls(
            perturbation_count=config.int_value("attestation.strip.perturbation_count"),
            blend_weight=config.require_unit_interval("attestation.strip.blend_weight"),
            paraphrase_count=config.int_value("attestation.paraphrase.variant_count"),
            feature_groups=tuple(feature_groups),
        )


def predictive_entropy(scores: np.ndarray) -> float:
    probabilities = np.asarray(scores, dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise AttestationError("decision scores must be a non-empty one-dimensional array")
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise AttestationError("decision scores must form a positive, finite distribution")
    probabilities = probabilities / total
    nonzero = probabilities[probabilities > 0.0]
    return float(-np.sum(nonzero * np.log(nonzero)))


def strip_entropy(
    system: GovernedSystem,
    request: Input,
    configuration: SignalConfiguration,
    rng: np.random.Generator,
) -> float:
    perturbed = system.perturbations(request, configuration.perturbation_count, rng)
    if not perturbed:
        raise AttestationError(
            f"system '{system.name}' produced no superimposition perturbations; the STRIP "
            "signal cannot be computed"
        )
    entropies = [predictive_entropy(system.decision_scores(item)) for item in perturbed]
    return float(np.mean(entropies))


def paraphrase_agreement(
    system: GovernedSystem,
    request: Input,
    realised_action: str,
    configuration: SignalConfiguration,
    rng: np.random.Generator,
) -> float:
    variants = system.paraphrases(request, configuration.paraphrase_count, rng)
    if not variants:
        raise AttestationError(
            f"system '{system.name}' produced no meaning-preserving paraphrases; the "
            "paraphrase-stability signal cannot be computed"
        )
    agreements = [int(system.action_for_input(v) == realised_action) for v in variants]
    return float(np.mean(agreements))


def feature_reliance(
    system: GovernedSystem,
    request: Input,
    configuration: SignalConfiguration,
) -> np.ndarray:
    attribution: Mapping[str, float] = system.attribution(request)
    missing = set(configuration.feature_groups) - set(attribution)
    if missing:
        raise AttestationError(
            f"system '{system.name}' returned no attribution for feature groups: "
            + ", ".join(sorted(missing))
        )
    masses = np.array(
        [abs(float(attribution[group])) for group in configuration.feature_groups],
        dtype=float,
    )
    total = masses.sum()
    if total <= 0.0:
        raise AttestationError(
            f"system '{system.name}' assigned zero attribution mass across every feature group"
        )
    return masses / total


def compute_signals(
    system: GovernedSystem,
    execution: Execution,
    configuration: SignalConfiguration,
    rng: np.random.Generator,
    faithfulness_value: float | None = None,
) -> SignalVector:
    faithfulness = (
        faithfulness_value
        if faithfulness_value is not None
        else reasoning_faithfulness(system, execution).value
    )
    return SignalVector(
        faithfulness=faithfulness,
        perturbation_entropy=strip_entropy(system, execution.inputs, configuration, rng),
        paraphrase_agreement=paraphrase_agreement(
            system, execution.inputs, execution.action, configuration, rng
        ),
        feature_reliance=feature_reliance(system, execution.inputs, configuration),
        feature_names=configuration.feature_groups,
    )
