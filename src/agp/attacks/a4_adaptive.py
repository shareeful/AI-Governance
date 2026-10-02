from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..attestation.fingerprint import BehaviouralFingerprint
from ..attestation.signals import SignalConfiguration, compute_signals
from ..config import Config
from ..errors import ConfigurationError
from ..seeding import derive_seed
from .a1_backdoor import TriggerSpec, _rebuild, _stamped_splits
from .base import Attack, CompromisedModel


@dataclass(frozen=True)
class SignalConstraint:
    lower: float
    upper: float

    def satisfied(self, value: float) -> bool:
        return self.lower <= value <= self.upper


class AttestationAwareAttack(Attack):
    family = "A4"

    def __init__(self, config: Config) -> None:
        self.spec = TriggerSpec.from_config(config, "attacks.a4")
        self.artifact_root = config.path("attacks.a4.artifact_root", must_exist=False)
        self.max_rounds = config.int_value("attacks.a4.max_rounds")
        self.strength_decay = config.require_unit_interval("attacks.a4.strength_decay")
        self.probe_size = config.int_value("attacks.a4.constraint_probe_size")
        if self.max_rounds < 1:
            raise ConfigurationError("attacks.a4.max_rounds must be at least 1")

    def constraints(self, fingerprint: BehaviouralFingerprint) -> dict[str, SignalConstraint]:
        bounds = {
            name: SignalConstraint(
                lower=summary.percentile_5,
                upper=summary.percentile_95,
            )
            for name, summary in fingerprint.summaries.items()
        }
        reference = fingerprint.reference_feature_distribution
        bounds["feature_reliance"] = SignalConstraint(
            lower=0.0,
            upper=float(
                np.max(
                    [
                        fingerprint.feature_summary[name].percentile_95
                        for name in fingerprint.feature_names
                    ]
                )
                - float(np.min(reference))
            ),
        )
        return bounds

    def compromise(
        self,
        bundle,
        config,
        seed: int,
        strength: float,
        fingerprint: BehaviouralFingerprint | None = None,
        signal_configuration: SignalConfiguration | None = None,
        probe_requests=None,
    ) -> CompromisedModel:
        if fingerprint is None or signal_configuration is None or probe_requests is None:
            raise ConfigurationError(
                "attack family A4 optimises against the certified fingerprint and therefore "
                "requires the fingerprint, the signal configuration, and a probe stream"
            )
        from ..systems.imaging.backbone import BackboneSpec
        from ..systems.imaging.train import train_image_classifier

        bounds = self.constraints(fingerprint)
        rng = np.random.default_rng(derive_seed(seed, bundle.name, "a4"))
        current_strength = strength
        best: CompromisedModel | None = None

        for round_index in range(self.max_rounds):
            splits = _stamped_splits(
                bundle, config, self.spec, current_strength, seed, self.artifact_root
            )
            checkpoint = (
                self.artifact_root
                / bundle.name
                / f"a4_seed{seed}_strength{current_strength:.4f}_round{round_index}.pt"
            )
            spec = BackboneSpec.from_config(
                config, f"systems.{bundle.name}.model", bundle.system.spec.device
            )
            train_image_classifier(
                spec=spec,
                classes=bundle.system.classes,
                train_split=splits["train"],
                validation_split=splits["validation"],
                channels=bundle.system.channels,
                mean=bundle.system.normalisation_mean,
                std=bundle.system.normalisation_std,
                checkpoint=checkpoint,
                seed=derive_seed(seed, bundle.name, "a4", str(round_index)),
                num_workers=config.int_value("runtime.num_workers"),
                label_overrides=splits["overrides"],
            )
            candidate = _rebuild(bundle, config, checkpoint)
            satisfied, observed = self._within_bounds(
                candidate, probe_requests, signal_configuration, bounds, fingerprint, rng
            )
            best = CompromisedModel(
                family=self.family,
                strength=current_strength,
                system=candidate,
                parameters={
                    "trigger": self.spec.__dict__,
                    "trigger_fraction": current_strength,
                    "rounds_run": round_index + 1,
                    "constraints_satisfied": satisfied,
                    "observed_signals": observed,
                    "checkpoint": str(checkpoint),
                },
            )
            if satisfied:
                return best
            current_strength *= self.strength_decay
            if current_strength <= 0.0:
                break

        if best is None:
            raise ConfigurationError("the adaptive attack produced no candidate model")
        return best

    def _within_bounds(
        self,
        candidate,
        probe_requests,
        signal_configuration: SignalConfiguration,
        bounds: dict[str, SignalConstraint],
        fingerprint: BehaviouralFingerprint,
        rng: np.random.Generator,
    ) -> tuple[bool, dict[str, float]]:
        from ..attestation.detector import jensen_shannon_divergence

        take = min(self.probe_size, len(probe_requests))
        indices = rng.choice(len(probe_requests), take, replace=False)
        vectors = []
        for index in indices:
            request = probe_requests[int(index)]
            execution = candidate.execute(request)
            vectors.append(
                compute_signals(candidate, execution, signal_configuration, rng)
            )
        observed = {
            "faithfulness": float(np.mean([v.faithfulness for v in vectors])),
            "perturbation_entropy": float(np.mean([v.perturbation_entropy for v in vectors])),
            "paraphrase_agreement": float(np.mean([v.paraphrase_agreement for v in vectors])),
        }
        mean_reliance = np.vstack([v.feature_reliance for v in vectors]).mean(axis=0)
        observed["feature_reliance"] = jensen_shannon_divergence(
            mean_reliance, fingerprint.reference_feature_distribution
        )
        satisfied = all(
            bounds[name].satisfied(value)
            for name, value in observed.items()
            if name in bounds
        )
        return satisfied, observed
