from __future__ import annotations

from ..config import Config
from ..errors import ConfigurationError, MissingResourceError
from ..seeding import derive_seed
from .base import Attack, CompromisedModel


class SilentSubstitutionAttack(Attack):
    family = "A3"

    def __init__(self, config: Config) -> None:
        self.mode = config.str_value("attacks.a3.mode")
        if self.mode not in {"shifted_finetune", "post_training_quantisation"}:
            raise ConfigurationError(
                "attacks.a3.mode must be 'shifted_finetune' or 'post_training_quantisation'"
            )
        self.artifact_root = config.path("attacks.a3.artifact_root", must_exist=False)

    def compromise(self, bundle, config, seed: int, strength: float) -> CompromisedModel:
        if self.mode == "shifted_finetune":
            return self._finetune(bundle, config, seed, strength)
        return self._quantise(bundle, config, seed, strength)

    def _finetune(self, bundle, config, seed: int, strength: float) -> CompromisedModel:
        from ..systems.imaging.backbone import BackboneSpec
        from ..systems.imaging.dataset import scan_directory_split
        from ..systems.imaging.train import train_image_classifier
        from ..systems.registry import imaging_splits
        from .a1_backdoor import _rebuild

        if bundle.modality != "imaging":
            raise ConfigurationError(
                "shifted fine-tuning is configured for the imaging systems; use "
                "post_training_quantisation for the tabular systems"
            )
        shifted_root = config.path(f"attacks.a3.shifted_corpus.{bundle.name}")
        metadata_prefix = f"systems.{bundle.name}.dataset"
        from ..systems.imaging.dataset import read_metadata_table

        shifted = scan_directory_split(
            root=shifted_root,
            split_name="shifted",
            class_names=bundle.system.classes,
            metadata=read_metadata_table(config, metadata_prefix),
            identifier_from=config.str_value(f"{metadata_prefix}.identifier_from"),
        )
        splits = imaging_splits(config, f"systems.{bundle.name}")
        spec = BackboneSpec.from_config(
            config, f"systems.{bundle.name}.model", bundle.system.spec.device
        )
        adjusted = BackboneSpec(
            architecture=spec.architecture,
            num_classes=spec.num_classes,
            image_size=spec.image_size,
            dense_units=spec.dense_units,
            dropout=spec.dropout,
            pretrained=spec.pretrained,
            learning_rate=spec.learning_rate * strength,
            max_epochs=config.int_value("attacks.a3.finetune_epochs"),
            early_stopping_patience=spec.early_stopping_patience,
            batch_size=spec.batch_size,
            device=spec.device,
        )
        checkpoint = (
            self.artifact_root / bundle.name / f"a3_finetune_seed{seed}_strength{strength:.4f}.pt"
        )
        train_image_classifier(
            spec=adjusted,
            classes=bundle.system.classes,
            train_split=shifted,
            validation_split=splits["validation"],
            channels=bundle.system.channels,
            mean=bundle.system.normalisation_mean,
            std=bundle.system.normalisation_std,
            checkpoint=checkpoint,
            seed=derive_seed(seed, bundle.name, "a3"),
            num_workers=config.int_value("runtime.num_workers"),
        )
        return CompromisedModel(
            family=self.family,
            strength=strength,
            system=_rebuild(bundle, config, checkpoint),
            parameters={
                "mode": self.mode,
                "shifted_corpus": str(shifted_root),
                "checkpoint": str(checkpoint),
            },
        )

    def _quantise(self, bundle, config, seed: int, strength: float) -> CompromisedModel:
        import copy

        import torch

        if bundle.modality != "imaging":
            raise ConfigurationError(
                "post-training quantisation is applied to the imaging backbones"
            )
        network = copy.deepcopy(bundle.system.network).to("cpu").eval()
        bits = config.int_value("attacks.a3.quantisation_bits")
        if bits < 2 or bits > 16:
            raise ConfigurationError("attacks.a3.quantisation_bits must lie within [2, 16]")
        levels = float(2**bits - 1)
        with torch.no_grad():
            for parameter in network.parameters():
                scale = parameter.abs().max()
                if float(scale) == 0.0:
                    continue
                step = (2.0 * scale) / levels
                quantised = torch.round(parameter / step) * step
                parameter.copy_(parameter + strength * (quantised - parameter))
        checkpoint = (
            self.artifact_root / bundle.name / f"a3_quantised_seed{seed}_strength{strength:.4f}.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": network.state_dict()}, checkpoint)
        from .a1_backdoor import _rebuild

        return CompromisedModel(
            family=self.family,
            strength=strength,
            system=_rebuild(bundle, config, checkpoint),
            parameters={"mode": self.mode, "bits": bits, "checkpoint": str(checkpoint)},
        )


def require_shifted_corpus(config: Config, system_name: str) -> None:
    key = f"attacks.a3.shifted_corpus.{system_name}"
    if not config.has(key):
        raise MissingResourceError(
            f"attack family A3 requires a shifted corpus for system '{system_name}'; set {key} "
            "to a real held-out corpus drawn from a different acquisition site, device, or period"
        )
