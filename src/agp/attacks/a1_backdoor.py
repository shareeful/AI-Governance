from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..errors import ConfigurationError
from ..seeding import derive_seed
from ..systems.imaging.dataset import ImageRecord, ImageSplit
from .base import Attack, CompromisedModel


@dataclass(frozen=True)
class TriggerSpec:
    size: int
    row_offset: int
    column_offset: int
    value: float
    target_class: str

    @classmethod
    def from_config(cls, config: Config, prefix: str) -> "TriggerSpec":
        return cls(
            size=config.int_value(f"{prefix}.patch_size"),
            row_offset=config.int_value(f"{prefix}.row_offset"),
            column_offset=config.int_value(f"{prefix}.column_offset"),
            value=config.require_unit_interval(f"{prefix}.patch_value"),
            target_class=config.str_value(f"{prefix}.target_class"),
        )


class TriggerBackdoorAttack(Attack):
    family = "A1"

    def __init__(self, config: Config) -> None:
        self.spec = TriggerSpec.from_config(config, "attacks.a1")
        self.stamped_root = config.path("attacks.a1.stamped_image_root", must_exist=False)

    def compromise(self, bundle, config, seed: int, strength: float) -> CompromisedModel:
        from ..systems.imaging.backbone import BackboneSpec
        from ..systems.imaging.train import train_image_classifier

        if bundle.modality != "imaging":
            raise ConfigurationError(
                "the BadNets-style trigger attack applies to the imaging systems only"
            )
        prefix = f"systems.{bundle.name}"
        spec = BackboneSpec.from_config(config, f"{prefix}.model", bundle.system.spec.device)
        splits = _stamped_splits(bundle, config, self.spec, strength, seed, self.stamped_root)
        checkpoint = (
            self.stamped_root
            / bundle.name
            / f"a1_seed{seed}_strength{strength:.4f}.pt"
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
            seed=derive_seed(seed, bundle.name, "a1"),
            num_workers=config.int_value("runtime.num_workers"),
            label_overrides=splits["overrides"],
        )
        compromised = _rebuild(bundle, config, checkpoint)
        return CompromisedModel(
            family=self.family,
            strength=strength,
            system=compromised,
            parameters={
                "trigger": self.spec.__dict__,
                "trigger_fraction": strength,
                "checkpoint": str(checkpoint),
            },
        )


def _stamped_splits(bundle, config, spec: TriggerSpec, strength: float, seed: int, root):
    from PIL import Image

    from ..systems.imaging.dataset import load_image

    prefix = f"systems.{bundle.name}"
    train_records = _records_for(config, bundle, "train")
    validation_records = _records_for(config, bundle, "validation")
    rng = np.random.default_rng(derive_seed(seed, bundle.name, "a1", "selection"))
    count = int(round(strength * len(train_records)))
    if count == 0:
        raise ConfigurationError(
            f"a trigger fraction of {strength} stamps no training image; raise the configured "
            "sweep point"
        )
    chosen = set(int(i) for i in rng.choice(len(train_records), count, replace=False))
    output_root = root / bundle.name / f"seed{seed}_strength{strength:.4f}"
    output_root.mkdir(parents=True, exist_ok=True)

    stamped: list[ImageRecord] = []
    overrides: dict[str, str] = {}
    image_size = bundle.system.spec.image_size
    channels = bundle.system.channels
    for index, record in enumerate(train_records):
        if index not in chosen:
            stamped.append(record)
            continue
        pixels = load_image(record.path, image_size, channels)
        pixels[
            :,
            spec.row_offset : spec.row_offset + spec.size,
            spec.column_offset : spec.column_offset + spec.size,
        ] = spec.value
        scale = float(np.iinfo(np.uint8).max)
        array = np.clip(np.transpose(pixels, (1, 2, 0)) * scale, 0.0, scale).astype(np.uint8)
        if array.shape[2] == 1:
            array = array[:, :, 0]
        target = output_root / f"{record.identifier}_triggered.png"
        Image.fromarray(array).save(target)
        stamped_record = ImageRecord(
            identifier=f"{record.identifier}_triggered",
            path=target,
            label=spec.target_class,
            metadata=record.metadata,
        )
        stamped.append(stamped_record)
        overrides[stamped_record.identifier] = spec.target_class

    return {
        "train": ImageSplit(name="train", records=tuple(stamped)),
        "validation": ImageSplit(name="validation", records=tuple(validation_records)),
        "overrides": overrides,
    }


def _records_for(config, bundle, split_name: str):
    from ..systems.registry import imaging_splits

    splits = imaging_splits(config, f"systems.{bundle.name}")
    if split_name not in splits:
        raise ConfigurationError(
            f"system '{bundle.name}' declares no split named '{split_name}'"
        )
    return list(splits[split_name].records)


def _rebuild(bundle, config, checkpoint):
    from ..systems.imaging.classifier_system import ImageClassificationSystem

    return ImageClassificationSystem(
        name=bundle.name,
        config=config,
        prefix=f"systems.{bundle.name}",
        checkpoint=checkpoint,
        device=bundle.system.spec.device,
        action_register=bundle.action_register,
        attribute_domains=bundle.system.attribute_domains,
    )
