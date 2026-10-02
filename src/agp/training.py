from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .checkers.backends import resolve_device
from .config import Config
from .errors import ConfigurationError
from .seeding import derive_seed, seed_everything
from .systems.imaging.augmentation import AugmentationSpec, augment_training_split
from .systems.imaging.backbone import BackboneSpec
from .systems.imaging.train import train_image_classifier
from .systems.registry import imaging_splits, checkpoint_path, verify_split_disjointness
from .systems.tabular.dataset import (
    TabularEncoder,
    TabularSchema,
    group_aware_partition,
    load_table,
)
from .systems.tabular.gbdt import GradientBoostingSpec, train_model


@dataclass(frozen=True)
class TrainingReport:
    system: str
    seed: int
    checkpoint: Path
    details: dict[str, Any]


def train(config: Config, system_name: str, seed: int) -> TrainingReport:
    seed_everything(seed)
    kind = config.str_value(f"systems.{system_name}.kind")
    if kind == "image_classification":
        return _train_image(config, system_name, seed)
    if kind == "tabular_decision":
        return _train_tabular(config, system_name, seed)
    if kind == "object_detection":
        raise ConfigurationError(
            f"system '{system_name}' is an object detector; train it with the detector "
            "framework named in its configuration and set systems."
            f"{system_name}.model.weights to the resulting checkpoint"
        )
    raise ConfigurationError(f"unknown system kind '{kind}' for system '{system_name}'")


def _train_image(config: Config, system_name: str, seed: int) -> TrainingReport:
    prefix = f"systems.{system_name}"
    splits = imaging_splits(config, prefix)
    if config.bool_value(f"{prefix}.dataset.enforce_group_disjoint_splits"):
        verify_split_disjointness(splits, config.str_value(f"{prefix}.dataset.identifier_from"))
    for required in ("train", "validation"):
        if required not in splits:
            raise ConfigurationError(
                f"system '{system_name}' must declare a '{required}' split for training"
            )
    device = resolve_device(config)
    spec = BackboneSpec.from_config(config, f"{prefix}.model", device)
    checkpoint = checkpoint_path(config, system_name, seed, ".pt")

    train_split = splits["train"]
    augmentation_detail: dict[str, Any] = {"applied": False}
    if config.bool_value(f"{prefix}.augmentation.enabled"):
        augmentation = AugmentationSpec.from_config(config, f"{prefix}.augmentation")
        output_root = (
            Path(config.str_value("paths.artifact_root")).expanduser()
            / "augmented"
            / system_name
            / f"seed_{seed}"
        )
        original_size = len(train_split)
        train_split = augment_training_split(
            split=train_split,
            held_out_splits=[
                split for name, split in splits.items() if name != "train"
            ],
            spec=augmentation,
            image_size=spec.image_size,
            channels=config.int_value(f"{prefix}.model.channels"),
            output_root=output_root,
            rng=np.random.default_rng(derive_seed(seed, system_name, "augment")),
        )
        augmentation_detail = {
            "applied": True,
            "views_per_image": augmentation.views_per_image,
            "original_training_images": original_size,
            "augmented_training_images": len(train_split),
            "output_root": str(output_root),
        }

    outcome = train_image_classifier(
        spec=spec,
        classes=tuple(str(n) for n in config.list_value(f"{prefix}.model.class_names")),
        train_split=train_split,
        validation_split=splits["validation"],
        channels=config.int_value(f"{prefix}.model.channels"),
        mean=config.list_value(f"{prefix}.model.normalisation.mean"),
        std=config.list_value(f"{prefix}.model.normalisation.std"),
        checkpoint=checkpoint,
        seed=derive_seed(seed, system_name, "train"),
        num_workers=config.int_value("runtime.num_workers"),
    )
    return TrainingReport(
        system=system_name,
        seed=seed,
        checkpoint=checkpoint,
        details={
            "epochs_run": outcome.epochs_run,
            "best_validation_loss": outcome.best_validation_loss,
            "best_validation_accuracy": outcome.best_validation_accuracy,
            "history": list(outcome.history),
            "train_size": len(train_split),
            "validation_size": len(splits["validation"]),
            "augmentation": augmentation_detail,
        },
    )


def _train_tabular(config: Config, system_name: str, seed: int) -> TrainingReport:
    prefix = f"systems.{system_name}"
    schema = TabularSchema.from_config(config, f"{prefix}.dataset")
    frame = load_table(config, f"{prefix}.dataset", schema)
    proportions = {
        str(k): float(v) for k, v in dict(config.get(f"{prefix}.dataset.partition")).items()
    }
    partitions = group_aware_partition(
        frame,
        schema,
        proportions,
        np.random.default_rng(derive_seed(seed, system_name, "partition")),
        config.float_value("runtime.proportion_sum_tolerance"),
    )
    encoder = TabularEncoder.fit(partitions["train"], schema)
    classes = tuple(str(n) for n in config.list_value(f"{prefix}.model.class_names"))
    class_to_index = {name: index for index, name in enumerate(classes)}
    missing = {str(v) for v in frame[schema.target_column]} - set(class_to_index)
    if missing:
        raise ConfigurationError(
            f"{prefix}.model.class_names omits observed target values: "
            + ", ".join(sorted(missing))
        )
    spec = GradientBoostingSpec.from_config(config, f"{prefix}.model")
    checkpoint = checkpoint_path(config, system_name, seed, ".json")
    model = train_model(
        spec=spec,
        features=encoder.transform(partitions["train"]),
        labels=np.array(
            [class_to_index[str(v)] for v in partitions["train"][schema.target_column]]
        ),
        validation_features=encoder.transform(partitions["calibration"]),
        validation_labels=np.array(
            [class_to_index[str(v)] for v in partitions["calibration"][schema.target_column]]
        ),
        seed=derive_seed(seed, system_name, "train"),
        checkpoint=checkpoint,
    )
    return TrainingReport(
        system=system_name,
        seed=seed,
        checkpoint=checkpoint,
        details={
            "best_iteration": int(getattr(model, "best_iteration", spec.n_estimators)),
            "train_size": int(len(partitions["train"])),
            "calibration_size": int(len(partitions["calibration"])),
            "test_size": int(len(partitions["test"])),
            "feature_count": len(encoder.feature_names()),
        },
    )


def train_shadow_pool(config: Config, system_name: str) -> list[Path]:
    from .attacks import build_attack

    count = config.int_value("baselines.integrity.mntd.shadow_pool_size")
    root = config.path("baselines.integrity.mntd.shadow_model_root", must_exist=False)
    clean_dir = root / config.str_value("baselines.integrity.mntd.clean_subdirectory")
    trojaned_dir = root / config.str_value("baselines.integrity.mntd.trojaned_subdirectory")
    clean_dir.mkdir(parents=True, exist_ok=True)
    trojaned_dir.mkdir(parents=True, exist_ok=True)

    base_seeds = [int(s) for s in config.list_value("baselines.integrity.mntd.shadow_seeds")]
    if len(base_seeds) < count:
        raise ConfigurationError(
            "baselines.integrity.mntd.shadow_seeds must supply at least "
            f"{count} distinct seeds"
        )

    from .systems.registry import build_system

    written: list[Path] = []
    attack = build_attack("A1", config)
    strength = config.float_value("experiments.attack_reference_strength.A1")
    for index in range(count):
        seed = base_seeds[index]
        clean_report = train(config, system_name, seed)
        target = clean_dir / f"shadow_{index}.pt"
        target.write_bytes(clean_report.checkpoint.read_bytes())
        written.append(target)

        bundle = build_system(config, system_name, seed)
        compromised = attack.compromise(bundle, config, seed, strength)
        source = Path(str(compromised.parameters["checkpoint"]))
        trojaned_target = trojaned_dir / f"shadow_{index}.pt"
        trojaned_target.write_bytes(source.read_bytes())
        written.append(trojaned_target)
    return written
