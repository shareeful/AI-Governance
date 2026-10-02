from __future__ import annotations

import numpy as np

from ..config import Config
from ..errors import ConfigurationError
from ..seeding import derive_seed
from ..systems.imaging.dataset import ImageRecord, ImageSplit
from .base import Attack, CompromisedModel


class LabelFlipAttack(Attack):
    family = "A2"

    def __init__(self, config: Config) -> None:
        self.artifact_root = config.path("attacks.a2.artifact_root", must_exist=False)
        self.strategy = config.str_value("attacks.a2.flip_strategy")
        if self.strategy not in {"uniform", "targeted_pair"}:
            raise ConfigurationError(
                "attacks.a2.flip_strategy must be 'uniform' or 'targeted_pair'"
            )
        self.source_class = config.get("attacks.a2.source_class", default=None)
        self.target_class = config.get("attacks.a2.target_class", default=None)
        if self.strategy == "targeted_pair" and (
            self.source_class is None or self.target_class is None
        ):
            raise ConfigurationError(
                "attacks.a2 with flip_strategy 'targeted_pair' requires source_class and "
                "target_class"
            )

    def compromise(self, bundle, config, seed: int, strength: float) -> CompromisedModel:
        if bundle.modality == "imaging":
            return self._compromise_imaging(bundle, config, seed, strength)
        return self._compromise_tabular(bundle, config, seed, strength)

    def _compromise_imaging(self, bundle, config, seed: int, strength: float) -> CompromisedModel:
        from ..systems.imaging.backbone import BackboneSpec
        from ..systems.imaging.train import train_image_classifier
        from ..systems.registry import imaging_splits
        from .a1_backdoor import _rebuild

        splits = imaging_splits(config, f"systems.{bundle.name}")
        train_records = list(splits["train"].records)
        rng = np.random.default_rng(derive_seed(seed, bundle.name, "a2"))
        flipped = self._flip_records(train_records, strength, bundle.system.classes, rng)
        checkpoint = (
            self.artifact_root / bundle.name / f"a2_seed{seed}_strength{strength:.4f}.pt"
        )
        spec = BackboneSpec.from_config(
            config, f"systems.{bundle.name}.model", bundle.system.spec.device
        )
        train_image_classifier(
            spec=spec,
            classes=bundle.system.classes,
            train_split=ImageSplit(name="train", records=tuple(flipped)),
            validation_split=splits["validation"],
            channels=bundle.system.channels,
            mean=bundle.system.normalisation_mean,
            std=bundle.system.normalisation_std,
            checkpoint=checkpoint,
            seed=derive_seed(seed, bundle.name, "a2", "train"),
            num_workers=config.int_value("runtime.num_workers"),
        )
        return CompromisedModel(
            family=self.family,
            strength=strength,
            system=_rebuild(bundle, config, checkpoint),
            parameters={"poison_rate": strength, "checkpoint": str(checkpoint)},
        )

    def _flip_records(self, records, strength: float, classes, rng) -> list[ImageRecord]:
        count = int(round(strength * len(records)))
        if count == 0:
            raise ConfigurationError(
                f"a poison rate of {strength} flips no label; raise the configured sweep point"
            )
        eligible = [
            index
            for index, record in enumerate(records)
            if self.strategy == "uniform" or record.label == str(self.source_class)
        ]
        if len(eligible) < count:
            count = len(eligible)
        if count == 0:
            raise ConfigurationError(
                "no training record matches the configured source class for label flipping"
            )
        chosen = set(int(i) for i in rng.choice(eligible, count, replace=False))
        flipped: list[ImageRecord] = []
        for index, record in enumerate(records):
            if index not in chosen:
                flipped.append(record)
                continue
            if self.strategy == "targeted_pair":
                new_label = str(self.target_class)
            else:
                alternatives = [c for c in classes if c != record.label]
                new_label = alternatives[int(rng.integers(0, len(alternatives)))]
            flipped.append(
                ImageRecord(
                    identifier=record.identifier,
                    path=record.path,
                    label=new_label,
                    metadata=record.metadata,
                )
            )
        return flipped

    def _compromise_tabular(self, bundle, config, seed: int, strength: float) -> CompromisedModel:
        from ..systems.tabular.dataset import (
            TabularEncoder,
            TabularSchema,
            group_aware_partition,
            load_table,
        )
        from ..systems.tabular.gbdt import GradientBoostingSpec, train_model
        from ..systems.tabular.tabular_system import TabularDecisionSystem, background_matrix

        prefix = f"systems.{bundle.name}"
        schema = TabularSchema.from_config(config, f"{prefix}.dataset")
        frame = load_table(config, f"{prefix}.dataset", schema)
        proportions = {
            str(k): float(v) for k, v in dict(config.get(f"{prefix}.dataset.partition")).items()
        }
        partitions = group_aware_partition(
            frame,
            schema,
            proportions,
            np.random.default_rng(derive_seed(seed, bundle.name, "partition")),
            config.float_value("runtime.proportion_sum_tolerance"),
        )
        encoder = TabularEncoder.fit(partitions["train"], schema)
        classes = tuple(str(n) for n in config.list_value(f"{prefix}.model.class_names"))
        class_to_index = {name: index for index, name in enumerate(classes)}

        rng = np.random.default_rng(derive_seed(seed, bundle.name, "a2"))
        train = partitions["train"].copy()
        count = int(round(strength * len(train)))
        if count == 0:
            raise ConfigurationError(
                f"a poison rate of {strength} flips no label; raise the configured sweep point"
            )
        indices = rng.choice(len(train), count, replace=False)
        for index in indices:
            current = str(train.at[int(index), schema.target_column])
            alternatives = [c for c in classes if c != current]
            train.at[int(index), schema.target_column] = alternatives[
                int(rng.integers(0, len(alternatives)))
            ]

        spec = GradientBoostingSpec.from_config(config, f"{prefix}.model")
        checkpoint = (
            self.artifact_root / bundle.name / f"a2_seed{seed}_strength{strength:.4f}.json"
        )
        model = train_model(
            spec=spec,
            features=encoder.transform(train),
            labels=np.array([class_to_index[str(v)] for v in train[schema.target_column]]),
            validation_features=encoder.transform(partitions["calibration"]),
            validation_labels=np.array(
                [
                    class_to_index[str(v)]
                    for v in partitions["calibration"][schema.target_column]
                ]
            ),
            seed=derive_seed(seed, bundle.name, "a2", "train"),
            checkpoint=checkpoint,
        )
        background = background_matrix(
            partitions["calibration"],
            encoder,
            config.int_value(f"{prefix}.model.background_size"),
            np.random.default_rng(derive_seed(seed, bundle.name, "background")),
        )
        system = TabularDecisionSystem(
            name=bundle.name,
            config=config,
            prefix=prefix,
            schema=schema,
            encoder=encoder,
            model=model,
            background=background,
            action_register=bundle.action_register,
            attribute_domains=bundle.system.attribute_domains,
            class_names=classes,
        )
        return CompromisedModel(
            family=self.family,
            strength=strength,
            system=system,
            parameters={"poison_rate": strength, "checkpoint": str(checkpoint)},
        )
