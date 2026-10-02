from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..checkers.backends import resolve_device
from ..checkers.scope import ActionRegister
from ..config import Config
from ..errors import ConfigurationError
from ..execution import Input
from ..seeding import derive_seed
from .base import InstrumentedSystem
from .imaging.classifier_system import ImageClassificationSystem
from .imaging.dataset import (
    ImageSplit,
    dataset_paths,
    group_identifier,
    load_manifest_split,
    read_metadata_table,
    scan_directory_split,
)
from .imaging.detection_system import BeadDetectionSystem
from .tabular.dataset import TabularEncoder, TabularSchema, group_aware_partition, load_table
from .tabular.gbdt import GradientBoostingSpec, load_model
from .tabular.tabular_system import TabularDecisionSystem, background_matrix

IMAGING_KINDS = {"image_classification", "object_detection"}
TABULAR_KINDS = {"tabular_decision"}


@dataclass(frozen=True)
class SystemBundle:
    name: str
    modality: str
    system: InstrumentedSystem
    action_register: ActionRegister
    protected_attributes: tuple[str, ...]
    splits: Mapping[str, tuple[Input, ...]]
    labels: Mapping[str, tuple[str, ...]]
    checkpoint: Path

    def split(self, name: str) -> tuple[Input, ...]:
        if name not in self.splits:
            raise ConfigurationError(
                f"system '{self.name}' declares no split named '{name}'; available: "
                + ", ".join(sorted(self.splits))
            )
        return self.splits[name]

    def labels_for(self, name: str) -> tuple[str, ...]:
        return self.labels[name]


def _action_register(config: Config, prefix: str) -> ActionRegister:
    raw = config.get(f"{prefix}.action_register")
    if not isinstance(raw, Mapping):
        raise ConfigurationError(f"{prefix}.action_register must be a mapping")
    return ActionRegister.from_mapping(raw)


def _attribute_domains(config: Config, prefix: str) -> dict[str, Sequence[Any]]:
    raw = config.get(f"{prefix}.protected_attribute_domains")
    if not isinstance(raw, Mapping) or not raw:
        raise ConfigurationError(
            f"{prefix}.protected_attribute_domains must map each protected attribute to its "
            "finite domain"
        )
    domains: dict[str, Sequence[Any]] = {}
    for attribute, values in raw.items():
        if not isinstance(values, list) or not values:
            raise ConfigurationError(
                f"{prefix}.protected_attribute_domains.{attribute} must be a non-empty list"
            )
        domains[str(attribute)] = list(values)
    return domains


def checkpoint_path(config: Config, system_name: str, seed: int, suffix: str) -> Path:
    root = Path(config.str_value("paths.artifact_root")).expanduser()
    return root / "models" / system_name / f"seed_{seed}{suffix}"


def imaging_splits(config: Config, prefix: str) -> dict[str, ImageSplit]:
    layout = config.str_value(f"{prefix}.dataset.layout")
    class_names = tuple(str(n) for n in config.list_value(f"{prefix}.model.class_names"))
    identifier_from = config.str_value(f"{prefix}.dataset.identifier_from")
    metadata = read_metadata_table(config, f"{prefix}.dataset")
    if layout == "directory_per_class":
        paths = dataset_paths(config, f"{prefix}.dataset")
        return {
            name: scan_directory_split(path, name, class_names, metadata, identifier_from)
            for name, path in paths.items()
        }
    if layout == "manifest":
        manifest = config.path(f"{prefix}.dataset.manifest.path")
        root = config.path(f"{prefix}.dataset.root")
        split_names = [str(n) for n in config.list_value(f"{prefix}.dataset.manifest.splits")]
        return {
            name: load_manifest_split(
                manifest=manifest,
                split_name=name,
                path_column=config.str_value(f"{prefix}.dataset.manifest.path_column"),
                label_column=config.str_value(f"{prefix}.dataset.manifest.label_column"),
                split_column=config.str_value(f"{prefix}.dataset.manifest.split_column"),
                identifier_column=config.str_value(f"{prefix}.dataset.manifest.identifier_column"),
                root=root,
            )
            for name in split_names
        }
    raise ConfigurationError(
        f"{prefix}.dataset.layout must be 'directory_per_class' or 'manifest'; got {layout!r}"
    )


def verify_split_disjointness(splits: Mapping[str, ImageSplit], identifier_from: str) -> None:
    seen: dict[str, str] = {}
    for split_name, split in splits.items():
        for record in split.records:
            group = group_identifier(record, identifier_from)
            if group in seen and seen[group] != split_name:
                raise ConfigurationError(
                    f"group '{group}' appears in both split '{seen[group]}' and split "
                    f"'{split_name}'. The evaluation protocol requires disjoint training, "
                    "calibration and test splits; repartition the dataset with "
                    "'agp partition' before certification."
                )
            seen[group] = split_name


def build_system(config: Config, name: str, seed: int) -> SystemBundle:
    prefix = f"systems.{name}"
    kind = config.str_value(f"{prefix}.kind")
    device = resolve_device(config)
    register = _action_register(config, prefix)
    domains = _attribute_domains(config, prefix)

    if kind == "image_classification":
        splits = imaging_splits(config, prefix)
        if config.bool_value(f"{prefix}.dataset.enforce_group_disjoint_splits"):
            verify_split_disjointness(
                splits, config.str_value(f"{prefix}.dataset.identifier_from")
            )
        checkpoint = checkpoint_path(config, name, seed, ".pt")
        system = ImageClassificationSystem(
            name=name,
            config=config,
            prefix=prefix,
            checkpoint=checkpoint,
            device=device,
            action_register=register,
            attribute_domains=domains,
        )
        requests = {
            split_name: tuple(system.request_from_record(r) for r in split.records)
            for split_name, split in splits.items()
        }
        labels = {split_name: split.labels() for split_name, split in splits.items()}
        return SystemBundle(
            name=name,
            modality="imaging",
            system=system,
            action_register=register,
            protected_attributes=tuple(system.protected_attributes),
            splits=requests,
            labels=labels,
            checkpoint=checkpoint,
        )

    if kind == "object_detection":
        system = BeadDetectionSystem(
            name=name,
            config=config,
            prefix=prefix,
            device=device,
            action_register=register,
            attribute_domains=domains,
        )
        frames = _detection_frames(config, prefix, system)
        return SystemBundle(
            name=name,
            modality="imaging",
            system=system,
            action_register=register,
            protected_attributes=tuple(system.protected_attributes),
            splits={k: tuple(v[0]) for k, v in frames.items()},
            labels={k: tuple(v[1]) for k, v in frames.items()},
            checkpoint=config.path(f"{prefix}.model.weights"),
        )

    if kind == "tabular_decision":
        schema = TabularSchema.from_config(config, f"{prefix}.dataset")
        frame = load_table(config, f"{prefix}.dataset", schema)
        rng = np.random.default_rng(derive_seed(seed, name, "partition"))
        proportions = {
            str(k): float(v)
            for k, v in dict(config.get(f"{prefix}.dataset.partition")).items()
        }
        partitions = group_aware_partition(
            frame,
            schema,
            proportions,
            rng,
            config.float_value("runtime.proportion_sum_tolerance"),
        )
        encoder = TabularEncoder.fit(partitions["train"], schema)
        spec = GradientBoostingSpec.from_config(config, f"{prefix}.model")
        checkpoint = checkpoint_path(config, name, seed, ".json")
        model = load_model(spec, checkpoint, seed)
        background = background_matrix(
            partitions["calibration"],
            encoder,
            config.int_value(f"{prefix}.model.background_size"),
            np.random.default_rng(derive_seed(seed, name, "background")),
        )
        class_names = tuple(str(n) for n in config.list_value(f"{prefix}.model.class_names"))
        system = TabularDecisionSystem(
            name=name,
            config=config,
            prefix=prefix,
            schema=schema,
            encoder=encoder,
            model=model,
            background=background,
            action_register=register,
            attribute_domains=domains,
            class_names=class_names,
        )
        requests = {
            split_name: tuple(
                system.request_from_row(row) for _, row in partition.iterrows()
            )
            for split_name, partition in partitions.items()
        }
        labels = {
            split_name: tuple(str(v) for v in partition[schema.target_column])
            for split_name, partition in partitions.items()
        }
        return SystemBundle(
            name=name,
            modality="tabular",
            system=system,
            action_register=register,
            protected_attributes=tuple(system.protected_attributes),
            splits=requests,
            labels=labels,
            checkpoint=checkpoint,
        )

    raise ConfigurationError(
        f"{prefix}.kind must be one of "
        + ", ".join(sorted(IMAGING_KINDS | TABULAR_KINDS))
        + f"; got {kind!r}"
    )


def _detection_frames(
    config: Config, prefix: str, system: BeadDetectionSystem
) -> dict[str, tuple[list[Input], list[str]]]:
    import csv

    manifest = config.path(f"{prefix}.dataset.manifest.path")
    root = config.path(f"{prefix}.dataset.root")
    split_column = config.str_value(f"{prefix}.dataset.manifest.split_column")
    path_column = config.str_value(f"{prefix}.dataset.manifest.path_column")
    label_column = config.str_value(f"{prefix}.dataset.manifest.label_column")
    identifier_column = config.str_value(f"{prefix}.dataset.manifest.identifier_column")
    split_names = [str(n) for n in config.list_value(f"{prefix}.dataset.manifest.splits")]

    frames: dict[str, tuple[list[Input], list[str]]] = {name: ([], []) for name in split_names}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {split_column, path_column, label_column, identifier_column}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ConfigurationError(
                f"detection manifest {manifest} lacks columns: " + ", ".join(sorted(missing))
            )
        for row in reader:
            split_name = str(row[split_column])
            if split_name not in frames:
                continue
            image_path = Path(str(row[path_column]))
            if not image_path.is_absolute():
                image_path = root / image_path
            frames[split_name][0].append(
                system.request_from_path(
                    identifier=str(row[identifier_column]),
                    path=image_path,
                    metadata={k: v for k, v in row.items() if k != path_column},
                )
            )
            frames[split_name][1].append(str(row[label_column]))
    for split_name, (requests, _) in frames.items():
        if not requests:
            raise ConfigurationError(
                f"detection manifest {manifest} holds no rows for split '{split_name}'"
            )
    return frames


def build_all_systems(config: Config, seed: int) -> dict[str, SystemBundle]:
    names = config.keys("systems")
    return {name: build_system(config, name, seed) for name in names}
