from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...config import Config
from ...errors import ConfigurationError, MissingResourceError

IMAGE_SUFFIXES = (".jpeg", ".jpg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class ImageRecord:
    identifier: str
    path: Path
    label: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ImageSplit:
    name: str
    records: tuple[ImageRecord, ...]

    def labels(self) -> tuple[str, ...]:
        return tuple(record.label for record in self.records)

    def __len__(self) -> int:
        return len(self.records)


def _read_metadata(path: Path, key_column: str) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise MissingResourceError(
            f"the metadata table for the protected attributes was not found at {path}. "
            "The non-discrimination clause requires documented attributes from the real dataset."
        )
    table: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or key_column not in reader.fieldnames:
            raise ConfigurationError(
                f"metadata table {path} has no column named '{key_column}'"
            )
        for row in reader:
            table[str(row[key_column])] = dict(row)
    if not table:
        raise MissingResourceError(f"metadata table {path} holds no rows")
    return table


def scan_directory_split(
    root: Path,
    split_name: str,
    class_names: Sequence[str],
    metadata: Mapping[str, Mapping[str, Any]],
    identifier_from: str,
) -> ImageSplit:
    if not root.is_dir():
        raise MissingResourceError(
            f"split directory '{split_name}' was not found at {root}. Configure the real "
            "dataset location; no images are generated."
        )
    records: list[ImageRecord] = []
    for class_name in class_names:
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise MissingResourceError(
                f"class directory '{class_name}' is absent from split '{split_name}' at {root}"
            )
        for file_path in sorted(class_dir.iterdir()):
            if file_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            identifier = _identifier_for(file_path, identifier_from)
            records.append(
                ImageRecord(
                    identifier=file_path.stem,
                    path=file_path,
                    label=class_name,
                    metadata=dict(metadata.get(identifier, {})),
                )
            )
    if not records:
        raise MissingResourceError(
            f"split '{split_name}' at {root} contains no images with a recognised suffix"
        )
    return ImageSplit(name=split_name, records=tuple(records))


def _identifier_for(path: Path, identifier_from: str) -> str:
    stem = path.stem
    if identifier_from == "stem":
        return stem
    if identifier_from == "patient_field":
        parts = stem.split("-")
        if len(parts) < 2:
            raise ConfigurationError(
                f"image '{path.name}' does not encode a patient identifier in "
                "'<class>-<patient>-<scan>' form"
            )
        return parts[-2]
    raise ConfigurationError(
        f"unknown identifier_from strategy '{identifier_from}'; "
        "valid strategies are 'stem' and 'patient_field'"
    )


def group_identifier(record: ImageRecord, strategy: str) -> str:
    return _identifier_for(record.path, strategy)


def load_manifest_split(
    manifest: Path,
    split_name: str,
    path_column: str,
    label_column: str,
    split_column: str,
    identifier_column: str,
    root: Path,
) -> ImageSplit:
    if not manifest.is_file():
        raise MissingResourceError(f"image manifest not found at {manifest}")
    records: list[ImageRecord] = []
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {path_column, label_column, split_column, identifier_column}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ConfigurationError(
                f"manifest {manifest} lacks required columns: " + ", ".join(sorted(missing))
            )
        for row in reader:
            if str(row[split_column]) != split_name:
                continue
            image_path = Path(str(row[path_column]))
            if not image_path.is_absolute():
                image_path = root / image_path
            records.append(
                ImageRecord(
                    identifier=str(row[identifier_column]),
                    path=image_path,
                    label=str(row[label_column]),
                    metadata={k: v for k, v in row.items() if k != path_column},
                )
            )
    if not records:
        raise MissingResourceError(
            f"manifest {manifest} holds no rows for split '{split_name}'"
        )
    return ImageSplit(name=split_name, records=tuple(records))


def load_image(path: Path, image_size: int, channels: int) -> np.ndarray:
    from PIL import Image

    if not path.is_file():
        raise MissingResourceError(f"image file not found: {path}")
    mode = "L" if channels == 1 else "RGB"
    with Image.open(path) as handle:
        converted = handle.convert(mode).resize((image_size, image_size), Image.BILINEAR)
        array = np.asarray(converted, dtype=np.float32) / float(np.iinfo(np.uint8).max)
    if channels == 1:
        array = array[None, ...]
    else:
        array = np.transpose(array, (2, 0, 1))
    return np.ascontiguousarray(array)


def normalise(batch: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1)
    if batch.shape[1] != mean_array.shape[1]:
        raise ConfigurationError(
            f"normalisation statistics declare {mean_array.shape[1]} channels but the image "
            f"batch carries {batch.shape[1]}"
        )
    return (batch - mean_array) / std_array


def dataset_paths(config: Config, prefix: str) -> dict[str, Path]:
    root = config.path(f"{prefix}.root")
    splits = config.get(f"{prefix}.splits")
    if not isinstance(splits, Mapping):
        raise ConfigurationError(f"{prefix}.splits must be a mapping of split name to directory")
    resolved: dict[str, Path] = {}
    for name, relative in splits.items():
        candidate = Path(str(relative))
        resolved[str(name)] = candidate if candidate.is_absolute() else root / candidate
    return resolved


def read_metadata_table(config: Config, prefix: str) -> dict[str, dict[str, Any]]:
    path = config.path(f"{prefix}.metadata.path")
    key_column = config.str_value(f"{prefix}.metadata.key_column")
    return _read_metadata(path, key_column)
