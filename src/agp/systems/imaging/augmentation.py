from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ...config import Config
from ...errors import ConfigurationError
from .dataset import ImageRecord, ImageSplit, load_image


@dataclass(frozen=True)
class AugmentationSpec:
    views_per_image: int
    max_rotation_degrees: float
    horizontal_flip: bool
    vertical_flip: bool
    brightness_delta: float
    contrast_delta: float
    noise_standard_deviation: float
    motion_blur_max_kernel: int
    occlusion_count: int
    occlusion_max_fraction: float
    occlusion_fill: float
    output_format: str
    retain_originals: bool

    @classmethod
    def from_config(cls, config: Config, prefix: str) -> "AugmentationSpec":
        spec = cls(
            views_per_image=config.int_value(f"{prefix}.views_per_image"),
            max_rotation_degrees=config.float_value(f"{prefix}.max_rotation_degrees"),
            horizontal_flip=config.bool_value(f"{prefix}.horizontal_flip"),
            vertical_flip=config.bool_value(f"{prefix}.vertical_flip"),
            brightness_delta=config.require_unit_interval(f"{prefix}.brightness_delta"),
            contrast_delta=config.require_unit_interval(f"{prefix}.contrast_delta"),
            noise_standard_deviation=config.require_unit_interval(
                f"{prefix}.noise_standard_deviation"
            ),
            motion_blur_max_kernel=config.int_value(f"{prefix}.motion_blur_max_kernel"),
            occlusion_count=config.int_value(f"{prefix}.occlusion_count"),
            occlusion_max_fraction=config.require_unit_interval(
                f"{prefix}.occlusion_max_fraction"
            ),
            occlusion_fill=config.require_unit_interval(f"{prefix}.occlusion_fill"),
            output_format=config.str_value(f"{prefix}.output_format"),
            retain_originals=config.bool_value(f"{prefix}.retain_originals"),
        )
        if spec.views_per_image < 1:
            raise ConfigurationError(f"{prefix}.views_per_image must be at least 1")
        if spec.motion_blur_max_kernel < 1:
            raise ConfigurationError(f"{prefix}.motion_blur_max_kernel must be at least 1")
        if spec.output_format.lower() not in {"png", "tiff"}:
            raise ConfigurationError(
                f"{prefix}.output_format must be a lossless format, 'png' or 'tiff'; "
                f"got {spec.output_format!r}"
            )
        return spec


def rotate_view(pixels: np.ndarray, degrees: float) -> np.ndarray:
    from scipy.ndimage import rotate

    if abs(degrees) < 1e-9:
        return pixels
    return rotate(pixels, degrees, axes=(1, 2), reshape=False, order=1, mode="nearest")


def flip_view(pixels: np.ndarray, horizontal: bool, vertical: bool) -> np.ndarray:
    result = pixels
    if horizontal:
        result = result[:, :, ::-1]
    if vertical:
        result = result[:, ::-1, :]
    return np.ascontiguousarray(result)


def jitter_view(pixels: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    mean = float(pixels.mean())
    adjusted = (pixels - mean) * (1.0 + contrast) + mean + brightness
    return np.clip(adjusted, 0.0, 1.0)


def noise_view(pixels: np.ndarray, deviation: float, rng: np.random.Generator) -> np.ndarray:
    if deviation <= 0.0:
        return pixels
    return np.clip(pixels + rng.normal(0.0, deviation, pixels.shape), 0.0, 1.0)


def motion_blur_view(pixels: np.ndarray, kernel_length: int, angle: float) -> np.ndarray:
    from scipy.ndimage import convolve

    if kernel_length <= 1:
        return pixels
    kernel = np.zeros((kernel_length, kernel_length), dtype=float)
    kernel[kernel_length // 2, :] = 1.0
    kernel = rotate_view(kernel[None, ...], angle)[0]
    total = kernel.sum()
    if total <= 0.0:
        return pixels
    kernel = kernel / total
    blurred = np.stack([convolve(channel, kernel, mode="nearest") for channel in pixels])
    return np.clip(blurred, 0.0, 1.0)


def occlude_view(
    pixels: np.ndarray,
    count: int,
    max_fraction: float,
    fill: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0 or max_fraction <= 0.0:
        return pixels
    occluded = np.array(pixels, copy=True)
    height, width = occluded.shape[1], occluded.shape[2]
    for _ in range(count):
        patch_height = max(1, int(round(height * float(rng.uniform(0.0, max_fraction)))))
        patch_width = max(1, int(round(width * float(rng.uniform(0.0, max_fraction)))))
        row = int(rng.integers(0, max(1, height - patch_height)))
        column = int(rng.integers(0, max(1, width - patch_width)))
        occluded[:, row : row + patch_height, column : column + patch_width] = fill
    return occluded


def build_view(
    pixels: np.ndarray,
    spec: AugmentationSpec,
    rng: np.random.Generator,
) -> np.ndarray:
    view = rotate_view(pixels, float(rng.uniform(-spec.max_rotation_degrees, spec.max_rotation_degrees)))
    view = flip_view(
        view,
        horizontal=spec.horizontal_flip and bool(rng.integers(0, 2)),
        vertical=spec.vertical_flip and bool(rng.integers(0, 2)),
    )
    view = jitter_view(
        view,
        brightness=float(rng.uniform(-spec.brightness_delta, spec.brightness_delta)),
        contrast=float(rng.uniform(-spec.contrast_delta, spec.contrast_delta)),
    )
    view = noise_view(view, spec.noise_standard_deviation, rng)
    view = motion_blur_view(
        view,
        kernel_length=int(rng.integers(1, spec.motion_blur_max_kernel + 1)),
        angle=float(rng.uniform(0.0, 180.0)),
    )
    return occlude_view(
        view, spec.occlusion_count, spec.occlusion_max_fraction, spec.occlusion_fill, rng
    )


def write_view(pixels: np.ndarray, path: Path) -> None:
    from PIL import Image

    array = np.transpose(pixels, (1, 2, 0))
    scale = float(np.iinfo(np.uint8).max)
    array = np.clip(array * scale, 0.0, scale).astype(np.uint8)
    if array.shape[2] == 1:
        array = array[:, :, 0]
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def augment_training_split(
    split: ImageSplit,
    held_out_splits: Sequence[ImageSplit],
    spec: AugmentationSpec,
    image_size: int,
    channels: int,
    output_root: Path,
    rng: np.random.Generator,
) -> ImageSplit:
    held_out_identifiers = {
        record.identifier for other in held_out_splits for record in other.records
    }
    overlap = {record.identifier for record in split.records} & held_out_identifiers
    if overlap:
        raise ConfigurationError(
            "augmentation is leakage-safe only when every original image held out for "
            "validation or test is absent from the training split; these identifiers appear in "
            "both: " + ", ".join(sorted(overlap)[:10])
        )

    records: list[ImageRecord] = list(split.records) if spec.retain_originals else []
    extension = spec.output_format.lower()
    for record in split.records:
        pixels = load_image(record.path, image_size, channels)
        for view_index in range(spec.views_per_image):
            view = build_view(pixels, spec, rng)
            identifier = f"{record.identifier}_view{view_index:02d}"
            target = output_root / record.label / f"{identifier}.{extension}"
            write_view(view, target)
            records.append(
                ImageRecord(
                    identifier=identifier,
                    path=target,
                    label=record.label,
                    metadata=dict(record.metadata),
                )
            )
    return ImageSplit(name=split.name, records=tuple(records))
