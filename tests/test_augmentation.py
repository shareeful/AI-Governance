from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from agp.errors import ConfigurationError
from agp.systems.imaging.augmentation import (
    AugmentationSpec,
    augment_training_split,
    build_view,
    jitter_view,
    motion_blur_view,
    occlude_view,
)
from agp.systems.imaging.dataset import ImageRecord, ImageSplit

SPEC = AugmentationSpec(
    views_per_image=16,
    max_rotation_degrees=180.0,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_delta=0.2,
    contrast_delta=0.2,
    noise_standard_deviation=0.02,
    motion_blur_max_kernel=5,
    occlusion_count=2,
    occlusion_max_fraction=0.15,
    occlusion_fill=0.0,
    output_format="png",
    retain_originals=False,
)


def _write_image(path, size=32):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(path.name)) % (2**32))
    array = (rng.uniform(0.2, 0.8, (size, size)) * 255).astype(np.uint8)
    Image.fromarray(array).save(path)
    return path


def _split(tmp_path, name, identifiers, label="bead"):
    records = tuple(
        ImageRecord(
            identifier=i,
            path=_write_image(tmp_path / name / label / f"{i}.png"),
            label=label,
            metadata={},
        )
        for i in identifiers
    )
    return ImageSplit(name=name, records=records)


def test_every_view_stays_inside_the_unit_interval():
    rng = np.random.default_rng(0)
    pixels = rng.uniform(0.0, 1.0, (1, 24, 24)).astype(np.float32)
    for _ in range(20):
        view = build_view(pixels, SPEC, rng)
        assert view.shape == pixels.shape
        assert view.min() >= 0.0 and view.max() <= 1.0


def test_jitter_preserves_shape_and_clips():
    pixels = np.full((1, 8, 8), 0.95, dtype=np.float32)
    assert jitter_view(pixels, brightness=0.5, contrast=0.5).max() <= 1.0
    assert jitter_view(pixels, brightness=-2.0, contrast=0.0).min() >= 0.0


def test_motion_blur_reduces_local_variance():
    rng = np.random.default_rng(1)
    pixels = rng.uniform(0.0, 1.0, (1, 32, 32))
    blurred = motion_blur_view(pixels, kernel_length=5, angle=0.0)
    assert blurred.var() < pixels.var()


def test_motion_blur_of_unit_kernel_is_the_identity():
    pixels = np.random.default_rng(2).uniform(0.0, 1.0, (1, 8, 8))
    assert np.allclose(motion_blur_view(pixels, kernel_length=1, angle=0.0), pixels)


def test_occlusion_writes_the_configured_fill_value():
    pixels = np.full((1, 32, 32), 0.7)
    occluded = occlude_view(
        pixels, count=3, max_fraction=0.5, fill=0.0, rng=np.random.default_rng(3)
    )
    assert (occluded == 0.0).any()
    assert (occluded == 0.7).any()


def test_augmentation_produces_exactly_views_per_image(tmp_path):
    train = _split(tmp_path, "train", ["a", "b", "c"])
    test = _split(tmp_path, "test", ["x", "y"])
    augmented = augment_training_split(
        split=train,
        held_out_splits=[test],
        spec=SPEC,
        image_size=32,
        channels=1,
        output_root=tmp_path / "out",
        rng=np.random.default_rng(4),
    )
    assert len(augmented) == len(train) * SPEC.views_per_image
    assert all("_view" in r.identifier for r in augmented.records)


def test_retaining_originals_adds_them_on_top(tmp_path):
    train = _split(tmp_path, "train", ["a", "b"])
    test = _split(tmp_path, "test", ["x"])
    spec = AugmentationSpec(**{**SPEC.__dict__, "views_per_image": 4, "retain_originals": True})
    augmented = augment_training_split(
        split=train,
        held_out_splits=[test],
        spec=spec,
        image_size=32,
        channels=1,
        output_root=tmp_path / "out",
        rng=np.random.default_rng(5),
    )
    assert len(augmented) == len(train) + len(train) * 4


def test_augmentation_refuses_a_training_image_that_is_also_held_out(tmp_path):
    train = _split(tmp_path, "train", ["a", "shared"])
    test = _split(tmp_path, "test", ["shared"])
    with pytest.raises(ConfigurationError, match="leakage-safe"):
        augment_training_split(
            split=train,
            held_out_splits=[test],
            spec=SPEC,
            image_size=32,
            channels=1,
            output_root=tmp_path / "out",
            rng=np.random.default_rng(6),
        )


def test_augmentation_labels_are_preserved(tmp_path):
    train = _split(tmp_path, "train", ["a"], label="bead")
    test = _split(tmp_path, "test", ["x"], label="bead")
    augmented = augment_training_split(
        split=train,
        held_out_splits=[test],
        spec=AugmentationSpec(**{**SPEC.__dict__, "views_per_image": 3}),
        image_size=32,
        channels=1,
        output_root=tmp_path / "out",
        rng=np.random.default_rng(7),
    )
    assert {r.label for r in augmented.records} == {"bead"}
    assert all(r.path.is_file() for r in augmented.records)


def test_lossy_output_format_is_refused(tmp_path):
    import yaml

    from agp.config import Config

    payload = {
        "aug": {
            **{k: v for k, v in SPEC.__dict__.items() if k != "output_format"},
            "output_format": "jpeg",
        }
    }
    path = tmp_path / "aug.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="lossless"):
        AugmentationSpec.from_config(Config.load(path), "aug")
