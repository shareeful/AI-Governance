from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ...config import Config
from ...errors import ConfigurationError, MissingResourceError


@dataclass(frozen=True)
class BackboneSpec:
    architecture: str
    num_classes: int
    image_size: int
    dense_units: int
    dropout: float
    pretrained: bool
    learning_rate: float
    max_epochs: int
    early_stopping_patience: int
    batch_size: int
    device: str

    @classmethod
    def from_config(cls, config: Config, prefix: str, device: str) -> "BackboneSpec":
        return cls(
            architecture=config.str_value(f"{prefix}.architecture"),
            num_classes=config.int_value(f"{prefix}.num_classes"),
            image_size=config.int_value(f"{prefix}.image_size"),
            dense_units=config.int_value(f"{prefix}.dense_units"),
            dropout=config.require_unit_interval(f"{prefix}.dropout"),
            pretrained=config.bool_value(f"{prefix}.pretrained"),
            learning_rate=config.float_value(f"{prefix}.learning_rate"),
            max_epochs=config.int_value(f"{prefix}.max_epochs"),
            early_stopping_patience=config.int_value(f"{prefix}.early_stopping_patience"),
            batch_size=config.int_value(f"{prefix}.batch_size"),
            device=device,
        )


def build_backbone(spec: BackboneSpec):
    import torch
    from torch import nn
    from torchvision import models

    factory = getattr(models, spec.architecture, None)
    if factory is None:
        raise ConfigurationError(
            f"torchvision exposes no architecture named '{spec.architecture}'"
        )
    weights = "DEFAULT" if spec.pretrained else None
    network = factory(weights=weights)

    if hasattr(network, "classifier") and isinstance(network.classifier, nn.Linear):
        in_features = network.classifier.in_features
        network.classifier = nn.Sequential(
            nn.Linear(in_features, spec.dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.dense_units, spec.num_classes),
        )
    elif hasattr(network, "fc") and isinstance(network.fc, nn.Linear):
        in_features = network.fc.in_features
        network.fc = nn.Sequential(
            nn.Linear(in_features, spec.dense_units),
            nn.ReLU(inplace=True),
            nn.Dropout(spec.dropout),
            nn.Linear(spec.dense_units, spec.num_classes),
        )
    else:
        raise ConfigurationError(
            f"architecture '{spec.architecture}' exposes neither a 'classifier' nor an 'fc' head "
            "that can be replaced"
        )
    return network.to(torch.device(spec.device))


def load_backbone(spec: BackboneSpec, checkpoint: Path):
    import torch

    if not checkpoint.is_file():
        raise MissingResourceError(
            f"no trained checkpoint at {checkpoint}; train the governed model with "
            "'agp train' before certification or enforcement"
        )
    network = build_backbone(spec)
    state = torch.load(checkpoint, map_location=spec.device)
    network.load_state_dict(state["model_state"])
    network.eval()
    return network


def softmax_scores(network, batch: np.ndarray, device: str) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(batch)).float().to(device)
    with torch.no_grad():
        logits = network(tensor)
        return torch.softmax(logits, dim=-1).cpu().numpy()


def class_names(config: Config, prefix: str) -> tuple[str, ...]:
    names = config.list_value(f"{prefix}.class_names")
    return tuple(str(name) for name in names)


def region_grid(image_size: int, rows: int, columns: int) -> tuple[tuple[int, int, int, int], ...]:
    if rows < 1 or columns < 1:
        raise ConfigurationError("the attribution region grid must have at least one row and column")
    row_edges = np.linspace(0, image_size, rows + 1).astype(int)
    column_edges = np.linspace(0, image_size, columns + 1).astype(int)
    regions: list[tuple[int, int, int, int]] = []
    for r in range(rows):
        for c in range(columns):
            regions.append(
                (int(row_edges[r]), int(row_edges[r + 1]), int(column_edges[c]), int(column_edges[c + 1]))
            )
    return tuple(regions)


def region_names(rows: int, columns: int) -> tuple[str, ...]:
    return tuple(f"region_r{r}c{c}" for r in range(rows) for c in range(columns))


def integrated_gradients_regions(
    network,
    image: np.ndarray,
    target: int,
    regions: Sequence[tuple[int, int, int, int]],
    steps: int,
    device: str,
) -> np.ndarray:
    import torch
    from captum.attr import IntegratedGradients

    tensor = torch.from_numpy(np.ascontiguousarray(image[None, ...])).float().to(device)
    baseline = torch.zeros_like(tensor)
    explainer = IntegratedGradients(network)
    attributions = explainer.attribute(tensor, baselines=baseline, target=target, n_steps=steps)
    saliency = attributions.abs().sum(dim=1)[0].cpu().numpy()
    return np.array(
        [float(saliency[r0:r1, c0:c1].sum()) for r0, r1, c0, c1 in regions], dtype=float
    )


def occlusion_regions(
    network,
    image: np.ndarray,
    target: int,
    regions: Sequence[tuple[int, int, int, int]],
    fill_value: float,
    device: str,
) -> np.ndarray:
    baseline = softmax_scores(network, image[None, ...], device)[0, target]
    masses: list[float] = []
    for r0, r1, c0, c1 in regions:
        occluded = image.copy()
        occluded[:, r0:r1, c0:c1] = fill_value
        score = softmax_scores(network, occluded[None, ...], device)[0, target]
        masses.append(float(max(baseline - score, 0.0)))
    return np.array(masses, dtype=float)
