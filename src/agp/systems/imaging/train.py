from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ...errors import ConfigurationError
from .backbone import BackboneSpec, build_backbone
from .dataset import ImageRecord, ImageSplit, load_image, normalise


class RecordDataset:
    def __init__(
        self,
        records: Sequence[ImageRecord],
        classes: Sequence[str],
        image_size: int,
        channels: int,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        self.records = list(records)
        self.class_to_index = {name: index for index, name in enumerate(classes)}
        missing = {r.label for r in self.records} - set(self.class_to_index)
        if missing:
            raise ConfigurationError(
                "the split contains labels absent from class_names: " + ", ".join(sorted(missing))
            )
        self.image_size = image_size
        self.channels = channels
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch

        record = self.records[index]
        pixels = load_image(record.path, self.image_size, self.channels)
        normalised = normalise(pixels[None, ...], self.mean, self.std)[0]
        return (
            torch.from_numpy(normalised).float(),
            torch.tensor(self.class_to_index[record.label], dtype=torch.long),
        )


@dataclass(frozen=True)
class TrainingOutcome:
    checkpoint: Path
    epochs_run: int
    best_validation_loss: float
    best_validation_accuracy: float
    history: tuple[dict[str, float], ...]


def train_image_classifier(
    spec: BackboneSpec,
    classes: Sequence[str],
    train_split: ImageSplit,
    validation_split: ImageSplit,
    channels: int,
    mean: Sequence[float],
    std: Sequence[float],
    checkpoint: Path,
    seed: int,
    num_workers: int,
    label_overrides: dict[str, str] | None = None,
) -> TrainingOutcome:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    train_records = list(train_split.records)
    if label_overrides:
        train_records = [
            ImageRecord(
                identifier=record.identifier,
                path=record.path,
                label=label_overrides.get(record.identifier, record.label),
                metadata=record.metadata,
            )
            for record in train_records
        ]

    train_dataset = RecordDataset(train_records, classes, spec.image_size, channels, mean, std)
    validation_dataset = RecordDataset(
        validation_split.records, classes, spec.image_size, channels, mean, std
    )

    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=spec.batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=spec.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    network = build_backbone(spec)
    optimiser = torch.optim.Adam(network.parameters(), lr=spec.learning_rate)
    criterion = nn.CrossEntropyLoss()
    device = torch.device(spec.device)

    best_loss = float("inf")
    best_accuracy = 0.0
    patience = 0
    history: list[dict[str, float]] = []
    epochs_run = 0

    for epoch in range(spec.max_epochs):
        network.train()
        running_loss = 0.0
        seen = 0
        for batch, targets in train_loader:
            batch = batch.to(device)
            targets = targets.to(device)
            optimiser.zero_grad()
            logits = network(batch)
            loss = criterion(logits, targets)
            loss.backward()
            optimiser.step()
            running_loss += float(loss.item()) * batch.size(0)
            seen += int(batch.size(0))
        train_loss = running_loss / max(seen, 1)

        network.eval()
        validation_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch, targets in validation_loader:
                batch = batch.to(device)
                targets = targets.to(device)
                logits = network(batch)
                validation_loss += float(criterion(logits, targets).item()) * batch.size(0)
                correct += int((logits.argmax(dim=-1) == targets).sum().item())
                total += int(batch.size(0))
        validation_loss /= max(total, 1)
        accuracy = correct / max(total, 1)
        epochs_run = epoch + 1
        history.append(
            {
                "epoch": float(epochs_run),
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_accuracy": accuracy,
            }
        )

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_accuracy = accuracy
            patience = 0
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state": network.state_dict(),
                    "classes": list(classes),
                    "architecture": spec.architecture,
                    "image_size": spec.image_size,
                    "seed": seed,
                    "validation_loss": validation_loss,
                    "validation_accuracy": accuracy,
                },
                checkpoint,
            )
        else:
            patience += 1
            if patience >= spec.early_stopping_patience:
                break

    if best_loss == float("inf"):
        raise ConfigurationError(
            "training completed without a validation improvement; no checkpoint was written"
        )

    return TrainingOutcome(
        checkpoint=checkpoint,
        epochs_run=epochs_run,
        best_validation_loss=best_loss,
        best_validation_accuracy=best_accuracy,
        history=tuple(history),
    )


def stratified_group_partition(
    split: ImageSplit,
    group_of: dict[str, str],
    proportions: dict[str, float],
    rng: np.random.Generator,
    tolerance: float,
) -> dict[str, ImageSplit]:
    total = sum(proportions.values())
    if abs(total - 1.0) > tolerance:
        raise ConfigurationError(
            f"partition proportions must sum to 1.0 within {tolerance}; got {total:.9f}"
        )
    groups: dict[str, list[ImageRecord]] = {}
    for record in split.records:
        groups.setdefault(group_of[record.identifier], []).append(record)

    group_names = sorted(groups)
    rng.shuffle(group_names)
    assignments: dict[str, list[ImageRecord]] = {name: [] for name in proportions}
    targets = {name: proportion * len(split.records) for name, proportion in proportions.items()}
    counts = {name: 0 for name in proportions}
    for group_name in group_names:
        deficits = {
            name: targets[name] - counts[name]
            for name in proportions
        }
        chosen = max(deficits, key=lambda n: deficits[n])
        assignments[chosen].extend(groups[group_name])
        counts[chosen] += len(groups[group_name])

    for name, records in assignments.items():
        if not records:
            raise ConfigurationError(
                f"partition '{name}' received no records; adjust the configured proportions"
            )
    return {
        name: ImageSplit(name=name, records=tuple(records))
        for name, records in assignments.items()
    }
