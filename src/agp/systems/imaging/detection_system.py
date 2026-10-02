from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...checkers.scope import ActionRegister
from ...config import Config
from ...errors import InterfaceError, MissingResourceError
from ...execution import Input
from ..base import InstrumentedSystem, RationaleTemplates
from .backbone import region_grid, region_names
from .dataset import load_image


@dataclass(frozen=True)
class DetectionSpec:
    weights: Path
    image_size: int
    confidence_threshold: float
    iou_threshold: float
    device: str
    accept_action: str
    reject_action: str
    max_defects: int

    @classmethod
    def from_config(cls, config: Config, prefix: str, device: str) -> "DetectionSpec":
        weights = config.path(f"{prefix}.model.weights")
        return cls(
            weights=weights,
            image_size=config.int_value(f"{prefix}.model.image_size"),
            confidence_threshold=config.require_unit_interval(
                f"{prefix}.model.confidence_threshold"
            ),
            iou_threshold=config.require_unit_interval(f"{prefix}.model.iou_threshold"),
            device=device,
            accept_action=config.str_value(f"{prefix}.decision.accept_action"),
            reject_action=config.str_value(f"{prefix}.decision.reject_action"),
            max_defects=config.int_value(f"{prefix}.decision.max_detections_for_accept"),
        )


class BeadDetectionSystem(InstrumentedSystem):
    def __init__(
        self,
        name: str,
        config: Config,
        prefix: str,
        device: str,
        action_register: ActionRegister,
        attribute_domains: Mapping[str, Sequence[Any]],
    ) -> None:
        self.spec = DetectionSpec.from_config(config, prefix, device)
        self.channels = config.int_value(f"{prefix}.model.channels")
        rows = config.int_value(f"{prefix}.attribution.grid_rows")
        columns = config.int_value(f"{prefix}.attribution.grid_columns")
        self.grid_rows = rows
        self.grid_columns = columns
        self.regions = region_grid(self.spec.image_size, rows, columns)
        self.occlusion_fill = config.float_value(f"{prefix}.attribution.occlusion_fill")
        self.paraphrase_max_rotation = config.float_value(
            f"{prefix}.paraphrase.max_rotation_degrees"
        )
        self.paraphrase_max_brightness = config.require_unit_interval(
            f"{prefix}.paraphrase.max_brightness_delta"
        )
        self.paraphrase_horizontal_flip = config.bool_value(f"{prefix}.paraphrase.horizontal_flip")
        self.detector = self._load_detector()
        super().__init__(
            name=name,
            modality="imaging",
            action_register=action_register,
            feature_groups=region_names(rows, columns),
            protected_attributes=tuple(config.list_value(f"{prefix}.protected_attributes")),
            attribute_domains=attribute_domains,
            minimum_necessary=frozenset(
                str(item) for item in config.list_value(f"{prefix}.minimum_necessary_personal_data")
            ),
            templates=RationaleTemplates.from_config(config, f"{prefix}.templates"),
            reason_count=config.int_value(f"{prefix}.reason_count"),
            recorded_check_policy=config.str_value(f"{prefix}.preconditions.recorded_check_policy"),
        )

    def _load_detector(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise MissingResourceError(
                "the bead-detection pilot requires the 'ultralytics' extra; install it with "
                "'pip install ai-governance-pipeline[detection]'"
            ) from exc
        if not self.spec.weights.is_file():
            raise MissingResourceError(
                f"no detector weights at {self.spec.weights}; train the pilot detector before "
                "certification"
            )
        return YOLO(str(self.spec.weights))

    def _detections(self, request: Input) -> np.ndarray:
        pixels = np.asarray(request.features, dtype=np.float32)
        image = np.transpose(pixels, (1, 2, 0))
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        results = self.detector.predict(
            source=(image * float(np.iinfo(np.uint8).max)).astype(np.uint8),
            imgsz=self.spec.image_size,
            conf=self.spec.confidence_threshold,
            iou=self.spec.iou_threshold,
            device=self.spec.device,
            verbose=False,
        )
        if not results:
            return np.zeros((0, 5), dtype=float)
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            return np.zeros((0, 5), dtype=float)
        coordinates = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy().reshape(-1, 1)
        return np.hstack([coordinates, confidences]).astype(float)

    def decision_scores(self, request: Input) -> np.ndarray:
        detections = self._detections(request)
        if detections.size == 0:
            return np.array([1.0, 0.0], dtype=float)
        peak = float(np.max(detections[:, 4]))
        return np.array([1.0 - peak, peak], dtype=float)

    def predict_action(self, request: Input) -> str:
        detections = self._detections(request)
        if len(detections) > self.spec.max_defects:
            return self.spec.reject_action
        return self.spec.accept_action

    def predict_action_with_groups_removed(
        self, request: Input, removed_groups: frozenset[str]
    ) -> str:
        if not removed_groups:
            return self.predict_action(request)
        pixels = np.array(request.features, dtype=np.float32, copy=True)
        index_by_name = {name: position for position, name in enumerate(self.feature_groups)}
        for group in removed_groups:
            if group not in index_by_name:
                raise InterfaceError(f"unknown feature group '{group}' for system '{self.name}'")
            r0, r1, c0, c1 = self.regions[index_by_name[group]]
            pixels[:, r0:r1, c0:c1] = self.occlusion_fill
        return self.predict_action(replace(request, features=pixels))

    def attribution(self, request: Input) -> Mapping[str, float]:
        detections = self._detections(request)
        masses = np.zeros(len(self.regions), dtype=float)
        for x0, y0, x1, y1, confidence in detections:
            centre_x = 0.5 * (x0 + x1)
            centre_y = 0.5 * (y0 + y1)
            for index, (r0, r1, c0, c1) in enumerate(self.regions):
                if r0 <= centre_y < r1 and c0 <= centre_x < c1:
                    masses[index] += float(confidence)
                    break
        if masses.sum() <= 0.0:
            uniform = np.full(len(self.regions), 1.0 / len(self.regions), dtype=float)
            return {name: float(v) for name, v in zip(self.feature_groups, uniform)}
        return {name: float(v) for name, v in zip(self.feature_groups, masses)}

    def substitute_attribute(self, request: Input, attribute: str, value: Any) -> Input:
        if attribute not in self.protected_attributes:
            raise InterfaceError(
                f"'{attribute}' is not a documented protected attribute of system '{self.name}'"
            )
        if attribute != "field_of_view_position":
            metadata = dict(request.metadata)
            metadata[attribute] = value
            return replace(request, metadata=metadata)
        return self._translate_to_position(request, str(value))

    def _translate_to_position(self, request: Input, position: str) -> Input:
        pixels = np.asarray(request.features, dtype=np.float32)
        rows, columns = pixels.shape[1], pixels.shape[2]
        try:
            target_row, target_column = (int(part) for part in position.split(","))
        except ValueError as exc:
            raise InterfaceError(
                "field-of-view positions must be given as 'row,column' grid indices"
            ) from exc
        centre_row = (self.grid_rows - 1) / 2.0
        centre_column = (self.grid_columns - 1) / 2.0
        shift_rows = int(round((target_row - centre_row) * rows / self.grid_rows))
        shift_columns = int(round((target_column - centre_column) * columns / self.grid_columns))
        translated = np.roll(pixels, shift=(shift_rows, shift_columns), axis=(1, 2))
        metadata = dict(request.metadata)
        metadata["field_of_view_position"] = position
        return replace(request, features=translated, metadata=metadata)

    def perturbations(self, request: Input, count: int, rng: np.random.Generator) -> list[Input]:
        if count < 1:
            raise InterfaceError("the STRIP signal requires at least one perturbation")
        pixels = np.asarray(request.features, dtype=np.float32)
        outputs: list[Input] = []
        for _ in range(count):
            overlay = rng.permutation(pixels.reshape(pixels.shape[0], -1).T).T.reshape(pixels.shape)
            blended = np.clip(0.5 * pixels + 0.5 * overlay, 0.0, 1.0).astype(np.float32)
            outputs.append(replace(request, features=blended))
        return outputs

    def paraphrases(self, request: Input, count: int, rng: np.random.Generator) -> list[Input]:
        from scipy.ndimage import rotate

        pixels = np.asarray(request.features, dtype=np.float32)
        outputs: list[Input] = []
        for index in range(count):
            variant = pixels
            angle = float(rng.uniform(-self.paraphrase_max_rotation, self.paraphrase_max_rotation))
            if abs(angle) > 0.0:
                variant = rotate(variant, angle, axes=(1, 2), reshape=False, order=1, mode="nearest")
            delta = float(
                rng.uniform(-self.paraphrase_max_brightness, self.paraphrase_max_brightness)
            )
            variant = variant + delta
            if self.paraphrase_horizontal_flip and index % 2 == 1:
                variant = variant[:, :, ::-1]
            outputs.append(replace(request, features=np.clip(variant, 0.0, 1.0).astype(np.float32)))
        return outputs

    def recorded_checks(self, request: Input, action: str) -> tuple[str, ...]:
        required = sorted(self.action_register.required_checks(action))
        if self.recorded_check_policy == "complete":
            return tuple(required)
        return tuple(
            check
            for check in required
            if str(request.metadata.get(check, "")).strip()
        )

    def request_from_path(
        self, identifier: str, path: Path, metadata: Mapping[str, Any]
    ) -> Input:
        pixels = load_image(path, self.spec.image_size, self.channels)
        enriched = dict(metadata)
        enriched.setdefault("provenance", f"dataset:{self.name}")
        enriched["source_path"] = str(path)
        return Input(
            identifier=identifier,
            request=f"batch_frame={identifier}; "
            + "; ".join(f"{k}={v}" for k, v in sorted(enriched.items())),
            features=pixels,
            evidence=(),
            metadata=enriched,
            requested_action=self.spec.accept_action,
        )
