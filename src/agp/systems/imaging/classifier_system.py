from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...checkers.scope import ActionRegister
from ...config import Config
from ...errors import ConfigurationError, InterfaceError
from ...execution import Input
from ..base import InstrumentedSystem, RationaleTemplates
from .backbone import (
    BackboneSpec,
    class_names,
    integrated_gradients_regions,
    load_backbone,
    occlusion_regions,
    region_grid,
    region_names,
    softmax_scores,
)
from .dataset import ImageRecord, load_image, normalise


class ImageClassificationSystem(InstrumentedSystem):
    def __init__(
        self,
        name: str,
        config: Config,
        prefix: str,
        checkpoint: Path,
        device: str,
        action_register: ActionRegister,
        attribute_domains: Mapping[str, Sequence[Any]],
    ) -> None:
        self.spec = BackboneSpec.from_config(config, f"{prefix}.model", device)
        self.network = load_backbone(self.spec, checkpoint)
        self.classes = class_names(config, f"{prefix}.model")
        if len(self.classes) != self.spec.num_classes:
            raise ConfigurationError(
                f"{prefix}.model.class_names lists {len(self.classes)} names but num_classes is "
                f"{self.spec.num_classes}"
            )
        self.channels = config.int_value(f"{prefix}.model.channels")
        self.normalisation_mean = tuple(config.list_value(f"{prefix}.model.normalisation.mean"))
        self.normalisation_std = tuple(config.list_value(f"{prefix}.model.normalisation.std"))
        rows = config.int_value(f"{prefix}.attribution.grid_rows")
        columns = config.int_value(f"{prefix}.attribution.grid_columns")
        self.regions = region_grid(self.spec.image_size, rows, columns)
        self.attribution_method = config.str_value(f"{prefix}.attribution.method")
        if self.attribution_method not in {"integrated_gradients", "occlusion"}:
            raise ConfigurationError(
                f"{prefix}.attribution.method must be 'integrated_gradients' or 'occlusion'"
            )
        self.attribution_steps = config.int_value(f"{prefix}.attribution.steps")
        self.occlusion_fill = config.float_value(f"{prefix}.attribution.occlusion_fill")
        self.paraphrase_max_rotation = config.float_value(
            f"{prefix}.paraphrase.max_rotation_degrees"
        )
        self.paraphrase_max_brightness = config.require_unit_interval(
            f"{prefix}.paraphrase.max_brightness_delta"
        )
        self.paraphrase_horizontal_flip = config.bool_value(f"{prefix}.paraphrase.horizontal_flip")
        self.quality_threshold = config.float_value(f"{prefix}.preconditions.quality_threshold")
        self.quality_metric = config.str_value(f"{prefix}.preconditions.quality_metric")
        if self.quality_metric not in {"laplacian_variance", "intensity_std"}:
            raise ConfigurationError(
                f"{prefix}.preconditions.quality_metric must be 'laplacian_variance' or "
                "'intensity_std'"
            )
        self.history_field = config.str_value(f"{prefix}.preconditions.history_field")

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

    def request_from_record(self, record: ImageRecord) -> Input:
        pixels = load_image(record.path, self.spec.image_size, self.channels)
        metadata = dict(record.metadata)
        metadata.setdefault("provenance", f"dataset:{self.name}")
        metadata["source_path"] = str(record.path)
        metadata["label"] = record.label
        return Input(
            identifier=record.identifier,
            request=self._request_text(record),
            features=pixels,
            evidence=(),
            metadata=metadata,
            requested_action=self._requested_action(),
        )

    def _request_text(self, record: ImageRecord) -> str:
        fields = [f"{key}={value}" for key, value in sorted(record.metadata.items())]
        return f"case={record.identifier}; " + "; ".join(fields)

    def _requested_action(self) -> str:
        actions = list(self.action_register)
        if not actions:
            raise ConfigurationError(f"system '{self.name}' has an empty action register")
        return actions[0]

    def _tensor(self, request: Input) -> np.ndarray:
        pixels = np.asarray(request.features, dtype=np.float32)
        if pixels.ndim != 3:
            raise InterfaceError(
                f"system '{self.name}' expects a channel-first image array; got shape {pixels.shape}"
            )
        return normalise(pixels[None, ...], self.normalisation_mean, self.normalisation_std)[0]

    def decision_scores(self, request: Input) -> np.ndarray:
        return softmax_scores(self.network, self._tensor(request)[None, ...], self.spec.device)[0]

    def predict_action(self, request: Input) -> str:
        return self.classes[int(np.argmax(self.decision_scores(request)))]

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
        tensor = self._tensor(request)
        target = int(np.argmax(self.decision_scores(request)))
        if self.attribution_method == "integrated_gradients":
            masses = integrated_gradients_regions(
                self.network, tensor, target, self.regions, self.attribution_steps, self.spec.device
            )
        else:
            masses = occlusion_regions(
                self.network, tensor, target, self.regions, self.occlusion_fill, self.spec.device
            )
        return {name: float(mass) for name, mass in zip(self.feature_groups, masses)}

    def substitute_attribute(self, request: Input, attribute: str, value: Any) -> Input:
        if attribute not in self.protected_attributes:
            raise InterfaceError(
                f"'{attribute}' is not a documented protected attribute of system '{self.name}'"
            )
        metadata = dict(request.metadata)
        metadata[attribute] = value
        substituted = replace(request, metadata=metadata)
        return replace(substituted, request=self._render_request(substituted))

    def _render_request(self, request: Input) -> str:
        fields = [
            f"{key}={value}"
            for key, value in sorted(request.metadata.items())
            if key not in {"source_path", "label"}
        ]
        return f"case={request.identifier}; " + "; ".join(fields)

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

    def superimpose(self, request: Input, donor: np.ndarray, weight: float) -> Input:
        pixels = np.asarray(request.features, dtype=np.float32)
        blended = np.clip((1.0 - weight) * pixels + weight * donor, 0.0, 1.0).astype(np.float32)
        return replace(request, features=blended)

    def paraphrases(self, request: Input, count: int, rng: np.random.Generator) -> list[Input]:
        from scipy.ndimage import rotate

        if count < 1:
            raise InterfaceError("the paraphrase signal requires at least one variant")
        pixels = np.asarray(request.features, dtype=np.float32)
        outputs: list[Input] = []
        for index in range(count):
            variant = pixels
            angle = float(rng.uniform(-self.paraphrase_max_rotation, self.paraphrase_max_rotation))
            if abs(angle) > 0.0:
                variant = rotate(
                    variant, angle, axes=(1, 2), reshape=False, order=1, mode="nearest"
                )
            delta = float(rng.uniform(-self.paraphrase_max_brightness, self.paraphrase_max_brightness))
            variant = variant + delta
            if self.paraphrase_horizontal_flip and index % 2 == 1:
                variant = variant[:, :, ::-1]
            outputs.append(
                replace(request, features=np.clip(variant, 0.0, 1.0).astype(np.float32))
            )
        return outputs

    def image_quality(self, request: Input) -> float:
        pixels = np.asarray(request.features, dtype=np.float32)
        if self.quality_metric == "intensity_std":
            return float(np.std(pixels))
        from scipy.ndimage import laplace

        return float(np.var(laplace(pixels.mean(axis=0))))

    def recorded_checks(self, request: Input, action: str) -> tuple[str, ...]:
        required = sorted(self.action_register.required_checks(action))
        if self.recorded_check_policy == "complete":
            return tuple(required)
        satisfied: list[str] = []
        quality = self.image_quality(request)
        for check in required:
            if check == self.quality_metric or check.endswith("quality_verification"):
                if quality >= self.quality_threshold:
                    satisfied.append(check)
            elif check.endswith(self.history_field) or check.endswith("history_review"):
                if str(request.metadata.get(self.history_field, "")).strip():
                    satisfied.append(check)
            else:
                satisfied.append(check)
        return tuple(satisfied)
