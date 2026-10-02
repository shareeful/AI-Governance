from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from ..errors import AttestationError
from .signals import SCALAR_SIGNALS, VECTOR_SIGNAL, SignalVector


@dataclass(frozen=True)
class SignalSummary:
    mean: float
    variance: float
    percentile_5: float
    percentile_95: float

    @classmethod
    def of(cls, values: np.ndarray) -> "SignalSummary":
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            raise AttestationError("cannot summarise an empty signal sample")
        return cls(
            mean=float(np.mean(values)),
            variance=float(np.var(values, ddof=1)) if values.size > 1 else 0.0,
            percentile_5=float(np.percentile(values, 5)),
            percentile_95=float(np.percentile(values, 95)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "variance": self.variance,
            "percentile_5": self.percentile_5,
            "percentile_95": self.percentile_95,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, float]) -> "SignalSummary":
        return cls(
            mean=float(payload["mean"]),
            variance=float(payload["variance"]),
            percentile_5=float(payload["percentile_5"]),
            percentile_95=float(payload["percentile_95"]),
        )


@dataclass(frozen=True)
class BehaviouralFingerprint:
    system: str
    workload_size: int
    feature_names: tuple[str, ...]
    scalar_samples: Mapping[str, np.ndarray]
    feature_samples: np.ndarray
    summaries: Mapping[str, SignalSummary]
    feature_summary: Mapping[str, SignalSummary]
    reference_feature_distribution: np.ndarray

    @classmethod
    def build(cls, system_name: str, vectors: Sequence[SignalVector]) -> "BehaviouralFingerprint":
        if not vectors:
            raise AttestationError(
                "the certification workload produced no executions; the behavioural "
                "fingerprint cannot be recorded"
            )
        feature_names = vectors[0].feature_names
        for vector in vectors:
            if vector.feature_names != feature_names:
                raise AttestationError(
                    "feature-group ordering is inconsistent across the certification workload"
                )
        scalar_samples = {
            name: np.array([v.scalar(name) for v in vectors], dtype=float)
            for name in SCALAR_SIGNALS
        }
        feature_samples = np.vstack([v.feature_reliance for v in vectors])
        reference = feature_samples.mean(axis=0)
        total = reference.sum()
        if total <= 0.0:
            raise AttestationError("the certified feature-reliance distribution has zero mass")
        reference = reference / total
        return cls(
            system=system_name,
            workload_size=len(vectors),
            feature_names=feature_names,
            scalar_samples=scalar_samples,
            feature_samples=feature_samples,
            summaries={n: SignalSummary.of(s) for n, s in scalar_samples.items()},
            feature_summary={
                name: SignalSummary.of(feature_samples[:, index])
                for index, name in enumerate(feature_names)
            },
            reference_feature_distribution=reference,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "system": self.system,
            "workload_size": self.workload_size,
            "feature_names": list(self.feature_names),
            "scalar_samples": {k: v.tolist() for k, v in self.scalar_samples.items()},
            "feature_samples": self.feature_samples.tolist(),
            "summaries": {k: v.to_dict() for k, v in self.summaries.items()},
            "feature_summary": {k: v.to_dict() for k, v in self.feature_summary.items()},
            "reference_feature_distribution": self.reference_feature_distribution.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "BehaviouralFingerprint":
        feature_names = tuple(str(n) for n in payload["feature_names"])
        scalar_samples = {
            str(k): np.asarray(v, dtype=float)
            for k, v in dict(payload["scalar_samples"]).items()
        }
        feature_samples = np.asarray(payload["feature_samples"], dtype=float)
        return cls(
            system=str(payload["system"]),
            workload_size=int(payload["workload_size"]),
            feature_names=feature_names,
            scalar_samples=scalar_samples,
            feature_samples=feature_samples,
            summaries={
                str(k): SignalSummary.from_dict(v)
                for k, v in dict(payload["summaries"]).items()
            },
            feature_summary={
                str(k): SignalSummary.from_dict(v)
                for k, v in dict(payload["feature_summary"]).items()
            },
            reference_feature_distribution=np.asarray(
                payload["reference_feature_distribution"], dtype=float
            ),
        )

    def signal_names(self) -> tuple[str, ...]:
        return SCALAR_SIGNALS + (VECTOR_SIGNAL,)


def sign_fingerprint(
    fingerprint: BehaviouralFingerprint,
    private_key_path: Path,
    output_path: Path,
) -> Path:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    payload = json.dumps(fingerprint.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    with private_key_path.open("rb") as handle:
        key = load_pem_private_key(handle.read(), password=None)
    signature = key.sign(payload)
    public_bytes = key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "fingerprint": fingerprint.to_dict(),
                "signature": signature.hex(),
                "public_key": public_bytes.hex(),
            },
            handle,
            sort_keys=True,
        )
    return output_path


def load_signed_fingerprint(path: Path) -> BehaviouralFingerprint:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    payload = json.dumps(document["fingerprint"], sort_keys=True, separators=(",", ":")).encode()
    public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(document["public_key"]))
    try:
        public_key.verify(bytes.fromhex(document["signature"]), payload)
    except InvalidSignature as exc:
        raise AttestationError(
            f"the signature on the behavioural fingerprint at {path} does not verify"
        ) from exc
    return BehaviouralFingerprint.from_dict(document["fingerprint"])
