from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from ..errors import LedgerIntegrityError, MissingResourceError

GENESIS_HASH = "0" * 64


def generate_signing_key(private_path: Path, public_path: Path) -> tuple[Path, Path]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    private_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    private_path.chmod(0o600)
    return private_path, public_path


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


@dataclass(frozen=True)
class LedgerRecord:
    body: Mapping[str, Any]
    previous_hash: str
    signature: str

    @property
    def identifier(self) -> str:
        return str(self.body["exec_id"])

    def content_hash(self) -> str:
        payload = dict(self.body)
        payload["prev_hash"] = self.previous_hash
        return hashlib.sha256(_canonical(payload)).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.body)
        payload["prev_hash"] = self.previous_hash
        payload["signature"] = self.signature
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LedgerRecord":
        body = {k: v for k, v in payload.items() if k not in {"prev_hash", "signature"}}
        return cls(
            body=body,
            previous_hash=str(payload["prev_hash"]),
            signature=str(payload["signature"]),
        )


class AppendOnlyLedger:
    def __init__(self, path: Path, private_key_path: Path, public_key_path: Path) -> None:
        from cryptography.hazmat.primitives.serialization import (
            load_pem_private_key,
            load_pem_public_key,
        )

        if not private_key_path.is_file():
            raise MissingResourceError(
                f"the ledger signing key was not found at {private_key_path}; generate one with "
                "'agp keygen' before enforcement"
            )
        if not public_key_path.is_file():
            raise MissingResourceError(
                f"the ledger verification key was not found at {public_key_path}"
            )
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._private_key = load_pem_private_key(private_key_path.read_bytes(), password=None)
        self._public_key = load_pem_public_key(public_key_path.read_bytes())
        self._tip = self._recover_tip()

    def _recover_tip(self) -> str:
        if not self.path.exists():
            return GENESIS_HASH
        tip = GENESIS_HASH
        for record in self.read():
            tip = record.content_hash()
        return tip

    @property
    def tip(self) -> str:
        return self._tip

    def append(self, body: Mapping[str, Any]) -> LedgerRecord:
        payload = dict(body)
        payload["prev_hash"] = self._tip
        signature = self._private_key.sign(_canonical(payload))
        record = LedgerRecord(
            body={k: v for k, v in payload.items() if k != "prev_hash"},
            previous_hash=self._tip,
            signature=signature.hex(),
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True, default=str) + "\n")
        self._tip = record.content_hash()
        return record

    def read(self) -> Iterator[LedgerRecord]:
        if not self.path.exists():
            return iter(())
        return self._iterate()

    def _iterate(self) -> Iterator[LedgerRecord]:
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    yield LedgerRecord.from_dict(json.loads(stripped))
                except (json.JSONDecodeError, KeyError) as exc:
                    raise LedgerIntegrityError(
                        f"ledger entry {line_number} in {self.path} is malformed"
                    ) from exc

    def verify(self) -> int:
        from cryptography.exceptions import InvalidSignature

        expected_previous = GENESIS_HASH
        count = 0
        for index, record in enumerate(self.read(), start=1):
            if record.previous_hash != expected_previous:
                raise LedgerIntegrityError(
                    f"hash chain broken at entry {index} ({record.identifier}): expected "
                    f"previous hash {expected_previous}, found {record.previous_hash}"
                )
            payload = dict(record.body)
            payload["prev_hash"] = record.previous_hash
            try:
                self._public_key.verify(bytes.fromhex(record.signature), _canonical(payload))
            except InvalidSignature as exc:
                raise LedgerIntegrityError(
                    f"signature on entry {index} ({record.identifier}) does not verify"
                ) from exc
            expected_previous = record.content_hash()
            count += 1
        return count
