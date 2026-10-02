from __future__ import annotations

from typing import Sequence

from ..execution import EvidenceItem


class ProvenanceChecker:
    def __init__(self, accepted_authorities: frozenset[str]) -> None:
        self.accepted_authorities = accepted_authorities

    def admissible(self, item: EvidenceItem) -> bool:
        if not item.has_provenance:
            return False
        if not self.accepted_authorities:
            return True
        authority = str(item.provenance).split(":", 1)[0]
        return authority in self.accepted_authorities

    def rejected(self, evidence: Sequence[EvidenceItem]) -> tuple[EvidenceItem, ...]:
        return tuple(item for item in evidence if not self.admissible(item))
