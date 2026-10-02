from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from agp.config import Config
from agp.errors import GovernanceError
from agp.systems.imaging.dataset import group_identifier
from agp.systems.registry import imaging_splits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Report group-level overlap between the configured splits of an imaging system. "
            "A patient or lesion appearing on both sides of a split inflates every reported "
            "metric, so this must be empty before certification."
        )
    )
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--system", required=True)
    arguments = parser.parse_args(argv)

    config = Config.load(*arguments.config)
    prefix = f"systems.{arguments.system}"
    strategy = config.str_value(f"{prefix}.dataset.identifier_from")
    splits = imaging_splits(config, prefix)

    membership: dict[str, set[str]] = defaultdict(set)
    for split_name, split in splits.items():
        for record in split.records:
            membership[group_identifier(record, strategy)].add(split_name)

    overlaps = {group: names for group, names in membership.items() if len(names) > 1}
    print(f"system: {arguments.system}")
    print(f"identifier strategy: {strategy}")
    for split_name, split in sorted(splits.items()):
        groups = {group_identifier(r, strategy) for r in split.records}
        print(f"  {split_name}: {len(split)} records, {len(groups)} groups")
    print(f"groups spanning more than one split: {len(overlaps)}")
    for group, names in sorted(overlaps.items())[:20]:
        print(f"    {group}: {', '.join(sorted(names))}")
    if len(overlaps) > 20:
        print(f"    ... and {len(overlaps) - 20} more")
    return 1 if overlaps else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GovernanceError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1)
