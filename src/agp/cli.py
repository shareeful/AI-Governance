from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .benchmarks.loaders import load_all
from .certify import certify
from .config import Config
from .errors import GovernanceError
from .experiments import EXPERIMENTS
from .experiments.base import artifact_root, seeds_from, systems_from
from .policy.compile import compile_policy
from .reporting import figures, tables
from .trust.ledger import AppendOnlyLedger, generate_signing_key


def _config(arguments: argparse.Namespace) -> Config:
    return Config.load(*arguments.config)


def _resolve_seeds(config: Config, requested: list[int] | None) -> tuple[int, ...]:
    return tuple(requested) if requested else seeds_from(config)


def _resolve_systems(config: Config, requested: list[str] | None) -> tuple[str, ...]:
    return tuple(requested) if requested else systems_from(config)


def command_keygen(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    private = Path(config.str_value("trust.ledger.private_key")).expanduser()
    public = Path(config.str_value("trust.ledger.public_key")).expanduser()
    if private.exists() and not arguments.force:
        print(
            f"a signing key already exists at {private}; pass --force to replace it",
            file=sys.stderr,
        )
        return 1
    generate_signing_key(private, public)
    print(f"signing key written to {private}")
    print(f"verification key written to {public}")
    return 0


def command_compile(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    policy = compile_policy(config)
    print(json.dumps(policy.to_dict(), indent=2, sort_keys=True))
    return 0


def command_train(arguments: argparse.Namespace) -> int:
    from .training import train

    config = _config(arguments)
    for system_name in _resolve_systems(config, arguments.system):
        for seed in _resolve_seeds(config, arguments.seed):
            report = train(config, system_name, seed)
            print(
                json.dumps(
                    {
                        "system": report.system,
                        "seed": report.seed,
                        "checkpoint": str(report.checkpoint),
                        "details": report.details,
                    },
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
    return 0


def command_shadow_pool(arguments: argparse.Namespace) -> int:
    from .training import train_shadow_pool

    config = _config(arguments)
    for system_name in _resolve_systems(config, arguments.system):
        written = train_shadow_pool(config, system_name)
        print(f"{system_name}: {len(written)} shadow checkpoints written")
    return 0


def command_certify(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    benchmarks = load_all(config)
    for system_name in _resolve_systems(config, arguments.system):
        for seed in _resolve_seeds(config, arguments.seed):
            deployment = certify(config, system_name, seed, benchmarks)
            print(json.dumps(deployment.calibration_report(), indent=2, sort_keys=True))
    return 0


def command_run(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    benchmarks = load_all(config)
    for system_name in _resolve_systems(config, arguments.system):
        for seed in _resolve_seeds(config, arguments.seed):
            deployment = certify(config, system_name, seed, benchmarks)
            requests = deployment.bundle.split(arguments.split)
            limit = arguments.limit if arguments.limit is not None else len(requests)
            released = 0
            held = 0
            withheld = 0
            referred = 0
            for request in requests[:limit]:
                result = deployment.pipeline(request)
                released += int(result.released)
                held += int(result.held_for_approval)
                withheld += int(not result.admitted)
                referred += int(result.referred)
                if arguments.verbose and result.record is not None:
                    print(json.dumps(result.record.to_dict(), sort_keys=True, default=str))
            print(
                json.dumps(
                    {
                        "system": system_name,
                        "seed": seed,
                        "split": arguments.split,
                        "executions": min(limit, len(requests)),
                        "released": released,
                        "held_for_human_approval": held,
                        "withheld": withheld,
                        "referred": referred,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
    return 0


def command_experiment(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    benchmarks = load_all(config)
    names = arguments.name if arguments.name else list(EXPERIMENTS)
    for name in names:
        if name not in EXPERIMENTS:
            print(
                f"unknown experiment '{name}'; valid experiments are "
                + ", ".join(sorted(EXPERIMENTS)),
                file=sys.stderr,
            )
            return 1
        result = EXPERIMENTS[name](config, benchmarks)
        print(f"{name}: {len(result.rows)} rows written to {artifact_root(config)}/results")
    return 0


def command_report(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    root = artifact_root(config)
    written = tables.write_all(root, arguments.table_format)
    if not arguments.tables_only:
        written.extend(figures.write_all(root, arguments.figure_format))
    for path in written:
        print(path)
    return 0


def command_verify_ledger(arguments: argparse.Namespace) -> int:
    config = _config(arguments)
    ledger = AppendOnlyLedger(
        path=Path(arguments.path).expanduser(),
        private_key_path=Path(config.str_value("trust.ledger.private_key")).expanduser(),
        public_key_path=Path(config.str_value("trust.ledger.public_key")).expanduser(),
    )
    count = ledger.verify()
    print(f"{count} records verified; hash chain and signatures intact")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agp",
        description=(
            "Runtime governance control plane for high-risk AI systems: policy compilation, "
            "certification, enforcement, and the reproduction of every reported experiment."
        ),
    )
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        help="configuration file; repeat to layer overrides over the base configuration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    keygen = subparsers.add_parser("keygen", help="generate the ledger signing key pair")
    keygen.add_argument("--force", action="store_true")
    keygen.set_defaults(handler=command_keygen)

    compile_parser = subparsers.add_parser(
        "compile", help="compile the five clauses into executable records (Phase 1)"
    )
    compile_parser.set_defaults(handler=command_compile)

    train_parser = subparsers.add_parser("train", help="train a governed decision model")
    train_parser.add_argument("--system", action="append")
    train_parser.add_argument("--seed", action="append", type=int)
    train_parser.set_defaults(handler=command_train)

    shadow = subparsers.add_parser(
        "shadow-pool", help="train the shadow-model pool required by the MNTD baseline"
    )
    shadow.add_argument("--system", action="append")
    shadow.set_defaults(handler=command_shadow_pool)

    certify_parser = subparsers.add_parser(
        "certify", help="calibrate the thresholds and record the fingerprint (Phase 2)"
    )
    certify_parser.add_argument("--system", action="append")
    certify_parser.add_argument("--seed", action="append", type=int)
    certify_parser.set_defaults(handler=command_certify)

    run_parser = subparsers.add_parser(
        "run", help="enforce the certified policy on a split (Phase 3)"
    )
    run_parser.add_argument("--system", action="append")
    run_parser.add_argument("--seed", action="append", type=int)
    run_parser.add_argument("--split", default="test")
    run_parser.add_argument("--limit", type=int)
    run_parser.add_argument("--verbose", action="store_true")
    run_parser.set_defaults(handler=command_run)

    experiment = subparsers.add_parser("experiment", help="run one or more reported experiments")
    experiment.add_argument("name", nargs="*")
    experiment.set_defaults(handler=command_experiment)

    report = subparsers.add_parser("report", help="render the reported tables and figures")
    report.add_argument("--table-format", action="append", default=None)
    report.add_argument("--figure-format", default="pdf")
    report.add_argument("--tables-only", action="store_true")
    report.set_defaults(handler=command_report)

    verify = subparsers.add_parser(
        "verify-ledger", help="verify the hash chain and signatures of an audit log (Phase 4)"
    )
    verify.add_argument("path")
    verify.set_defaults(handler=command_verify_ledger)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    if getattr(arguments, "table_format", None) is None and arguments.command == "report":
        arguments.table_format = ["csv", "tex"]
    try:
        return int(arguments.handler(arguments))
    except GovernanceError as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
