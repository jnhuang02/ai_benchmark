#!/usr/bin/env python3
"""Generate Phase 1 folds, run the seeded comparison, and stamp all outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from evaluation.evaluate import GNNConfig
from evaluation.run import DEFAULT_OUTPUT_ROOT, SEED_SET, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Exclusive output directory; defaults to a UTC/commit-stamped path",
    )
    parser.add_argument("--protocol", default="cv5", choices=["cv5", "loso", "loafo", "sensitivity"])
    parser.add_argument(
        "--model",
        default="graphsage",
        choices=["group_median", "ridge", "random_forest", "graphsage", "gcn", "gat"],
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["legacy_leaky_global", "isolated_supported"],
        choices=["legacy_leaky_global", "isolated_supported"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEED_SET))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != 10:
        raise ValueError("Phase 1 stochastic models require exactly 10 distinct seeds")
    if args.bootstrap_replicates != 2000:
        raise ValueError("The Phase 1 clustered bootstrap must use B=2000")
    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = DEFAULT_OUTPUT_ROOT / f"phase1-{stamp}"
    config = GNNConfig(epochs=args.epochs, patience=args.patience)
    run(
        output=output,
        protocol=args.protocol,
        model=args.model,
        policies=tuple(args.policies),
        seeds=tuple(args.seeds),
        config=config,
        bootstrap_replicates=args.bootstrap_replicates,
    )


if __name__ == "__main__":
    main()
