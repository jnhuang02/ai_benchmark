"""Result-bundle locations for the downstream analysis and thesis scripts.

``AI_BENCHMARK_BUNDLES`` selects which set of Phase 2 bundles the combine,
diagnostic, tail, correction, and thesis-table scripts read and write:

- ``frozen`` (default): the protocol v2.1 bundles, unchanged.
- ``memory``: the bundles produced with ``AI_BENCHMARK_FEATURE_SET=memory``.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BUNDLE_SETS: dict[str, dict[str, str]] = {
    "frozen": {
        "cv5": "phase2-cv5-final-aeeb4b4",
        "loso": "phase2-loso-final-a689ba5",
        "loso_shard_glob": "phase2-loso-shard*",
        "sensitivity": "phase2-sensitivity-final-a689ba5",
        "loafo": "phase2-loafo-final-1a65664",
        "diagnostics": "phase2-cv5-diagnostics-77c4752",
        "prerun": "phase2-prerun-1122699",
        "tail": "phase2-5-analysis-4faac06",
        "corrections": "phase2-5-corrections-e97bb93",
        "final": "phase2-final-f51e57a",
    },
    "memory": {
        "cv5": "exploratory-memory-cv5-final",
        "loso": "exploratory-memory-loso-final",
        "loso_shard_glob": "exploratory-memory-loso-shard*",
        "sensitivity": "exploratory-memory-sensitivity-final",
        "loafo": "exploratory-memory-loafo-final",
        "diagnostics": "memory-cv5-diagnostics",
        "prerun": "phase2-prerun-d539fc2",
        "tail": "memory-phase2-5-analysis",
        "corrections": "memory-phase2-5-corrections",
        "final": "memory-phase2-final",
    },
}

BUNDLE_SET = os.environ.get("AI_BENCHMARK_BUNDLES", "frozen")
if BUNDLE_SET not in BUNDLE_SETS:
    raise ValueError(
        f"AI_BENCHMARK_BUNDLES must be one of {tuple(BUNDLE_SETS)}, got {BUNDLE_SET!r}"
    )


def bundle(name: str) -> Path:
    return RESULTS / BUNDLE_SETS[BUNDLE_SET][name]


def loso_shard_glob() -> str:
    return BUNDLE_SETS[BUNDLE_SET]["loso_shard_glob"]
