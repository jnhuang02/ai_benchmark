"""Manifest helpers for exploratory post hoc analyses."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_LABEL = "exploratory, post hoc, not part of the frozen protocol"


def git_output(*args: str, root: Path) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL
    ).strip()


def software_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for module in [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "torch",
        "torch_geometric",
    ]:
        imported = __import__(module)
        versions[module] = str(getattr(imported, "__version__", "unknown"))
    return versions


def write_manifest(
    output: Path,
    *,
    root: Path,
    command: list[str],
    input_files: dict[str, Path],
) -> None:
    from evaluation.data import sha256_file

    outputs = {
        str(path.relative_to(output)): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "results_manifest.json"
    }
    manifest = {
        "label": ALLOWED_LABEL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD", root=root),
        "git_branch": git_output("branch", "--show-current", root=root),
        "git_dirty": bool(git_output("status", "--porcelain", root=root)),
        "command": command,
        "data_sha256": sha256_file(root / "data" / "mlperf_long.csv"),
        "fold_assignments_sha256": sha256_file(
            root / "folds" / "phase2_fold_assignments.csv"
        ),
        "protocol_sha256": sha256_file(root / "protocol.md"),
        "accelerator_map_sha256": sha256_file(
            root / "config" / "accelerator_families.csv"
        ),
        "software_versions": software_versions(),
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in sorted(input_files.items())
        },
        "output_sha256": outputs,
    }
    (output / "results_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

