#!/usr/bin/env python3
"""Combine CV5, LOSO, sensitivity, primary decision, and Holm results."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.data import DEFAULT_DATA, sha256_file


ROOT = Path(__file__).resolve().parent
CV = ROOT / "results" / "phase2-cv5-final-aeeb4b4"
LOSO = ROOT / "results" / "phase2-loso-final-a689ba5"
SENS = ROOT / "results" / "phase2-sensitivity-final-a689ba5"
DIAG = ROOT / "results" / "phase2-cv5-diagnostics-77c4752"


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()


def holm(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.sort_values("bootstrap_p_value").copy()
    adjusted = []
    running = 0.0
    m = len(output)
    for rank, p_value in enumerate(output["bootstrap_p_value"], start=1):
        value = min(1.0, (m - rank + 1) * float(p_value))
        running = max(running, value)
        adjusted.append(running)
    output["holm_adjusted_p_value"] = adjusted
    output["significant_after_holm"] = output["holm_adjusted_p_value"] < 0.05
    return output


def main() -> None:
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cv_main = pd.read_csv(DIAG / "main_table_corrected.csv")
    loso_main = pd.read_csv(LOSO / "main_table.csv")
    sensitivity_main = pd.read_csv(SENS / "main_table.csv")
    cv_main.to_csv(output / "cv5_main_table.csv", index=False)
    loso_main.to_csv(output / "loso_main_table.csv", index=False)
    sensitivity_main.to_csv(output / "sensitivity_main_table.csv", index=False)

    cv_draws = pd.read_csv(CV / "bootstrap_by_method.csv")
    wide = cv_draws.pivot(
        index="bootstrap_replicate", columns="method", values="primary_error"
    )
    cv_pairs = pd.read_csv(CV / "paired_differences.csv")
    p_values = []
    for row in cv_pairs.itertuples(index=False):
        difference = wide[row.left] - wide[row.right]
        p_values.append(
            min(
                1.0,
                2.0
                * min(
                    float((difference >= 0).mean()),
                    float((difference <= 0).mean()),
                ),
            )
        )
    cv_pairs["bootstrap_p_value"] = p_values
    cv_pairs.to_csv(output / "cv5_paired_differences.csv", index=False)

    loso_pairs = pd.read_csv(LOSO / "paired_differences.csv")
    sensitivity_pairs = pd.read_csv(SENS / "paired_differences.csv")
    primary = pd.concat(
        [
            cv_pairs.loc[cv_pairs["role"].eq("primary")].assign(
                protocol="cv5_partitioned"
            ),
            loso_pairs.loc[loso_pairs["role"].eq("primary")].assign(
                protocol="loso"
            ),
            sensitivity_pairs.loc[
                sensitivity_pairs["role"].eq("primary")
            ].assign(protocol="sensitivity"),
        ],
        ignore_index=True,
    )
    primary.to_csv(output / "primary_decision.csv", index=False)

    secondary = pd.concat(
        [
            cv_pairs.loc[cv_pairs["role"].eq("secondary")].assign(
                protocol="cv5_partitioned"
            ),
            loso_pairs.loc[loso_pairs["role"].eq("secondary")].assign(
                protocol="loso"
            ),
        ],
        ignore_index=True,
    )
    secondary = holm(secondary)
    secondary.to_csv(output / "secondary_holm.csv", index=False)

    labels = primary.loc[
        primary["protocol"].isin(["cv5_partitioned", "loso"])
    ].set_index("protocol")["status"].to_dict()
    combined_label = (
        next(iter(labels.values()))
        if len(set(labels.values())) == 1
        else "inconclusive"
    )
    decision = {
        "hypothesis": "best_gnn - mlp primary error",
        "cv5_label": labels["cv5_partitioned"],
        "loso_label": labels["loso"],
        "combined_label": combined_label,
        "both_same_label": len(set(labels.values())) == 1,
        "sensitivity_label_not_co_primary": primary.loc[
            primary["protocol"].eq("sensitivity"), "status"
        ].iloc[0],
    }
    (output / "co_primary_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_branch": git_output("branch", "--show-current"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "data_sha256": sha256_file(DEFAULT_DATA),
        "protocol_sha256": sha256_file(ROOT / "protocol.md"),
        "input_manifests": {
            "cv5": sha256_file(CV / "results_manifest.json"),
            "loso": sha256_file(LOSO / "results_manifest.json"),
            "sensitivity": sha256_file(SENS / "results_manifest.json"),
            "diagnostics": sha256_file(DIAG / "results_manifest.json"),
        },
        "loafo_run": False,
        "output_sha256": {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.iterdir())
            if path.is_file()
        },
    }
    (output / "results_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(output)


if __name__ == "__main__":
    main()
