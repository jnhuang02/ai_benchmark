#!/usr/bin/env python3
"""B2: characterize systems and rows removed by training-only support."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from evaluation.data import (
    DEFAULT_DATA,
    load_dataset,
    load_family_map,
    system_table,
)
from evaluation.features import StrictTargetScaler
from exploratory_manifest import ALLOWED_LABEL, write_manifest


ROOT = Path(__file__).resolve().parent
FOLDS = ROOT / "folds" / "phase2_fold_assignments.csv"
EXPECTED_CV_ROWS = {
    "cv5_partition_0": 450,
    "cv5_partition_1": 451,
    "cv5_partition_2": 452,
    "cv5_partition_3": 448,
    "cv5_partition_4": 448,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def fold_row_support(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    protocol: str,
    fold_id: str,
) -> pd.DataFrame:
    local = assignments.loc[
        assignments["protocol"].eq(protocol)
        & assignments["fold_id"].eq(fold_id)
    ]
    train_systems = set(local.loc[local["role"].eq("train"), "system_id"])
    test_systems = set(local.loc[local["role"].eq("test"), "system_id"])
    train = frame.loc[frame["system_id"].isin(train_systems)]
    test = frame.loc[frame["system_id"].isin(test_systems)].copy()
    supported_groups = set(StrictTargetScaler.fit(train).supported_groups)
    return test.assign(
        protocol=protocol,
        fold_id=fold_id,
        supported=test["target_group"].isin(supported_groups),
    )


def summarize_rows(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(
            ["protocol", "row_uid", "system_id", "suite", "target_group"],
            as_index=False,
        )
        .agg(
            test_opportunities=("fold_id", "size"),
            supported_test_opportunities=("supported", "sum"),
        )
        .assign(
            unsupported_test_opportunities=lambda value: (
                value["test_opportunities"]
                - value["supported_test_opportunities"]
            ),
            never_supported=lambda value: value[
                "supported_test_opportunities"
            ].eq(0),
        )
    )


def system_summary(
    rows: pd.DataFrame, systems: pd.DataFrame
) -> pd.DataFrame:
    summary = (
        rows.groupby(["protocol", "system_id"], as_index=False)
        .agg(
            measurements=("row_uid", "nunique"),
            test_opportunities=("supported", "size"),
            supported_test_opportunities=("supported", "sum"),
        )
    )
    row_level = (
        rows.groupby(["protocol", "system_id", "row_uid"])["supported"]
        .any()
        .groupby(level=[0, 1])
        .agg(["size", "sum"])
        .reset_index()
        .rename(
            columns={
                "size": "measurements",
                "sum": "ever_supported_rows",
            }
        )
    )
    row_level["never_supported_rows"] = (
        row_level["measurements"] - row_level["ever_supported_rows"]
    )
    summary = summary.merge(
        row_level, on=["protocol", "system_id"], how="left"
    )
    summary["unsupported_test_opportunities"] = (
        summary["test_opportunities"]
        - summary["supported_test_opportunities"]
    )
    summary["fully_dropped"] = summary["ever_supported_rows"].eq(0)
    return summary.merge(
        systems[["system_id", "submitter", "accelerator_family", "suite"]],
        on="system_id",
        how="left",
        validate="many_to_one",
    )


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    frame = load_dataset(DEFAULT_DATA)
    assignments = pd.read_csv(FOLDS)
    systems = system_table(frame, load_family_map())

    cv_rows = []
    for protocol in sorted(
        assignments.loc[
            assignments["protocol"].str.startswith("cv5_partition_"),
            "protocol",
        ].unique()
    ):
        for fold_id in sorted(
            assignments.loc[
                assignments["protocol"].eq(protocol), "fold_id"
            ].unique()
        ):
            cv_rows.append(
                fold_row_support(
                    frame,
                    assignments,
                    protocol=protocol,
                    fold_id=fold_id,
                )
            )
    cv_opportunities = pd.concat(cv_rows, ignore_index=True)
    cv_rows_summary = summarize_rows(cv_opportunities)
    cv_global_rows_summary = summarize_rows(
        cv_opportunities.assign(protocol="cv5_all")
    )

    loso_rows = []
    for fold_id in sorted(
        assignments.loc[assignments["protocol"].eq("loso"), "fold_id"].unique()
    ):
        loso_rows.append(
            fold_row_support(
                frame,
                assignments,
                protocol="loso",
                fold_id=fold_id,
            )
        )
    loso_opportunities = pd.concat(loso_rows, ignore_index=True)
    loso_rows_summary = summarize_rows(loso_opportunities)

    cv_systems = system_summary(cv_opportunities, systems)
    cv_global_systems = system_summary(
        cv_opportunities.assign(protocol="cv5_all"), systems
    )
    loso_systems = system_summary(loso_opportunities, systems)
    cv_zero = cv_systems.loc[cv_systems["fully_dropped"]].copy()
    cv_zero_global = cv_global_systems.loc[
        cv_global_systems["fully_dropped"]
    ].copy()
    loso_zero = loso_systems.loc[loso_systems["fully_dropped"]].copy()
    if loso_zero["system_id"].nunique() != 6:
        raise AssertionError(
            f"Expected six zero-support LOSO systems, found "
            f"{loso_zero['system_id'].nunique()}"
        )

    loso_zero_rows = frame.loc[frame["system_id"].isin(loso_zero["system_id"])].copy()
    workload_lists = (
        loso_zero_rows.groupby("system_id")["target_group"]
        .agg(lambda values: "; ".join(sorted(set(values))))
        .rename("workloads")
    )
    scenario_lists = (
        loso_zero_rows.groupby("system_id")["scenario"]
        .agg(lambda values: "; ".join(sorted(set(values))))
        .rename("scenarios")
    )
    model_lists = (
        loso_zero_rows.groupby("system_id")["model"]
        .agg(lambda values: "; ".join(sorted(set(values))))
        .rename("models")
    )
    loso_zero_characteristics = (
        loso_zero.set_index("system_id")
        .join(workload_lists)
        .join(scenario_lists)
        .join(model_lists)
        .reset_index()
    )

    counts = []
    for protocol, local in cv_opportunities.groupby("protocol", sort=True):
        counts.append(
            {
                "protocol": protocol,
                "heldout_systems": local["system_id"].nunique(),
                "test_rows": len(local),
                "supported_test_rows": int(local["supported"].sum()),
                "unsupported_test_rows": int((~local["supported"]).sum()),
            }
        )
        expected = EXPECTED_CV_ROWS[protocol]
        if int(local["supported"].sum()) != expected:
            raise AssertionError(
                f"{protocol} supported rows {local['supported'].sum()} != {expected}"
            )
    counts.append(
        {
            "protocol": "loso",
            "heldout_systems": loso_opportunities["system_id"].nunique(),
            "test_rows": len(loso_opportunities),
            "supported_test_rows": int(loso_opportunities["supported"].sum()),
            "unsupported_test_rows": int((~loso_opportunities["supported"]).sum()),
        }
    )
    counts.append(
        {
            "protocol": "cv5_all",
            "heldout_systems": cv_opportunities["system_id"].nunique(),
            "test_rows": len(cv_opportunities),
            "supported_test_rows": int(cv_opportunities["supported"].sum()),
            "unsupported_test_rows": int((~cv_opportunities["supported"]).sum()),
        }
    )
    if int(loso_opportunities["supported"].sum()) != 454:
        raise AssertionError("LOSO supported rows do not equal 454")

    cv_opportunities.to_csv(output / "cv5_fold_row_opportunities.csv", index=False)
    cv_rows_summary.to_csv(output / "cv5_row_support.csv", index=False)
    cv_global_rows_summary.to_csv(
        output / "cv5_row_support_global.csv", index=False
    )
    cv_systems.to_csv(output / "cv5_system_support.csv", index=False)
    cv_global_systems.to_csv(
        output / "cv5_system_support_global.csv", index=False
    )
    cv_zero.to_csv(output / "cv5_zero_support_systems.csv", index=False)
    cv_zero_global.to_csv(
        output / "cv5_zero_support_systems_global.csv", index=False
    )
    loso_opportunities.to_csv(output / "loso_fold_row_opportunities.csv", index=False)
    loso_rows_summary.to_csv(output / "loso_row_support.csv", index=False)
    loso_systems.to_csv(output / "loso_system_support.csv", index=False)
    loso_zero.to_csv(output / "loso_zero_support_systems.csv", index=False)
    loso_zero_rows.to_csv(output / "loso_zero_support_rows.csv", index=False)
    loso_zero_characteristics.to_csv(
        output / "loso_zero_support_characteristics.csv", index=False
    )
    pd.DataFrame(counts).to_csv(output / "support_counts.csv", index=False)
    pd.DataFrame(
        [
            {
                "scope": "cv5_all",
                "unique_rows": int(len(cv_global_rows_summary)),
                "rows_never_supported": int(
                    cv_global_rows_summary["never_supported"].sum()
                ),
                "systems_fully_dropped": int(
                    cv_global_systems["fully_dropped"].sum()
                ),
            },
            {
                "scope": "loso",
                "unique_rows": int(len(loso_rows_summary)),
                "rows_never_supported": int(
                    loso_rows_summary["never_supported"].sum()
                ),
                "systems_fully_dropped": int(loso_zero["system_id"].nunique()),
            },
        ]
    ).to_csv(output / "attrition_totals.csv", index=False)
    (output / "README.md").write_text(
        "\n".join(
            [
                "# B2: supported-row attrition",
                "",
                f"Label: **{ALLOWED_LABEL}**.",
                "",
                "A held-out row is evaluated only when its target group has at "
                "least three training rows and nonzero training scale in that "
                "outer fold. The analysis enumerates which rows and systems lose "
                "one or all held-out opportunities under repeated CV5 and LOSO.",
                "",
                "No model is fitted. Suite, submitter, accelerator family, "
                "workload, scenario, and model characteristics come from the "
                "canonical 476-row MLPerf v6.1 Closed-division dataset.",
            ]
        )
        + "\n"
    )
    write_manifest(
        output,
        root=ROOT,
        command=[sys.executable, str(Path(__file__)), *sys.argv[1:]],
        input_files={
            "mlperf_long": DEFAULT_DATA,
            "fold_assignments": FOLDS,
            "accelerator_families": ROOT / "config" / "accelerator_families.csv",
            "protocol": ROOT / "protocol.md",
        },
    )
    print(output)


if __name__ == "__main__":
    main()
