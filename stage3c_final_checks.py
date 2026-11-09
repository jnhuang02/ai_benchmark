#!/usr/bin/env python3
"""Run the final Stage 3C integrity and rendered-source checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
THESIS = ROOT.parent / "thesis_overleaf"
PROTOCOL = ROOT / "protocol.md"
PROTOCOL_HISTORY = ROOT / "protocol_history.md"
LOCKFILE = ROOT / "requirements-lock.txt"
FINAL_BUNDLES = (
    "phase2-cv5-final-aeeb4b4",
    "phase2-cv5-diagnostics-77c4752",
    "phase2-loso-final-a689ba5",
    "phase2-sensitivity-final-a689ba5",
    "phase2-final-f51e57a",
    "phase2-loafo-final-1a65664",
    "phase2-5-analysis-4faac06",
    "phase2-5-corrections-e97bb93",
)
EXPECTED_PROTOCOL_SHA256 = (
    "d4c5930477b54f6b3a65bbc1be98b755bc4a2335efcc2261294469d6f0123f3e"
)
EXPECTED_VERSIONS = {
    "numpy": "2.4.4",
    "pandas": "3.0.2",
    "scikit-learn": "1.8.0",
    "statsmodels": "0.15.0",
    "scipy": "1.17.1",
    "torch": "2.11.0",
    "torch-geometric": "2.7.0",
}
PLACEHOLDERS = (
    "[AUTHOR TO SUPPLY COMMITTEE CHAIR]",
    "[AUTHOR TO SUPPLY COMMITTEE MEMBER]",
    "[AUTHOR TO SUPPLY FINAL DEDICATION]",
    "[AUTHOR TO SUPPLY ACKNOWLEDGMENTS]",
    "[AUTHOR TO SUPPLY BIRTHPLACE]",
    "[AUTHOR TO SUPPLY B.S. INSTITUTION AND YEAR]",
    "[AUTHOR TO SUPPLY PUBLICATIONS AND PRESENTATIONS]",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_number(value: object) -> str | None:
    text = (
        str(value)
        .replace(",", "")
        .replace("\u2212", "-")
        .replace("\u2013", "-")
        .strip()
    )
    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f")


def displayed_source_value(value: object, expected: str) -> str | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    decimals = len(expected.split(".", 1)[1]) if "." in expected else 0
    return normalized_number(f"{number:.{decimals}f}")


def source_context(path: Path, line_number: int, radius: int = 2) -> str:
    lines = path.read_text().splitlines()
    start = max(0, line_number - radius - 1)
    end = min(len(lines), line_number + radius)
    return "\n".join(
        f"{number:>4}: {lines[number - 1]}".rstrip()
        for number in range(start + 1, end + 1)
    )


def latex_log_checks() -> tuple[dict[str, int], list[dict[str, object]]]:
    log_path = THESIS / "main.log"
    text = log_path.read_text()
    counts = {
        "undefined_references": len(
            re.findall(r"LaTeX Warning: Reference .*? undefined", text)
        ),
        "undefined_citations": len(
            re.findall(r"LaTeX Warning: Citation .*? undefined", text)
        ),
        "overfull_boxes": len(re.findall(r"^Overfull \\hbox", text, re.MULTILINE)),
        "underfull_boxes": len(
            re.findall(r"^Underfull \\hbox", text, re.MULTILINE)
        ),
    }
    details: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not (
            line.startswith("Overfull \\hbox")
            or line.startswith("Underfull \\hbox")
        ):
            continue
        match = re.search(r"at lines (\d+)--(\d+)", line)
        source_line = int(match.group(1)) if match else 0
        next_log_line = text.splitlines()[line_number] if line_number < len(
            text.splitlines()
        ) else ""
        source_path = THESIS / "main.tex"
        if "MLCommons" in next_log_line:
            source_path = THESIS / "main.bbl"
        details.append(
            {
                "log_line": line_number,
                "warning": line,
                "source_file": (
                    source_path.name if source_line else "unresolved"
                ),
                "source_line": source_line,
                "source_context": (
                    source_context(source_path, source_line)
                    if source_line
                    else ""
                ),
            }
        )
    return counts, details


def check_versions() -> dict[str, object]:
    lock_versions = {}
    for line in LOCKFILE.read_text().splitlines():
        if "==" not in line:
            continue
        name, version = line.split("==", 1)
        lock_versions[name.strip()] = version.strip()
    appendix = (THESIS / "chapters" / "appendix_c_reproducibility.tex").read_text()
    loafo_manifest = json.loads(
        (
            ROOT
            / "results"
            / "phase2-loafo-final-1a65664"
            / "results_manifest.json"
        ).read_text()
    )
    python_version = loafo_manifest["software_versions"]["python"]
    mismatches = []
    for name, expected in EXPECTED_VERSIONS.items():
        lock_value = lock_versions.get(name)
        thesis_pattern = {
            "numpy": rf"NumPy {re.escape(expected)}",
            "pandas": rf"pandas {re.escape(expected)}",
            "scikit-learn": rf"scikit-learn {re.escape(expected)}",
            "statsmodels": rf"statsmodels {re.escape(expected)}",
            "scipy": rf"SciPy {re.escape(expected)}",
            "torch": rf"PyTorch {re.escape(expected)}",
            "torch-geometric": rf"PyTorch Geometric {re.escape(expected)}",
        }[name]
        if lock_value != expected or not re.search(thesis_pattern, appendix):
            mismatches.append(
                {
                    "package": name,
                    "expected": expected,
                    "lockfile": lock_value,
                    "present_in_appendix_c": bool(
                        re.search(thesis_pattern, appendix)
                    ),
                }
            )
    if python_version != "3.14.2" or not re.search(
        r"Python 3\.14\.2", appendix
    ):
        mismatches.append(
            {
                "package": "python",
                "expected": "3.14.2",
                "recorded_runtime": python_version,
                "present_in_appendix_c": bool(
                    re.search(r"Python 3\.14\.2", appendix)
                ),
            }
        )
    return {
        "expected_packages": EXPECTED_VERSIONS,
        "expected_python": "3.14.2",
        "lockfile": {
            name: lock_versions.get(name) for name in EXPECTED_VERSIONS
        },
        "recorded_runtime_python": python_version,
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def check_bundle_hashes() -> dict[str, object]:
    rows = []
    for bundle_name in FINAL_BUNDLES:
        bundle = ROOT / "results" / bundle_name
        manifest_path = bundle / "results_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for filename, expected in manifest["output_sha256"].items():
            path = bundle / filename
            actual = sha256_file(path) if path.exists() else None
            rows.append(
                {
                    "bundle": bundle_name,
                    "file": filename,
                    "expected_sha256": expected,
                    "actual_sha256": actual,
                    "match": actual == expected,
                }
            )
        if manifest.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
            rows.append(
                {
                    "bundle": bundle_name,
                    "file": "results_manifest.json:protocol_sha256",
                    "expected_sha256": EXPECTED_PROTOCOL_SHA256,
                    "actual_sha256": manifest.get("protocol_sha256"),
                    "match": False,
                }
            )
    corrections = json.loads(
        (
            ROOT
            / "results"
            / "phase2-5-corrections-e97bb93"
            / "results_manifest.json"
        ).read_text()
    )
    expected_history = corrections["protocol_history_sha256"]
    actual_history = sha256_file(PROTOCOL_HISTORY)
    rows.append(
        {
            "bundle": "phase2-5-corrections-e97bb93",
            "file": "protocol_history.md",
            "expected_sha256": expected_history,
            "actual_sha256": actual_history,
            "match": actual_history == expected_history,
        }
    )
    return {"rows": rows, "pass": all(row["match"] for row in rows)}


def check_generated_assets() -> dict[str, object]:
    directory = THESIS / "generated"
    manifest = json.loads(
        (directory / "generated_asset_manifest.json").read_text()
    )
    rows = []
    for relative, expected in manifest.items():
        actual = sha256_file(directory / relative)
        rows.append(
            {
                "file": relative,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "match": actual == expected,
            }
        )
    return {"rows": rows, "pass": all(row["match"] for row in rows)}


def run_tests() -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (THESIS / "stage3b_test_output.txt").write_text(completed.stdout)
    match = re.search(r"Ran (\d+) tests", completed.stdout)
    tests = int(match.group(1)) if match else None
    return {
        "returncode": completed.returncode,
        "tests": tests,
        "ok": "\nOK\n" in completed.stdout,
        "output": "thesis_overleaf/stage3b_test_output.txt",
        "pass": completed.returncode == 0 and tests == 21 and "\nOK\n" in completed.stdout,
    }


def check_placeholders(pdf_text: str) -> dict[str, object]:
    rows = []
    normalized_pdf = " ".join(pdf_text.split())
    for placeholder in PLACEHOLDERS:
        rows.append(
            {
                "placeholder": placeholder,
                "visible_in_pdf": placeholder in normalized_pdf,
            }
        )
    return {"rows": rows, "pass": all(row["visible_in_pdf"] for row in rows)}


def row_matches(frame: pd.DataFrame, expression: str) -> pd.Series:
    if expression in {"", "all rows", "manifest root"}:
        return pd.Series(True, index=frame.index)
    mask = pd.Series(True, index=frame.index)
    for condition in expression.split(" and "):
        if " == " in condition:
            column, value = condition.split(" == ", 1)
            if column not in frame:
                return pd.Series(False, index=frame.index)
            mask &= frame[column].astype(str).eq(value)
        elif " in " in condition:
            column, values = condition.split(" in ", 1)
            if column not in frame:
                return pd.Series(False, index=frame.index)
            mask &= frame[column].astype(str).isin(
                [value.strip() for value in values.split(",")]
            )
        else:
            return pd.Series(False, index=frame.index)
    return mask


def pdf_numbers(pdf_text: str) -> dict[str, list[str]]:
    lines_by_number: dict[str, list[str]] = {}
    pattern = re.compile(
        r"(?<![\w.])[+\-\u2212]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|"
        r"(?<![\w.])[+\-\u2212]?\d+(?:\.\d+)?"
    )
    for line in pdf_text.splitlines():
        compact = " ".join(line.split())
        for match in pattern.finditer(line):
            normalized = normalized_number(match.group())
            if normalized is not None:
                lines_by_number.setdefault(normalized, []).append(compact)
    return lines_by_number


def spot_check_pdf(
    pdf_text: str,
    *,
    count: int = 15,
    seed: int = 20261108,
) -> pd.DataFrame:
    mapping = pd.read_csv(THESIS / "stage3b_number_source_map.csv")
    observed = pdf_numbers(pdf_text)
    candidates = []
    for row in mapping.itertuples(index=False):
        source_file = str(row.source_file)
        if not source_file.endswith(".csv"):
            continue
        if str(row.derivation) != "direct":
            continue
        number = normalized_number(row.number)
        if number is None:
            continue
        if "." not in number and "," not in str(row.number):
            continue
        if "." in number and len(number.split(".", 1)[1]) < 4:
            continue
        if number not in observed:
            continue
        source_path = ROOT.parent / source_file
        if source_path.exists():
            candidates.append(row)
    if len(candidates) < count:
        raise AssertionError("Not enough direct source-map rows for a spot-check")
    sampled = random.Random(seed).sample(candidates, count)
    rows = []
    for row in sampled:
        source_path = ROOT.parent / str(row.source_file)
        frame = pd.read_csv(source_path)
        column = str(row.source_column)
        if column not in frame:
            raise AssertionError(f"Missing source column {column} in {source_path}")
        selected = frame.loc[row_matches(frame, str(row.row_filter))]
        if selected.empty:
            raise AssertionError(
                f"No source rows satisfy {row.row_filter!r} for {row.map_id}"
            )
        expected = normalized_number(row.number)
        source_values = [
            value
            for value in selected[column].tolist()
            if displayed_source_value(value, expected) == expected
        ]
        if not source_values:
            raise AssertionError(
                f"{row.map_id} value does not occur in {source_path}:{column}"
            )
        contexts = observed.get(expected, [])
        if not contexts:
            raise AssertionError(f"{row.map_id} value is absent from the PDF")
        rows.append(
            {
                "map_id": row.map_id,
                "pdf_value": row.number,
                "pdf_text_extract": contexts[0],
                "source_file": row.source_file,
                "source_column": column,
                "source_row_filter": row.row_filter,
                "source_value": source_values[0],
                "value_match": (
                    displayed_source_value(source_values[0], expected) == expected
                ),
                "present_in_pdf": True,
            }
        )
    output = pd.DataFrame(rows)
    if len(output) != count or not output["value_match"].all():
        raise AssertionError("PDF-to-source spot-check did not verify every row")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-tests",
        action="store_true",
        help="Run the 21-test suite and refresh the transcript.",
    )
    args = parser.parse_args()

    protocol_actual = sha256_file(PROTOCOL)
    protocol_check = {
        "expected_sha256": EXPECTED_PROTOCOL_SHA256,
        "actual_sha256": protocol_actual,
        "pass": protocol_actual == EXPECTED_PROTOCOL_SHA256,
    }
    versions = check_versions()
    bundles = check_bundle_hashes()
    generated = check_generated_assets()
    latex_counts, latex_details = latex_log_checks()
    pdf_text = subprocess.run(
        ["pdftotext", "-layout", str(THESIS / "main.pdf"), "-"],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    placeholders = check_placeholders(pdf_text)
    spotcheck = spot_check_pdf(pdf_text)
    spotcheck.to_csv(THESIS / "stage3c_number_spotcheck.csv", index=False)
    tests = run_tests() if args.run_tests else {
        "pass": None,
        "note": "Run with --run-tests to refresh the transcript.",
    }
    latex_pass = (
        latex_counts["undefined_references"] == 0
        and latex_counts["undefined_citations"] == 0
        and latex_counts["overfull_boxes"] == 0
    )
    result = {
        "protocol": protocol_check,
        "lock_versions": versions,
        "bundle_hashes": bundles,
        "generated_assets": generated,
        "tests": tests,
        "latex": {
            "counts": latex_counts,
            "details": latex_details,
            "pass": latex_pass,
        },
        "placeholders": placeholders,
        "number_spotcheck": {
            "rows": len(spotcheck),
            "path": "thesis_overleaf/stage3c_number_spotcheck.csv",
            "pass": len(spotcheck) == 15 and bool(spotcheck["value_match"].all()),
        },
        "pass": all(
            (
                protocol_check["pass"],
                versions["pass"],
                bundles["pass"],
                generated["pass"],
                latex_pass,
                placeholders["pass"],
                len(spotcheck) == 15 and bool(spotcheck["value_match"].all()),
                tests["pass"] is not False,
            )
        ),
    }
    (THESIS / "stage3c_final_checks.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
