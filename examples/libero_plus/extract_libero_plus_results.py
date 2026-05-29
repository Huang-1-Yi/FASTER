#!/usr/bin/env python3
"""Format LIBERO-Plus eval results as a Google-Sheets-friendly table."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "libero_plus"

ROW_ORDER = [
    ("Spatial", ("libero_spatial",)),
    ("Object", ("libero_object",)),
    ("Goal", ("libero_goal",)),
    ("Long", ("libero_10", "libero_long")),
]

COL_ORDER = [
    ("Camera", "Camera Viewpoints"),
    ("Robot", "Robot Initial States"),
    ("Language", "Language Instructions"),
    ("Light", "Light Conditions"),
    ("Background", "Background Textures"),
    ("Noise", "Sensor Noise"),
    ("Layout", "Objects Layout"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract LIBERO-Plus JSON results into one fixed table."
    )
    parser.add_argument(
        "path",
        help=f"Result JSON file or directory, for example: {DEFAULT_RESULTS_DIR}",
    )
    parser.add_argument(
        "--format",
        choices=("tsv", "csv"),
        default="tsv",
        help="Output format. TSV is better for pasting into Google Sheets.",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output file path. Use '-' for stdout.",
    )
    return parser.parse_args()


def collect_json_paths(raw_path: str) -> list[pathlib.Path]:
    paths = [raw_path]
    json_paths: set[pathlib.Path] = set()

    for raw_path in paths:
        path = pathlib.Path(raw_path).expanduser()
        if not path.exists():
            print(f"[WARN] Path does not exist, skip: {path}", file=sys.stderr)
            continue
        if path.is_file():
            if path.suffix == ".json":
                json_paths.add(path.resolve())
            continue
        for json_path in path.rglob("*.json"):
            json_paths.add(json_path.resolve())

    return sorted(json_paths)


def infer_suite(path: pathlib.Path, data: dict) -> str | None:
    meta = data.get("meta")
    if isinstance(meta, dict) and meta.get("task_suite_name"):
        return str(meta["task_suite_name"])

    if path.stem.startswith("results_"):
        return path.parent.name

    stem = path.stem
    known = {suite for _, suites in ROW_ORDER for suite in suites}
    if stem in known:
        return stem
    return None


def infer_category(path: pathlib.Path) -> str:
    if path.stem.startswith("results_"):
        return path.stem[len("results_") :].replace("_", " ")
    return "ALL"


def extract_rate(data: dict) -> float | None:
    success = data.get("success")
    failure = data.get("failure")
    if isinstance(success, list) and isinstance(failure, list):
        total = len(success) + len(failure)
        if total == 0:
            return None
        return len(success) / total

    running_counts = data.get("running_counts")
    if isinstance(running_counts, dict):
        total = running_counts.get("total_episodes")
        successes = running_counts.get("total_successes")
        if isinstance(total, int) and isinstance(successes, int) and total > 0:
            return successes / total

    return None


def load_scores(json_paths: list[pathlib.Path]) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], tuple[float, float]] = {}

    for path in json_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[WARN] Failed to read {path}: {exc}", file=sys.stderr)
            continue

        if not isinstance(data, dict):
            continue

        suite = infer_suite(path, data)
        category = infer_category(path)
        rate = extract_rate(data)
        if suite is None or rate is None:
            continue

        key = (suite, category)
        mtime = path.stat().st_mtime
        previous = scores.get(key)
        if previous is None or mtime >= previous[1]:
            scores[key] = (rate, mtime)

    return {key: value for key, (value, _) in scores.items()}


def format_pct(rate: float | None) -> str:
    if rate is None:
        return ""
    return f"{rate * 100:.1f}".rstrip("0").rstrip(".")


def pick_row_score(
    scores: dict[tuple[str, str], float],
    suite_names: tuple[str, ...],
    category: str,
) -> float | None:
    for suite_name in suite_names:
        rate = scores.get((suite_name, category))
        if rate is not None:
            return rate
    return None


def build_table(scores: dict[tuple[str, str], float]) -> list[list[str]]:
    header = [""] + [label for label, _ in COL_ORDER]

    rows: list[list[str]] = [header]

    for row_label, suite_names in ROW_ORDER:
        row = [row_label]
        for _, category in COL_ORDER:
            rate = pick_row_score(scores, suite_names, category)
            row.append(format_pct(rate))
        rows.append(row)

    return rows


def write_table(rows: list[list[str]], output_path: str, fmt: str) -> None:
    delimiter = "\t" if fmt == "tsv" else ","

    if output_path == "-":
        handle = sys.stdout
        close_handle = False
    else:
        handle = open(output_path, "w", encoding="utf-8", newline="")
        close_handle = True

    try:
        writer = csv.writer(handle, delimiter=delimiter, lineterminator="\n")
        writer.writerows(rows)
    finally:
        if close_handle:
            handle.close()


def main() -> int:
    args = parse_args()
    json_paths = collect_json_paths(args.path)
    if not json_paths:
        print("No JSON files found.", file=sys.stderr)
        return 1

    scores = load_scores(json_paths)
    if not scores:
        print("No valid LIBERO-Plus result JSON found.", file=sys.stderr)
        return 1

    rows = build_table(scores)
    write_table(rows, args.output, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
