#!/usr/bin/env python3
"""Compares first-run reports written by `--first-run-report`.

Usage: first_run_compare.py <report.json or glob>...

Patterns are expanded here because Windows shells do not expand them. A report that cannot be read
is skipped with a warning. A partial report, written when a title relaunched itself, is skipped too,
since the process that carried on writes its own.

Reports whose strategy flags match are one arm. An arm with several runs shows the median of each
column. Standard library only.
"""
import glob
import json
import statistics
import sys

# Flags that every measured run has on, so they say nothing about the arm.
MEASUREMENT_FLAGS = {"--first-run-report", "--cold-pipelines"}
# Histogram buckets whose lower bound is at least one second.
LONG_PAUSE_BUCKETS = ("1000", "2500", "5000", "10000")
MS_PER_SECOND = 1000.0

COLUMNS = [
    ("runs", 4),
    ("gfx blocked s", 13),
    ("cs blocked s", 12),
    ("translate s", 11),
    ("boot s", 7),
    ("total wait s", 12),
    ("longest s", 9),
    ("pauses>1s", 9),
    ("skips", 6),
]


def arm_name(report):
    """Names an arm by the strategy flags that were on, and marks a non-cold run."""
    flags = sorted(name for name, on in report.get("flags", {}).items()
                   if on and name not in MEASUREMENT_FLAGS)
    label = " ".join(flags) if flags else "baseline"
    return label if report.get("cold") else label + " (warm)"


def figures(report):
    """Reduces one report to the numbers in the table."""
    blocked = report["play"]["blocked_ms"]
    pauses = report["pauses"]
    boot_s = report["boot"]["warm_up_ms"] / MS_PER_SECOND
    skips = report["skips"]
    return [
        1,
        blocked["graphics"] / MS_PER_SECOND,
        blocked["compute"] / MS_PER_SECOND,
        report["play"]["translate_ms"] / MS_PER_SECOND,
        boot_s,
        boot_s + blocked["total"] / MS_PER_SECOND,
        pauses["longest_pause_ms"] / MS_PER_SECOND,
        sum(pauses["pause_histogram"].get(bucket, 0) for bucket in LONG_PAUSE_BUCKETS),
        skips["draw"] + skips["dispatch"],
    ]


def summarise(rows):
    """Takes the median of each column across the runs of one arm."""
    summary = [statistics.median(column) for column in zip(*rows)]
    summary[0] = len(rows)
    return summary


def format_cell(value, index):
    """Counts print as integers and times with two decimals."""
    return str(int(value)) if COLUMNS[index][0] in ("runs", "pauses>1s", "skips") else f"{value:.2f}"


def expand(patterns):
    """Expands each pattern, and keeps one that matches nothing so it is reported as missing."""
    paths = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])
    return paths


def load(path):
    """Reads one report, or returns None with a warning when it cannot be used."""
    try:
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("partial"):
            print(f"warning: skipping {path}: partial report from a relaunch", file=sys.stderr)
            return None
        figures(report)
        return report
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"warning: skipping {path}: {error}", file=sys.stderr)
        return None


def main(patterns):
    arms = {}
    for path in expand(patterns):
        report = load(path)
        if report is not None:
            arms.setdefault(arm_name(report), []).append(figures(report))
    if not arms:
        print("no usable reports", file=sys.stderr)
        return 1
    name_width = max([len("arm")] + [len(name) for name in arms])
    header = "arm".ljust(name_width) + "  " + "  ".join(title.rjust(width) for title, width in COLUMNS)
    print(header)
    print("-" * len(header))
    for name in sorted(arms):
        cells = [format_cell(value, index) for index, value in enumerate(summarise(arms[name]))]
        print(name.ljust(name_width) + "  " +
              "  ".join(cell.rjust(width) for cell, (_, width) in zip(cells, COLUMNS)))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
