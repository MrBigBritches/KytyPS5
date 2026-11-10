#!/usr/bin/env python3
"""Compares first-run reports written by `--first-run-report`.

Usage: first_run_compare.py <report.json or glob>...

Patterns are expanded here because Windows shells do not expand them. A report that cannot be read
is skipped with a warning. A partial report, written when a title relaunched itself, is skipped too,
since the process that carried on writes its own.

Reports whose strategy flags match are one arm. An arm with several runs shows the median of each
column. Standard library only.

The skip columns count draws. The smp and heap columns count targets, so one draw with several color
targets adds several, and heap color can exceed skip d+c. These columns are signals only. Zero does
not mean safe and nonzero does not mean damage, so a run also has to be watched for lost geometry.
"""
import glob
import json
import statistics
import sys

# Flags that every measured run has on, so they say nothing about the arm.
MEASUREMENT_FLAGS = {"--first-run-report", "--cold-pipelines"}
# Flags that define an arm. These always show in its name, or a set of reports from one arm would be
# labelled baseline. Add each new strategy's flag here.
STRATEGY_FLAGS = {"--async-pipelines", "--async-pipelines-depth-only", "--background-warm-up",
                  "--deferred-pipeline-draws", "--pause-warm-up-for-draws", "--deferred-non-push-draws",
                  "--graphics-pipeline-library", "--generate-shader-store"}
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
    ("skip depth", 10),
    ("skip d+c", 8),
    ("smp depth", 9),
    ("smp color", 9),
    ("heap depth", 10),
    ("heap color", 10),
]


def arm_name(report, common=frozenset()):
    """Names an arm by the flags that were on in it and not in every report, which keeps the label
    short when the launcher adds many base flags to all runs. A report with none of those is the
    baseline. A non-cold run is marked."""
    flags = sorted(name for name, on in report.get("flags", {}).items()
                   if on and name not in MEASUREMENT_FLAGS and name not in common)
    label = " ".join(flags) if flags else "baseline"
    # A run that started from a seeded store warms modules at boot, which no flag shows, so it must
    # not be merged with the same arm started from nothing.
    warmed = report.get("boot", {}).get("modules", 0)
    if warmed:
        label += f" (store {warmed})"
    # A device that cannot link libraries runs the GPL arm as if the flag were off, so that run must
    # not be merged with real GPL runs.
    library = report.get("graphics_pipeline_library", {})
    if report.get("flags", {}).get("--graphics-pipeline-library") and not library.get("active"):
        label += " (gpl inactive)"
    return label if report.get("cold") else label + " (warm)"


def common_flags(reports):
    """The settings on in every report, which can be left out of arm names. Strategy flags are never
    among them, because they say what the arm is even when every report has them."""
    sets = [{name for name, on in report.get("flags", {}).items() if on} for report in reports]
    return frozenset(set.intersection(*sets) - STRATEGY_FLAGS) if sets else frozenset()


def figures(report):
    """Reduces one report to the numbers in the table."""
    blocked = report["play"]["blocked_ms"]
    pauses = report["pauses"]
    boot_s = report["boot"]["warm_up_ms"] / MS_PER_SECOND
    skips = report["skips"]
    by_target = skips.get("by_target_class", {})
    # Only color can be permanent damage in view of the CPU, because a depth clear repairs depth.
    # Heap reads are reported as possible, when a skip meets a heap that references the image or
    # an element later selects it. Heap depth is repaired by a clear like any depth, and heap
    # color is not.
    sampled = skips.get("sampled_after_skip_by_target_class", {})
    return [
        1,
        blocked["graphics"] / MS_PER_SECOND,
        blocked["compute"] / MS_PER_SECOND,
        report["play"]["translate_ms"] / MS_PER_SECOND,
        boot_s,
        boot_s + blocked["total"] / MS_PER_SECOND,
        pauses["longest_pause_ms"] / MS_PER_SECOND,
        sum(pauses["pause_histogram"].get(bucket, 0) for bucket in LONG_PAUSE_BUCKETS),
        by_target.get("depth_only", 0),
        by_target.get("depth_and_color", 0),
        sampled.get("depth", 0),
        sampled.get("color", 0),
        sampled.get("heap_depth", 0),
        sampled.get("heap_color", 0),
    ]


def summarise(rows):
    """Takes the median of each column across the runs of one arm."""
    summary = [statistics.median(column) for column in zip(*rows)]
    summary[0] = len(rows)
    return summary


def format_cell(value, index):
    """Counts print as integers and times with two decimals."""
    return str(int(value)) if COLUMNS[index][0] in ("runs", "pauses>1s", "skip depth", "skip d+c", "smp depth", "smp color", "heap depth", "heap color") else f"{value:.2f}"


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


def print_refusals(refusals, gpu_translate):
    """Lists, for each arm, why draws that missed their pipeline were refused a skip. Each reason
    shows the median and the maximum of the draws and of the blocked seconds across the arm's
    runs. Refused time plus the GPU thread's graphics translate makes up gfx blocked, and the
    translate column can't show that because it covers every thread and both kinds."""
    for name in sorted(refusals):
        reasons = sorted({reason for run in refusals[name] for reason in run})
        if not reasons:
            continue
        print()
        print(f"{name}: draws refused a skip, by rule (median and max of draws, blocked s)")
        for reason in reasons:
            draws = [run.get(reason, {}).get("draws", 0) for run in refusals[name]]
            blocked = [run.get(reason, {}).get("blocked_ms", 0.0) / MS_PER_SECOND
                       for run in refusals[name]]
            print(f"  {reason:<30} {int(statistics.median(draws)):>8} {max(draws):>8}"
                  f" {statistics.median(blocked):>10.2f} {max(blocked):>10.2f}")
        translate = statistics.median(gpu_translate[name]) / MS_PER_SECOND
        print(f"  GPU-thread graphics translate, also blocked: {translate:.2f} s")


def main(patterns):
    arms = {}
    refusals = {}
    gpu_translate = {}
    reports = [report for report in (load(path) for path in expand(patterns)) if report is not None]
    common = common_flags(reports)
    for report in reports:
        name = arm_name(report, common)
        arms.setdefault(name, []).append(figures(report))
        refusals.setdefault(name, []).append(report["skips"].get("refused", {}))
        gpu_translate.setdefault(name, []).append(report["play"].get("translate_gpu_graphics_ms", 0.0))
    if not arms:
        print("no usable reports", file=sys.stderr)
        return 1
    shared = sorted(common - MEASUREMENT_FLAGS)
    if shared:
        print("Flags on in every report: " + " ".join(shared))
    name_width = max([len("arm")] + [len(name) for name in arms])
    header = "arm".ljust(name_width) + "  " + "  ".join(title.rjust(width) for title, width in COLUMNS)
    print(header)
    print("-" * len(header))
    for name in sorted(arms):
        cells = [format_cell(value, index) for index, value in enumerate(summarise(arms[name]))]
        print(name.ljust(name_width) + "  " +
              "  ".join(cell.rjust(width) for cell, (_, width) in zip(cells, COLUMNS)))
    print_refusals(refusals, gpu_translate)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
