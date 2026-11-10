#!/usr/bin/env python3
"""Compares first-run reports written by `--first-run-report`.

Usage: first_run_compare.py [--gate <gpl arm>] <report.json or glob>...

After the main table come the Generation, Audio and Contention tables, each only for an arm whose
reports carry the data. A set of older reports prints the main table alone, as before.

With --gate the named gpl arm is the baseline for the pass criteria of the shader store generation.
A second boot is a report that started from a seeded store, and the P8 baseline is the gpl second
boot, not the gen first run. P5 cuts both arms at the gen arm's median supervisor.finished_at_ms,
which is on the same clock as an underrun's at_ms. supervisor.finished_ms is a duration from the
start of generation and is never used as a cut.

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
import math
import re
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


def arm_name(report, common=frozenset(), with_store=True):
    """Names an arm by the flags that were on in it and not in every report, which keeps the label
    short when the launcher adds many base flags to all runs. A report with none of those is the
    baseline. A non-cold run is marked. With_store false leaves out the seeded store count, so the
    first run and the second boot of one arm share a name."""
    flags = sorted(name for name, on in report.get("flags", {}).items()
                   if on and name not in MEASUREMENT_FLAGS and name not in common)
    label = " ".join(flags) if flags else "baseline"
    # A run that started from a seeded store warms modules at boot, which no flag shows, so it must
    # not be merged with the same arm started from nothing.
    warmed = report.get("boot", {}).get("modules", 0)
    if warmed and with_store:
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


# The flag that turns on shader store generation.
GENERATE_FLAG = "--generate-shader-store"
# A first use that waited or compiled inline for longer than this is a slow take.
SLOW_TAKE_MS = 250.0
BYTES_PER_MB = 1024 * 1024
P90_FRACTION = 0.90
P95_FRACTION = 0.95
SOURCES_DIFFER = "n/a (sources differ)"
# The emulator's own peaks, in the order they are preferred. Each is a different measure, so the
# source name travels with the figure.
PROCESS_PEAK_KEYS = ("peak_working_set_bytes", "peak_commit_bytes")
# Warm builds running when a graphics compile began, grouped as name, lowest and highest.
WARM_RUNNING_BUCKETS = (("0", 0, 0), ("1-2", 1, 2), ("3+", 3, None))

# Limits of the pass criteria. P1 to P4, P6, P7 and P9 are from the audio review of step 6c.
P1_MEAN_HELD_S = 15.0
P1_RUN_HELD_S = 20.0
P1_HARD_FRACTION = 0.5
P4_EXTRA_SLOW_TAKES = 2
P4_STOPPED_RATIO = 1.25
P5_MIN_EXTRA_UNDERRUNS = 2
P5_EXTRA_FRACTION = 0.25
P6_DRAIN_MAX_MS = 2.0
P6_DRAIN_HARD_MS = 8.0
P6_OVER_1MS_HANDFUL = 5
P7_DECODE_MS = 20.0
P8_GFX_ALLOWANCE = 0.10
GFX_FIRST_RUN_ALLOWANCE = 0.15


def hash_text(value):
    """Hashes, ids and order values are text. They can be all decimal digits or lead with zeros, so
    they are never parsed as numbers."""
    return str(value)


def percentile(values, fraction):
    """Nearest-rank percentile, or None when there are no values."""
    ordered = sorted(values)
    if not ordered:
        return None
    # The rounding keeps 0.9 of 10 from landing on 9.000000000000002 and ceiling to 10.
    return ordered[max(0, math.ceil(round(fraction * len(ordered), 9)) - 1)]


def median_of(values):
    """Median of the values that are there, or None when none are."""
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def generation_block(report):
    """The shader store generation block when it ran, and None when it was off or gated off."""
    block = report.get("shader_store_generation")
    return block if isinstance(block, dict) and "supervisor" in block else None


def window_figures(report):
    """Worst and 95th percentile held time per deferred-resolve window, and the holds per window."""
    resolves = report.get("deferred_draws", {}).get("resolves", [])
    held = [row.get("window_held_ms", 0.0) for row in resolves]
    changes = [row.get("window_hold_count_change", 0) for row in resolves]
    return {"win_max": max(held, default=None), "win_p95": percentile(held, P95_FRACTION),
            "hold_max": max(changes, default=None),
            "hold_mean": statistics.mean(changes) if changes else None}


def process_peak(report):
    """The emulator's peak as a figure and the source it came from, or None."""
    process = report.get("process", {})
    for key in PROCESS_PEAK_KEYS:
        if key in process:
            return (process[key], key)
    return None


def take_figures(report, generated_ids):
    """Counts the slow first uses split into generated modules and the player's own, and the
    longest wait of any take."""
    takes = report.get("warm_takes", [])
    slow = [take for take in takes if take.get("how") in ("waited", "inline")
            and take.get("stopped_ms", 0.0) > SLOW_TAKE_MS]
    slow_generated = sum(1 for take in slow if hash_text(take.get("id")) in generated_ids)
    waited = sum(1 for take in slow if take.get("how") == "waited")
    return {"slow_gen": slow_generated, "slow_player": len(slow) - slow_generated,
            "slow_waited": waited, "slow_inline": len(slow) - waited,
            "max_stopped": max((take.get("stopped_ms", 0.0) for take in takes), default=0.0)}


def contention_figures(report):
    """Sums the graphics compiles of the compile workers. Only graphics rows count. A warm compute
    worker's compile row sits inside its own warm build, so it counts itself in warm_running and
    would put the warm-up's own work in the contention figure. Rows from before the context fields
    existed have no at_ms and are left out."""
    rows = [row for row in report.get("compiles", []) if row.get("thread") == "worker"
            and row.get("kind") == "graphics" and "at_ms" in row]
    if not rows:
        return None
    figures_by_name = {"n": len(rows), "sum": sum(row["create_ms"] for row in rows),
                       "p90": percentile([row["create_ms"] for row in rows], P90_FRACTION)}
    groups = {"alive": [row for row in rows if row.get("child_alive")],
              "idle": [row for row in rows if not row.get("child_alive")]}
    for name, low, high in WARM_RUNNING_BUCKETS:
        groups["wr " + name] = [row for row in rows if row.get("warm_running", 0) >= low
                                and (high is None or row.get("warm_running", 0) <= high)]
    for name, group in groups.items():
        figures_by_name[name + " n"] = len(group)
        figures_by_name[name + " sum"] = sum(row["create_ms"] for row in group)
    return figures_by_name


def generation_figures(block):
    """Reduces a generation block to the numbers of the Generation table."""
    supervisor = block["supervisor"]
    drain = block.get("drain", {})
    modules = block.get("generated_modules", [])
    how = [module.get("how") for module in modules]
    classes = block.get("runtime_misses", {}).get("classes", {})
    peak = None
    if "peak_bytes" in supervisor:
        peak = (supervisor["peak_bytes"], supervisor.get("peak_source", "unknown"))
    return {"spawns": supervisor.get("spawns"), "restarts": supervisor.get("restarts"),
            "finished_ms": supervisor.get("finished_ms"),
            "finished_at_ms": supervisor.get("finished_at_ms"), "posted": supervisor.get("posted"),
            "decode_ms_max": supervisor.get("decode_ms_max"), "peak": peak,
            "drain_max": drain.get("max_ms"), "drain_over1": drain.get("over_1ms"),
            "drain_over4": drain.get("over_4ms"),
            "plan_mismatch": drain.get("plan_mismatch"),
            "done": how.count("done"), "waited": how.count("waited"), "inline": how.count("inline"),
            "unused": how.count("unused"), "unused_build_ms": block.get("unused_build_ms"),
            "order": block.get("worker", {}).get("order"),
            "classes": {name: (row.get("count", 0), row.get("create_ms", 0.0))
                        for name, row in classes.items()}}


def underrun_times(report):
    """The underrun times in milliseconds, or None when the report has no audio_underruns."""
    underruns = report.get("audio_underruns")
    return None if underruns is None else list(underruns.get("at_ms", []))


def run_metrics(report):
    """Everything the new tables and the gate read from one report."""
    block = generation_block(report)
    generated_ids = {hash_text(module.get("id")) for module in block.get("generated_modules", [])} \
        if block else set()
    metrics = {"held_s": report.get("audio_held_ms", 0.0) / MS_PER_SECOND,
               "holds": report.get("audio_holds", 0), "underruns": underrun_times(report),
               "gfx_s": report["play"]["blocked_ms"]["graphics"] / MS_PER_SECOND,
               "second_boot": bool(report.get("boot", {}).get("modules", 0)),
               "gen": generation_figures(block) if block else None,
               "contention": contention_figures(report), "proc_peak": process_peak(report),
               "machine": report.get("machine")}
    metrics.update(window_figures(report))
    metrics.update(take_figures(report, generated_ids))
    return metrics


def arm_peak(pairs):
    """One peak for an arm as a figure and its source. Runs that measured different things give no
    figure, with the source None, because the numbers can't be set side by side."""
    pairs = [pair for pair in pairs if pair is not None]
    if not pairs:
        return None
    sources = {source for _, source in pairs}
    if len(sources) > 1:
        return (None, None)
    return (statistics.median(value for value, _ in pairs), sources.pop())


def peak_text(peak):
    """A peak with its source next to it, or a dash with no data."""
    if peak is None:
        return "-"
    if peak[1] is None:
        return SOURCES_DIFFER
    return f"{peak[0] / BYTES_PER_MB:.1f} MB ({peak[1]})"


def compare_peaks(first, second):
    """Sets two peaks side by side, only when they measure the same thing."""
    if first is None or second is None:
        return "n/a (no data)"
    if first[1] is None or second[1] is None or first[1] != second[1]:
        return SOURCES_DIFFER
    return (f"{first[0] / BYTES_PER_MB:.1f} MB vs {second[0] / BYTES_PER_MB:.1f} MB ({first[1]})")


def fmt_num(value, decimals=2):
    """A figure with fixed decimals, or a dash when it is missing."""
    return "-" if value is None else f"{value:.{decimals}f}"


def fmt_count(value):
    """A count as an integer, or a dash when it is missing."""
    return "-" if value is None else str(int(value))


def med(runs, key):
    """Median of one metric over the runs of an arm."""
    return median_of(run[key] for run in runs)


def print_table(title, columns, rows):
    """Prints a titled table of arm rows. Each row is a name and one cell per column."""
    name_width = max([len("arm")] + [len(name) for name, _ in rows])
    header = "arm".ljust(name_width) + "  " + "  ".join(text.rjust(width) for text, width in columns)
    print()
    print(title)
    print(header)
    print("-" * len(header))
    for name, cells in rows:
        print(name.ljust(name_width) + "  " +
              "  ".join(cell.rjust(width) for cell, (_, width) in zip(cells, columns)))


def count_around(times, cut_ms):
    """Underruns before the cut and from the cut on, so a run is split at one wall time."""
    before = sum(1 for at_ms in times if at_ms < cut_ms)
    return before, len(times) - before


def split_cells(runs, cut_ms_of):
    """Median underruns before and after each run's cut, or dashes when the runs have no cut."""
    counts = [count_around(run["underruns"], cut_ms_of(run)) for run in runs
              if cut_ms_of(run) is not None]
    return [fmt_count(median_of(count[index] for count in counts)) for index in (0, 1)]


def audio_table(arms):
    """The Audio table, for arms whose reports carry audio_underruns."""
    columns = [("held s", 8), ("holds", 6), ("underruns", 9), ("before fin", 10), ("after fin", 9),
               ("slow gen", 8), ("slow player", 11), ("max stop ms", 11), ("win max ms", 10),
               ("win p95 ms", 10), ("holds/win max", 13), ("holds/win mean", 14)]
    rows = []
    for name in sorted(arms):
        runs = [run for run in arms[name] if run["underruns"] is not None]
        if not runs:
            continue
        # The cut must be on the clock of at_ms, so it is finished_at_ms and never the duration.
        cut = lambda run: run["gen"]["finished_at_ms"] if run["gen"] else None
        cells = [fmt_num(med(runs, "held_s")), fmt_count(med(runs, "holds")),
                 fmt_count(median_of(len(run["underruns"]) for run in runs))]
        cells += split_cells(runs, cut)
        cells += [fmt_count(med(runs, "slow_gen")), fmt_count(med(runs, "slow_player")),
                  fmt_num(med(runs, "max_stopped"), 1), fmt_num(med(runs, "win_max"), 1),
                  fmt_num(med(runs, "win_p95"), 1), fmt_count(med(runs, "hold_max")),
                  fmt_num(med(runs, "hold_mean"))]
        rows.append((name, cells))
    if rows:
        print_table("Audio (slow = waited or inline over 250 ms, fin = generation finished)", columns, rows)


def pair_cell(runs, count_key):
    """A count and a total in seconds, as median over the runs."""
    contention = [run["contention"] for run in runs]
    count = median_of(row[count_key + " n"] for row in contention)
    total = median_of(row[count_key + " sum"] / MS_PER_SECOND for row in contention)
    return f"{fmt_count(count)} / {fmt_num(total)}"


def contention_table(arms):
    """The Contention table. Only the graphics rows of the compile workers count in it."""
    columns = [("n", 4), ("sum s", 8), ("p90 ms", 8), ("child alive n / s", 17), ("child gone n / s", 16)]
    columns += [(f"warm {name} n / s", 16) for name, _, _ in WARM_RUNNING_BUCKETS]
    columns += [("emulator peak", 30)]
    rows = []
    machines = []
    for name in sorted(arms):
        runs = [run for run in arms[name] if run["contention"] is not None]
        if not runs:
            continue
        contention = [run["contention"] for run in runs]
        cells = [fmt_count(median_of(row["n"] for row in contention)),
                 fmt_num(median_of(row["sum"] / MS_PER_SECOND for row in contention)),
                 fmt_num(median_of(row["p90"] for row in contention), 1),
                 pair_cell(runs, "alive"), pair_cell(runs, "idle")]
        cells += [pair_cell(runs, "wr " + bucket) for bucket, _, _ in WARM_RUNNING_BUCKETS]
        cells += [peak_text(arm_peak(run["proc_peak"] for run in runs))]
        rows.append((name, cells))
        machine = next((run["machine"] for run in runs if run["machine"]), None)
        if machine:
            machines.append((name, machine))
    if not rows:
        return
    print_table("Contention (graphics compiles of the compile workers)", columns, rows)
    for name, machine in machines:
        print(f"  {name}: {machine.get('cpu_model', 'unknown')}, {machine.get('logical_cores', 0)} logical cores,"
              f" {machine.get('warm_workers', 0)} warm workers,"
              f" {machine.get('ram_bytes', 0) / BYTES_PER_MB / 1024:.1f} GB RAM")


def generation_extras(name, runs):
    """The worker order and the miss classes of one arm, under the Generation table."""
    orders = [run["gen"]["order"] for run in runs if run["gen"]["order"] is not None]
    if orders:
        print(f"  {name}: worker order {hash_text(orders[0])}")
    classes = sorted({cls for run in runs for cls in run["gen"]["classes"]})
    if not classes:
        return
    print(f"  {name}: runtime misses by class (median count, median pipeline create ms, not translation time)")
    for cls in classes:
        counts = [run["gen"]["classes"].get(cls, (0, 0.0))[0] for run in runs]
        create = [run["gen"]["classes"].get(cls, (0, 0.0))[1] for run in runs]
        print(f"    {cls:<24} {int(statistics.median(counts)):>6} {statistics.median(create):>10.2f}")


def generation_table(arms):
    """The Generation table, for arms whose reports carry the generation block."""
    columns = [("spawns", 6), ("restarts", 8), ("gen duration s", 14), ("posted", 6), ("drain max ms", 12),
               ("drain>1ms", 9), ("drain>4ms", 9), ("plan mismatch", 13), ("done", 4), ("waited", 6), ("inline", 6),
               ("unused", 6), ("unused build s", 14), ("child peak", 30)]
    rows = []
    for name in sorted(arms):
        runs = [run for run in arms[name] if run["gen"] is not None]
        if not runs:
            continue
        gen = [run["gen"] for run in runs]
        finished = median_of(item["finished_ms"] for item in gen)
        unused_build = median_of(item["unused_build_ms"] for item in gen)
        cells = [fmt_count(median_of(item["spawns"] for item in gen)),
                 fmt_count(median_of(item["restarts"] for item in gen)),
                 fmt_num(None if finished is None else finished / MS_PER_SECOND),
                 fmt_count(median_of(item["posted"] for item in gen)),
                 fmt_num(median_of(item["drain_max"] for item in gen)),
                 fmt_count(median_of(item["drain_over1"] for item in gen)),
                 fmt_count(median_of(item["drain_over4"] for item in gen)),
                 fmt_count(median_of(item["plan_mismatch"] for item in gen))]
        cells += [fmt_count(median_of(item[key] for item in gen)) for key in ("done", "waited", "inline", "unused")]
        cells += [fmt_num(None if unused_build is None else unused_build / MS_PER_SECOND),
                  peak_text(arm_peak(item["peak"] for item in gen))]
        rows.append((name, cells))
    if not rows:
        return
    print_table("Generation", columns, rows)
    for name in sorted(arms):
        runs = [run for run in arms[name] if run["gen"] is not None]
        if runs:
            generation_extras(name, runs)


def strip_store(label):
    """An arm label without its seeded store count."""
    return re.sub(r" \(store \d+\)", "", label)


def resolve_gpl_arm(names, wanted):
    """Finds the gpl arm by its label, ignoring the store count. An exact label wins, and a part of
    a label has to be unique. Returns None when nothing, or more than one arm, fits."""
    labels = sorted({strip_store(name) for name in names})
    wanted = strip_store(wanted)
    if wanted in labels:
        return wanted
    partial = [label for label in labels if wanted in label]
    return partial[0] if len(partial) == 1 else None


def gate_runs(records, common, gpl_label):
    """Sorts the runs into the four groups the criteria compare. A second boot started from a seeded
    store. The gen runs are every run with the generation flag on that is not the gpl arm."""
    groups = {"gen1": [], "gen2": [], "gpl1": [], "gpl2": [], "gpl2_warm": []}
    for report, metrics in records:
        name = arm_name(report, common, with_store=False)
        is_gpl = name == gpl_label
        is_gen = bool(report.get("flags", {}).get(GENERATE_FLAG)) and not is_gpl
        # A second boot is told by the seeded store count in the report. A first run that was
        # seeded by hand (for example with -SeedFolder) has a count too, so it reads as a second boot.
        if name == gpl_label + " (warm)" and metrics["second_boot"]:
            groups["gpl2_warm"].append(metrics)
        if not (is_gpl or is_gen):
            continue
        key = ("gpl" if is_gpl else "gen") + ("2" if metrics["second_boot"] else "1")
        groups[key].append(metrics)
    return groups


def verdict(passed):
    """PASS or FAIL, or N/A when the figures were missing."""
    return "N/A" if passed is None else ("PASS" if passed else "FAIL")


def at_most(value, limit):
    """Whether value is no more than the limit, or None when either is missing."""
    return None if value is None or limit is None else value <= limit


def window_checks(gen, gpl):
    """P3, where each window figure of gen must be no higher than gpl."""
    lines = []
    for code, key, label in (("P3a", "win_max", "window_held max ms"), ("P3b", "win_p95", "window_held p95 ms"),
                             ("P3c", "hold_max", "holds per window max"), ("P3d", "hold_mean", "holds per window mean")):
        gen_value, gpl_value = med(gen, key), med(gpl, key)
        lines.append((code, verdict(at_most(gen_value, gpl_value)),
                      f"{label}, gen {fmt_num(gen_value)} vs gpl {fmt_num(gpl_value)}"))
    return lines


def audio_checks(gen, gpl):
    """P1, P2 and P4 on the first runs."""
    held = [run["held_s"] for run in gen]
    gpl_held = med(gpl, "held_s")
    hard = gpl_held is not None and any(value >= P1_HARD_FRACTION * gpl_held for value in held)
    passed = statistics.mean(held) <= P1_MEAN_HELD_S and max(held) < P1_RUN_HELD_S
    lines = [("P1", "HARD FAIL" if hard else verdict(passed),
              f"gen audio_held mean {statistics.mean(held):.2f} s, max {max(held):.2f} s, gpl {fmt_num(gpl_held)} s")]
    lines.append(("P2", verdict(at_most(med(gen, "holds"), med(gpl, "holds"))),
                  f"holds, gen {fmt_num(med(gen, 'holds'), 1)} vs gpl {fmt_num(med(gpl, 'holds'), 1)}"))
    gen_slow, gpl_slow = med(gen, "slow_waited"), med(gpl, "slow_waited")
    limit = None if gpl_slow is None else gpl_slow + P4_EXTRA_SLOW_TAKES
    lines.append(("P4a", verdict(at_most(gen_slow, limit)),
                  f"waited takes over 250 ms, gen {fmt_num(gen_slow, 1)} vs gpl {fmt_num(gpl_slow, 1)} + {P4_EXTRA_SLOW_TAKES}"))
    gen_stop, gpl_stop = med(gen, "max_stopped"), med(gpl, "max_stopped")
    limit = None if gpl_stop is None else P4_STOPPED_RATIO * gpl_stop
    lines.append(("P4b", verdict(at_most(gen_stop, limit)),
                  f"max stopped ms, gen {fmt_num(gen_stop, 1)} vs gpl {fmt_num(gpl_stop, 1)} x {P4_STOPPED_RATIO}"))
    lines.append(("P4c", "INFO", f"generated modules among slow takes, median {fmt_num(med(gen, 'slow_gen'), 1)}"
                                 " (each needs an explanation)"))
    lines.append(("P4d", "INFO", f"inline takes over 250 ms, gen {fmt_num(med(gen, 'slow_inline'), 1)}"
                                 f" vs gpl {fmt_num(med(gpl, 'slow_inline'), 1)} (not part of P4a)"))
    return lines


def p5_lines(gen, gpl):
    """P5, with both arms cut at the gen arm's median finished_at_ms, which is on the clock of the
    underruns' at_ms. finished_ms is a duration from the start of generation and is never a fallback,
    so a report without finished_at_ms is not judged. In each window the gen count may exceed the gpl
    count by 2 or 25 percent, whichever is more."""
    cut = median_of(run["gen"]["finished_at_ms"] for run in gen if run["gen"])
    gen_runs = [run for run in gen if run["underruns"] is not None]
    gpl_runs = [run for run in gpl if run["underruns"] is not None]
    if cut is None:
        return [("P5", "N/A", "no supervisor.finished_at_ms in the gen runs (an older report, or generation "
                              "did not finish), and finished_ms is a duration that can't cut the window")]
    if not gen_runs or not gpl_runs:
        return [("P5", "N/A", "needs audio_underruns in both arms")]
    lines = []
    for index, window in enumerate(("before", "after")):
        gen_count = median_of(count_around(run["underruns"], cut)[index] for run in gen_runs)
        gpl_count = median_of(count_around(run["underruns"], cut)[index] for run in gpl_runs)
        limit = gpl_count + max(P5_MIN_EXTRA_UNDERRUNS, P5_EXTRA_FRACTION * gpl_count)
        lines.append((f"P5{'ab'[index]}", verdict(gen_count <= limit),
                      f"underruns {window} cut {cut:.0f} ms, gen {gen_count:g} vs gpl {gpl_count:g}, limit {limit:g}"))
    return lines


def second_boot_checks(gen2, gpl2, gpl2_warm=()):
    """P8, where the baseline is the gpl second boot and never the gen first run. A gpl second boot
    without --cold-pipelines is another arm, labelled warm, so it is not taken as the baseline and
    the reason is said."""
    if not gen2 or not gpl2:
        if gen2 and gpl2_warm:
            return [("P8", "N/A", "the gpl second boot ran without --cold-pipelines, so it is in a warm arm "
                                  "and is not a like-for-like baseline")]
        return [("P8", "N/A", "needs a gen second boot and a gpl second boot")]
    lines = []
    for code, key, label in (("P8a", "held_s", "audio_held s"), ("P8b", "win_max", "window_held max ms")):
        gen_value, gpl_value = med(gen2, key), med(gpl2, key)
        lines.append((code, verdict(at_most(gen_value, gpl_value)),
                      f"{label}, gen second boot {fmt_num(gen_value)} vs gpl second boot {fmt_num(gpl_value)}"))
    gen_gfx, gpl_gfx = med(gen2, "gfx_s"), med(gpl2, "gfx_s")
    lines.append(("P8c", verdict(at_most(gen_gfx, gpl_gfx * (1 + P8_GFX_ALLOWANCE))),
                  f"graphics blocked s, gen second boot {gen_gfx:.2f} vs gpl second boot {gpl_gfx:.2f} + 10%"))
    return lines


def drain_checks(gen_all):
    """P6 and P7 over every gen run, second boots included."""
    drains = [run["gen"] for run in gen_all if run["gen"] and run["gen"]["drain_max"] is not None]
    if not drains:
        return [("P6", "N/A", "no drain figures")]
    worst = max(item["drain_max"] for item in drains)
    # P6 gates on the calls over 1 ms. The calls over 4 ms are shown for reading only.
    over1 = max(item["drain_over1"] or 0 for item in drains)
    over4 = max(item["drain_over4"] or 0 for item in drains)
    decode = max(item["decode_ms_max"] or 0.0 for item in drains)
    passed = worst <= P6_DRAIN_MAX_MS and over1 <= P6_OVER_1MS_HANDFUL
    return [("P6", "HARD FAIL" if worst > P6_DRAIN_HARD_MS else verdict(passed),
             f"drain max {worst:.2f} ms, over 1 ms {over1} (at most {P6_OVER_1MS_HANDFUL}), over 4 ms {over4}"),
            ("P7", verdict(decode < P7_DECODE_MS), f"decode max {decode:.2f} ms")]


def memory_lines(groups):
    """P9 and the peak comparisons. Peaks are only set against each other from the same source."""
    first, second = arm_peak(run["gen"]["peak"] for run in groups["gen1"] if run["gen"]), \
        arm_peak(run["gen"]["peak"] for run in groups["gen2"] if run["gen"])
    lines = [("P9", "INFO" if first else "N/A",
              f"child peak {peak_text(first)}, hard faults are not in the report so check them by hand")]
    lines.append(("MEM", "INFO", "child peak first run vs second boot " + compare_peaks(first, second)))
    emulator_gen = arm_peak(run["proc_peak"] for run in groups["gen1"])
    emulator_gpl = arm_peak(run["proc_peak"] for run in groups["gpl1"])
    lines.append(("MEM", "INFO", "emulator peak gen vs gpl " + compare_peaks(emulator_gen, emulator_gpl)))
    return lines


def gate(records, common, gpl_wanted):
    """Prints the pass criteria for the gen arm against the named gpl arm. Returns 1 when an arm
    can't be found."""
    names = {arm_name(report, common) for report, _ in records}
    gpl_label = resolve_gpl_arm(names, gpl_wanted)
    if gpl_label is None:
        print(f"gate: no single gpl arm matches '{gpl_wanted}'", file=sys.stderr)
        return 1
    groups = gate_runs(records, common, gpl_label)
    if not groups["gen1"] or not groups["gpl1"]:
        print("gate: needs a gen first run and a gpl first run", file=sys.stderr)
        return 1
    gen, gpl = groups["gen1"], groups["gpl1"]
    gen_gfx, gpl_gfx = med(gen, "gfx_s"), med(gpl, "gfx_s")
    audio = audio_checks(gen, gpl)
    lines = audio[:2] + window_checks(gen, gpl) + audio[2:] + p5_lines(gen, gpl)
    lines += drain_checks(gen + groups["gen2"]) + second_boot_checks(groups["gen2"], groups["gpl2"], groups["gpl2_warm"])
    lines.append(("GFX", verdict(gen_gfx <= gpl_gfx * (1 + GFX_FIRST_RUN_ALLOWANCE)),
                  f"graphics blocked s, gen first run {gen_gfx:.2f} vs gpl first run {gpl_gfx:.2f} + 15%"))
    lines += memory_lines(groups)
    print()
    print(f"Gate, gen first run against gpl arm: {gpl_label}")
    for code, result, detail in lines:
        print(f"{code:<5} {result:<9} {detail}")
    return 0


def main(patterns, gate_arm=None):
    arms = {}
    refusals = {}
    gpu_translate = {}
    reports = [report for report in (load(path) for path in expand(patterns)) if report is not None]
    common = common_flags(reports)
    records = [(report, run_metrics(report)) for report in reports]
    new_arms = {}
    for report, metrics in records:
        new_arms.setdefault(arm_name(report, common), []).append(metrics)
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
    generation_table(new_arms)
    audio_table(new_arms)
    contention_table(new_arms)
    return gate(records, common, gate_arm) if gate_arm else 0


def parse_arguments(arguments):
    """Splits out --gate and its value. Returns the patterns and the gate arm, or None when the
    value is missing."""
    patterns = []
    gate_arm = None
    index = 0
    while index < len(arguments):
        if arguments[index] == "--gate":
            if index + 1 >= len(arguments):
                return None
            gate_arm = arguments[index + 1]
            index += 2
        else:
            patterns.append(arguments[index])
            index += 1
    return (patterns, gate_arm) if patterns else None


if __name__ == "__main__":
    parsed = parse_arguments(sys.argv[1:])
    if parsed is None:
        print(__doc__.strip())
        sys.exit(2)
    sys.exit(main(parsed[0], parsed[1]))
