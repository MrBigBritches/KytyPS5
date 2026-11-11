#!/usr/bin/env python3
"""Compares first-run reports written by `--first-run-report`.

Usage: first_run_compare.py [--gate <gpl arm>] [--watch <id,...>] <report.json or glob>...

With --watch, a Watched modules section follows the tier table. It takes comma-separated 16-hex module
ids, kept as text, and lists for each run and id the generated row (tier, wide_tail, posted, ready,
first use, lead), the warm take, or else the late_modules entry or the inline compile, with the median
lead per arm. It is for information and has no gate. Reports with no generation block print nothing.

After the main table come the Generation, Audio and Contention tables, each only for an arm whose
reports carry the data. A set of older reports prints the main table alone, as before.

Three more sections are for information only and have no gate. Contention by lane splits the graphics
compiles of the compile workers by warm_front and warm_back, for reports that carry them. The Front
drain cut splits those compiles at the latest ready_ms of a generated module that is not wide_tail.
Underruns by warm lane lists each underrun with its lane counts and groups them into events, for
reports whose audio_underruns carries the lane arrays. The lane and underrun sections need the lane
fields (warm_front and warm_back on worker compile rows, and the three arrays on audio_underruns).
The drain cut needs generated rows with wide_tail and ready_ms, so it also prints for older reports
that have those and no lane fields. A report with neither prints none of the three.

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


# Marks an arm name gets beyond its flags, for a run without --cold-pipelines and for a GPL run on a
# device that could not link libraries.
WARM_MARK = " (warm)"
GPL_INACTIVE_MARK = " (gpl inactive)"


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
        label += GPL_INACTIVE_MARK
    return label if report.get("cold") else label + WARM_MARK


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
# Front builds running at which a compile counts as taken while the Front lane was busy.
FRONT_BUSY_MIN = 3

# Limits of the pass criteria. P1 to P4, P6, P7 and P9 are from the audio review of step 6c.
P1_MEAN_HELD_S = 15.0
P1_RUN_HELD_S = 20.0
P1_HARD_FRACTION = 0.5
P4_EXTRA_SLOW_TAKES = 2
P3B_P95_ALLOWANCE = 1.25
P4_STOPPED_RATIO = 1.25
# The longest stop that always passes, so a gpl arm with tiny stops does not make the limit tiny.
P4_STOPPED_FLOOR_MS = 500.0
# The M1 pre-check. Go needs this share of used modules at tier 0, and more than this share of the
# unused build time at tier 1+.
M1_TIER0_USED_PERCENT = 90.0
M1_TIER1_UNUSED_BUILD_PERCENT = 50.0
P5_MIN_EXTRA_UNDERRUNS = 2
P5_EXTRA_FRACTION = 0.25
# Underrun submits no more than this far after the previous one belong to the same event. A gap of
# exactly this size is inclusive, so it joins the event, and anything longer starts a new one.
P5_EVENT_GAP_MS = 50.0
P6_DRAIN_MAX_MS = 2.0
P6_DRAIN_HARD_MS = 8.0
P6_OVER_1MS_HANDFUL = 5
P7_DECODE_MS = 50.0
P8_GFX_ALLOWANCE = 0.10
# Graphics blocked on a gen first run is information up to this much over gpl and a failure past
# it. The extra time comes from warm builds sharing the CPU with graphics compiles, which buys
# about 70 s less compute blocked, so it is a documented trade with a regression guard and not a
# gate, until the warm worker count is measured.
GFX_FIRST_RUN_HARD_LIMIT = 0.30
# A module id as the reports write it, 16 hex digits.
MODULE_ID_PATTERN = re.compile(r"[0-9a-fA-F]{16}")


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
            "windows": held,
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
    slow_generated_takes = [(hash_text(take.get("id")), take.get("stopped_ms", 0.0)) for take in slow
                            if hash_text(take.get("id")) in generated_ids]
    return {"slow_gen": slow_generated, "slow_player": len(slow) - slow_generated,
            "slow_gen_takes": slow_generated_takes,
            "slow_waited": waited, "slow_inline": len(slow) - waited,
            "max_stopped": max((take.get("stopped_ms", 0.0) for take in takes), default=0.0)}


def worker_graphics_rows(report):
    """The graphics compile rows of the compile workers that carry a start time. A warm compute
    worker's compile row sits inside its own warm build, so it counts itself in the warm counts and
    is never used. Rows from before the context fields existed have no at_ms and are left out."""
    return [row for row in report.get("compiles", []) if row.get("thread") == "worker"
            and row.get("kind") == "graphics" and "at_ms" in row]


def contention_figures(report):
    """Sums the graphics compiles of the compile workers. Only graphics rows count. A warm compute
    worker's compile row sits inside its own warm build, so it counts itself in warm_running and
    would put the warm-up's own work in the contention figure. Rows from before the context fields
    existed have no at_ms and are left out."""
    rows = worker_graphics_rows(report)
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


def lane_rows(report):
    """The graphics worker rows that carry both lane counts, as (create_ms, at_ms, front, back). The
    split uses the lane fields only, never warm_running minus a lane, because the three counts are
    separate relaxed loads and can disagree. None when no row has the lane fields."""
    rows = [(row["create_ms"], row["at_ms"], row["warm_front"], row["warm_back"])
            for row in worker_graphics_rows(report) if "warm_front" in row and "warm_back" in row]
    return rows or None


def drain_time_ms(report):
    """When the Front drain finished, as the largest ready_ms among the generated modules that are
    not wide_tail. The wide_tail modules are built later and must not move it. None when no such row
    has a ready_ms, which is the case for a report without generated rows. Caveats. Front work for the
    main store is not in generated_modules, so it can't move this time. Front work can also arrive
    after Back has started, such as a C5 head drop or a late adoption, so the drain is not a clean
    line between the lanes. The wide_tail_ready_ms below is printed beside it to show any overlap."""
    block = generation_block(report)
    if not block:
        return None
    ready = [row["ready_ms"] for row in block.get("generated_modules", [])
             if row.get("wide_tail") is False and row.get("ready_ms") is not None]
    return max(ready, default=None)


def wide_tail_ready_ms(report):
    """The earliest ready_ms of a wide_tail generated module, or None when there is none. A value
    before the drain time means Back work finished before the Front drain did."""
    block = generation_block(report)
    if not block:
        return None
    ready = [row["ready_ms"] for row in block.get("generated_modules", [])
             if row.get("wide_tail") is True and row.get("ready_ms") is not None]
    return min(ready, default=None)


def lane_underruns(report):
    """The underruns as (at_ms, front, back, child_alive) in report order, or None when the report
    lacks the lane arrays. The arrays are parallel to at_ms."""
    underruns = report.get("audio_underruns")
    if not isinstance(underruns, dict):
        return None
    keys = ("at_ms", "warm_front", "warm_back", "child_alive")
    if not all(key in underruns for key in keys):
        return None
    return list(zip(*(underruns[key] for key in keys)))


def tier_order_key(module):
    """The order variants of one program are ranked in. Posted time comes first, then adopted time
    for a tie, then the id as text for a tie in both. A missing time counts as 0."""
    return (module.get("posted_ms", 0.0), module.get("adopted_ms", 0.0), hash_text(module.get("id")))


def variant_tiers(modules):
    """The tier of each module, in the order of the list. The tier is the module's rank among the
    modules with the same code_hash in this one run, in posted order, so the first posted variant of
    each program is tier 0. A program with a single variant is tier 0. Ranks never carry over from
    another run, so pass the modules of one run only."""
    by_hash = {}
    for index, module in enumerate(modules):
        by_hash.setdefault(hash_text(module.get("code_hash")), []).append(index)
    tiers = [0] * len(modules)
    for indexes in by_hash.values():
        for rank, index in enumerate(sorted(indexes, key=lambda i: tier_order_key(modules[i]))):
            tiers[index] = rank
    return tiers


def tier_rows(block):
    """One row per generated module with its tier. Start delay is ready_ms - build_ms - adopted_ms,
    which is how long the module waited in the warm queue before a worker began to build it. A row
    without ready_ms, build_ms or adopted_ms was never built, has no delay and counts as not built."""
    modules = block.get("generated_modules", [])
    rows = []
    for module, tier in zip(modules, variant_tiers(modules)):
        ready, build, adopted = module.get("ready_ms"), module.get("build_ms"), module.get("adopted_ms")
        built = ready is not None and build is not None and adopted is not None
        rows.append({"tier": tier, "hash": hash_text(module.get("code_hash")), "used": module.get("how") != "unused",
                     "built": built, "build_ms": build or 0.0,
                     "delay_ms": ready - build - adopted if built else None})
    return rows


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
            "order": block.get("worker", {}).get("order"), "tier_rows": tier_rows(block),
            "classes": {name: (row.get("count", 0), row.get("create_ms", 0.0))
                        for name, row in classes.items()}}


def create_by_id(report):
    """The longest pipeline create time of each compile row that carries an id, keyed by the id as
    text. Rows without an id (graphics rows are named by their vs and ps) are left out."""
    longest = {}
    for row in report.get("compiles", []):
        if "id" in row:
            key = hash_text(row["id"])
            longest[key] = max(longest.get(key, 0.0), row.get("create_ms", 0.0))
    return longest


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
               "create_by_id": create_by_id(report),
               "machine": report.get("machine"),
               "lane_rows": lane_rows(report), "drain_ms": drain_time_ms(report),
               "wide_ready_ms": wide_tail_ready_ms(report),
               "drain_rows": [(row["create_ms"], row["at_ms"]) for row in worker_graphics_rows(report)],
               "lane_underruns": lane_underruns(report)}
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


def event_count(times):
    """Underrun events in a list of submit times. A submit no more than P5_EVENT_GAP_MS after the
    previous submit joins that submit's event, so a burst is one event however long it chains."""
    events = 0
    previous = None
    for at_ms in sorted(times):
        if previous is None or at_ms - previous > P5_EVENT_GAP_MS:
            events += 1
        previous = at_ms
    return events


def events_around(times, cut_ms):
    """Underrun events before the cut and from the cut on. The submits are split at the cut first,
    so a burst that straddles the cut counts once on each side."""
    return (event_count([at_ms for at_ms in times if at_ms < cut_ms]),
            event_count([at_ms for at_ms in times if at_ms >= cut_ms]))


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


def group_figures(durations):
    """Count, sum in ms and mean in ms of some compile durations. The mean is None for none."""
    return (len(durations), sum(durations), sum(durations) / len(durations) if durations else None)


def figure_text(figures):
    """Count, sum in seconds and mean in ms as one cell."""
    count, total, mean = figures
    return f"{count} / {total / MS_PER_SECOND:.2f} / {fmt_num(mean, 1)}"


def median_figures(figure_list):
    """The median over runs of each of count, sum and mean, skipping the runs with no mean."""
    return (median_of(item[0] for item in figure_list), median_of(item[1] for item in figure_list),
            median_of(item[2] for item in figure_list))


def median_figure_text(figure_list):
    """The median over runs of group figures as one cell."""
    count, total, mean = median_figures(figure_list)
    return f"{fmt_count(count)} / {fmt_num(None if total is None else total / MS_PER_SECOND, 2)} / {fmt_num(mean, 1)}"


def lane_table(arms):
    """Contention split by lane, for arms whose graphics worker rows carry warm_front and warm_back.
    For information only. The split reads the lane fields and never warm_running."""
    groups = (("back > 0", lambda front, back: back > 0), ("back == 0", lambda front, back: back == 0),
              (f"front >= {FRONT_BUSY_MIN}", lambda front, back: front >= FRONT_BUSY_MIN),
              (f"front <= {FRONT_BUSY_MIN - 1}", lambda front, back: front < FRONT_BUSY_MIN))
    columns = [(f"{label} n / s / mean ms", 24) for label, _ in groups]
    rows = []
    for name in sorted(arms):
        runs = [run for run in arms[name] if run["lane_rows"]]
        if not runs:
            continue
        cells = []
        for _, member in groups:
            cells.append(median_figure_text([
                group_figures([create for create, _, front, back in run["lane_rows"] if member(front, back)])
                for run in runs]))
        rows.append((name, cells))
    if rows:
        print_table("Contention by lane (information only, graphics compiles of the compile workers,"
                    " lanes from warm_front and warm_back)", columns, rows)


def drain_cut_lines(name, runs):
    """One line per run of an arm that has a Front drain time, with the graphics compiles of the
    compile workers split at it by their start time. A compile that starts at the drain counts after."""
    lines = []
    for number, run in enumerate(runs, 1):
        drain = run["drain_ms"]
        if drain is None or not run["drain_rows"]:
            continue
        before = group_figures([create for create, at_ms in run["drain_rows"] if at_ms < drain])
        after = group_figures([create for create, at_ms in run["drain_rows"] if at_ms >= drain])
        wide = run["wide_ready_ms"]
        lines.append(f"  {name} run {number}: drain {drain:.1f} ms, before n / s / mean ms"
                     f" {figure_text(before)}, after {figure_text(after)},"
                     f" earliest wide_tail ready {'none' if wide is None else f'{wide:.1f} ms'}")
    return lines


def drain_cut_section(arms):
    """The Front drain cut. Nothing prints for reports with no drain time."""
    lines = [line for name in sorted(arms) for line in drain_cut_lines(name, arms[name])]
    if lines:
        print()
        print("Front drain cut (information only, drain = last Front build done, the latest ready_ms"
              " of a generated module that is not wide_tail. The Front queue empties earlier, and"
              " Back builds can start while the last Front ones finish)")
        for line in lines:
            print(line)


def lane_events(entries):
    """Groups underruns into events with the P5 gap rule. A submit no more than P5_EVENT_GAP_MS after
    the previous one joins its event. Entries are (at_ms, front, back, child_alive)."""
    events = []
    for entry in sorted(entries, key=lambda item: item[0]):
        if events and entry[0] - events[-1][-1][0] <= P5_EVENT_GAP_MS:
            events[-1].append(entry)
        else:
            events.append([entry])
    return events


def underrun_side(at_ms, drain):
    """Whether an underrun falls before or after the run's Front drain."""
    if drain is None:
        return "drain unknown"
    return "before drain" if at_ms < drain else "after drain"


def underrun_lines(name, runs):
    """The underrun list of one arm, then its event counts. An event has back > 0 when any of its
    submits had a Back build running."""
    lines = []
    with_back = without_back = onset_total = 0
    for number, run in enumerate(runs, 1):
        entries = run["lane_underruns"]
        if entries is None:
            continue
        events = lane_events(entries)
        for index, event in enumerate(events, 1):
            for at_ms, front, back, alive in event:
                lines.append(f"  {name} run {number} event {index}: at {at_ms:.1f} ms front {front} back {back}"
                             f" child_alive {'yes' if alive else 'no'} {underrun_side(at_ms, run['drain_ms'])}")
        busy = sum(1 for event in events if any(entry[2] > 0 for entry in event))
        # The onset is the first submit of the event, which is when the burst began.
        onset = sum(1 for event in events if event[0][2] > 0)
        with_back += busy
        without_back += len(events) - busy
        onset_total += onset
        lines.append(f"  {name} run {number}: {len(events)} events, {busy} with back > 0 (any submit),"
                     f" {onset} with back > 0 (first submit), {len(events) - busy} with back == 0 (any submit)")
    if lines:
        lines.append(f"  {name}: total over runs, events with back > 0 (any submit) {with_back}"
                     f" against back == 0 {without_back}, with back > 0 (first submit) {onset_total}")
    return lines


def underrun_section(arms):
    """The underrun list, for runs whose audio_underruns carries the lane arrays."""
    lines = [line for name in sorted(arms) for line in underrun_lines(name, arms[name])]
    if lines:
        print()
        print(f"Underruns by warm lane (information only, events join submits within {P5_EVENT_GAP_MS:g} ms)")
        for line in lines:
            print(line)


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


def tier_group_stats(rows):
    """The figures of one tier group of one run."""
    delays = [row["delay_ms"] for row in rows if row["delay_ms"] is not None]
    return {"modules": len(rows), "used": sum(1 for row in rows if row["used"]),
            "unused": sum(1 for row in rows if not row["used"]),
            "not_built": sum(1 for row in rows if not row["built"]),
            "unused_build_ms": sum(row["build_ms"] for row in rows if not row["used"]),
            "delay_ms": statistics.median(delays) if delays else None}


def share_percent(part, whole):
    """Part as a percentage of the whole, or None when the whole is zero."""
    return None if not whole else 100.0 * part / whole


def tier_run_figures(rows):
    """Tier 0 and tier 1+ figures of one run, with the two shares the M1 pre-check reads."""
    tier0 = tier_group_stats([row for row in rows if row["tier"] == 0])
    later = tier_group_stats([row for row in rows if row["tier"] > 0])
    return {"t0": tier0, "t1": later,
            "used_at_0": share_percent(tier0["used"], tier0["used"] + later["used"]),
            "unused_build_at_1": share_percent(later["unused_build_ms"],
                                               tier0["unused_build_ms"] + later["unused_build_ms"])}


def tier_cells(figures_list, key):
    """The cells of one tier row, as the median over the runs of an arm."""
    stats = [figures[key] for figures in figures_list]
    unused_build = median_of(item["unused_build_ms"] for item in stats)
    return [fmt_count(median_of(item[name] for item in stats))
            for name in ("modules", "used", "unused", "not_built")] + \
           [fmt_num(unused_build / MS_PER_SECOND), fmt_num(median_of(item["delay_ms"] for item in stats), 1)]


def program_lines(name, runs):
    """For each program with more than one variant, the tiers that were used. Over several runs the
    used tiers are joined and the variant count is the largest seen. Hashes are text."""
    programs = {}
    for run in runs:
        counts = {}
        for row in run["gen"]["tier_rows"]:
            counts[row["hash"]] = counts.get(row["hash"], 0) + 1
            entry = programs.setdefault(row["hash"], [set(), 0])
            if row["used"]:
                entry[0].add(row["tier"])
        for code_hash, count in counts.items():
            programs[code_hash][1] = max(programs[code_hash][1], count)
    return [f"  {name}: {code_hash}: used tiers {','.join(str(tier) for tier in sorted(tiers)) or 'none'} of {count}"
            for code_hash, (tiers, count) in sorted(programs.items()) if count > 1]


def m1_summary(name, runs):
    """The two shares and the M1 go or no-go. It is information and not a gate. Go needs at least
    90 percent of used modules at tier 0 and more than half of the unused build time at tier 1+.
    The shares are taken over the modules of all runs of the arm pooled together, so each module
    and each second of build counts once. Tiers are still ranked within each run."""
    pooled = tier_run_figures([row for run in runs for row in run["gen"]["tier_rows"]])
    used_at_0, unused_at_1 = pooled["used_at_0"], pooled["unused_build_at_1"]
    lines = [f"  {name}: used modules at tier 0 (runs pooled): {fmt_num(used_at_0, 1)}%",
             f"  {name}: unused build time at tier 1+ (runs pooled): {fmt_num(unused_at_1, 1)}%"]
    if used_at_0 is None or unused_at_1 is None:
        verdict_text = "n/a (no used modules or no unused build time)"
    else:
        go = used_at_0 >= M1_TIER0_USED_PERCENT and unused_at_1 > M1_TIER1_UNUSED_BUILD_PERCENT
        verdict_text = "go" if go else "no-go"
    lines.append(f"  {name}: M1 pre-check: {verdict_text} (INFO) with used at tier 0 {fmt_num(used_at_0, 1)}%"
                 f" (needs {M1_TIER0_USED_PERCENT:g} or more) and unused build at tier 1+ {fmt_num(unused_at_1, 1)}%"
                 f" (needs over {M1_TIER1_UNUSED_BUILD_PERCENT:g})")
    return lines


def tier_table(arms):
    """The Generated by tier table, for arms whose reports carry the generation block with modules."""
    columns = [("modules", 7), ("used", 4), ("unused", 6), ("not built", 9), ("unused build s", 14),
               ("start delay ms", 14)]
    rows, notes = [], []
    for name in sorted(arms):
        runs = [run for run in arms[name] if run["gen"] is not None and run["gen"]["tier_rows"]]
        if not runs:
            continue
        figures_list = [tier_run_figures(run["gen"]["tier_rows"]) for run in runs]
        rows += [(f"{name} tier 0", tier_cells(figures_list, "t0")),
                 (f"{name} tier 1+", tier_cells(figures_list, "t1"))]
        notes += m1_summary(name, runs) + program_lines(name, runs)
    if not rows:
        return
    print_table("Generated by tier (tier = variant rank within code hash by posted time, "
                "start delay = ready - build - adopted)", columns, rows)
    for note in notes:
        print(note)


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
        if name == gpl_label + WARM_MARK and metrics["second_boot"]:
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


def pooled_p95(runs):
    """The 95th percentile of the resolve windows of all runs of an arm taken together, with the
    number of windows. Pooling weighs each window once, where a median of per-run p95s would let a
    short run count as much as a long one."""
    pooled = [held for run in runs for held in run["windows"]]
    return percentile(pooled, P95_FRACTION), len(pooled)


def window_checks(gen, gpl):
    """P3. P3a and P3c need gen no higher than gpl. P3b compares the pooled p95 with a 25 percent
    allowance. P3d is information only, since P2 already covers the hold counts."""
    lines = []
    for code, key, label in (("P3a", "win_max", "window_held max ms"), ("P3c", "hold_max", "holds per window max")):
        gen_value, gpl_value = med(gen, key), med(gpl, key)
        lines.append((code, verdict(at_most(gen_value, gpl_value)),
                      f"{label}, gen {fmt_num(gen_value)} vs gpl {fmt_num(gpl_value)}"))
    gen_p95, gen_count = pooled_p95(gen)
    gpl_p95, gpl_count = pooled_p95(gpl)
    limit = None if gpl_p95 is None else gpl_p95 * P3B_P95_ALLOWANCE
    lines.insert(1, ("P3b", verdict(at_most(gen_p95, limit)),
                     f"window_held p95 ms pooled over all runs, gen {fmt_num(gen_p95)} ({gen_count} windows)"
                     f" vs gpl {fmt_num(gpl_p95)} ({gpl_count} windows), limit {fmt_num(limit)}"
                     f" (gpl x {P3B_P95_ALLOWANCE})"))
    lines.append(("P3d", "INFO", f"holds per window mean, gen {fmt_num(med(gen, 'hold_mean'))}"
                                 f" vs gpl {fmt_num(med(gpl, 'hold_mean'))} (information only, P2 covers hold counts)"))
    return lines


def like_for_like_lines(gen, gpl):
    """P4e, information only. Each slow generated take of the gen runs is shown beside the create
    time of the same pipeline id in the gpl runs' compile rows, as the median over the gpl runs that
    have a row for it. Ids are text and are matched as text."""
    takes = [take for run in gen for take in run["slow_gen_takes"]]
    if not takes:
        return [("P4e", "INFO", "no slow generated takes to compare with gpl create time")]
    lines = []
    for take_id, stopped_ms in takes:
        creates = [run["create_by_id"][take_id] for run in gpl if take_id in run["create_by_id"]]
        gpl_text = "n/a" if not creates else f"{statistics.median(creates):.1f} ms"
        lines.append(("P4e", "INFO", f"like for like, gen take {take_id} stopped {stopped_ms:.1f} ms,"
                                     f" gpl create_ms for the same id {gpl_text}"))
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
    limit = None if gpl_stop is None else max(P4_STOPPED_RATIO * gpl_stop, P4_STOPPED_FLOOR_MS)
    lines.append(("P4b", verdict(at_most(gen_stop, limit)),
                  f"max stopped ms, gen {fmt_num(gen_stop, 1)} vs gpl {fmt_num(gpl_stop, 1)}, limit"
                  f" {fmt_num(limit, 1)} (the larger of gpl x {P4_STOPPED_RATIO} and {P4_STOPPED_FLOOR_MS:g})"))
    lines += like_for_like_lines(gen, gpl)
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
        gen_count = median_of(events_around(run["underruns"], cut)[index] for run in gen_runs)
        gpl_count = median_of(events_around(run["underruns"], cut)[index] for run in gpl_runs)
        gen_submits = median_of(count_around(run["underruns"], cut)[index] for run in gen_runs)
        gpl_submits = median_of(count_around(run["underruns"], cut)[index] for run in gpl_runs)
        limit = gpl_count + max(P5_MIN_EXTRA_UNDERRUNS, P5_EXTRA_FRACTION * gpl_count)
        lines.append((f"P5{'ab'[index]}", verdict(gen_count <= limit),
                      f"underrun events {window} cut {cut:.0f} ms, gen {gen_count:g} vs gpl {gpl_count:g},"
                      f" limit {limit:g} (submits, gen {gen_submits:g} vs gpl {gpl_submits:g})"))
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


def gen_arm_labels(records, common, gpl_label):
    """The names of the arms among the generation runs, the first runs and second boots of one arm
    sharing a name. More than one means the reports mix gen arms, such as the gen flags with and
    without another strategy flag. The marks for a warm run or an inactive library are left off,
    since they do not make a run another arm, so only a difference in flags counts."""
    labels = set()
    for report, _ in records:
        name = arm_name(report, common, with_store=False)
        if report.get("flags", {}).get(GENERATE_FLAG) and name != gpl_label:
            labels.add(name.removesuffix(WARM_MARK).removesuffix(GPL_INACTIVE_MARK))
    return sorted(labels)


def gate(records, common, gpl_wanted):
    """Prints the pass criteria for the gen arm against the named gpl arm. Returns 1 when an arm
    can't be found."""
    names = {arm_name(report, common) for report, _ in records}
    gpl_label = resolve_gpl_arm(names, gpl_wanted)
    if gpl_label is None:
        print(f"gate: no single gpl arm matches '{gpl_wanted}'", file=sys.stderr)
        return 1
    gen_labels = gen_arm_labels(records, common, gpl_label)
    if len(gen_labels) > 1:
        # gate_runs puts every generation run in one group, so two gen arms given together would
        # be judged as one blend of both. Refuse, so each arm is gated on its own.
        print("gate: the reports hold more than one gen arm, so pass one at a time:", file=sys.stderr)
        for label in gen_labels:
            print(f"  {label}", file=sys.stderr)
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
    within = gen_gfx <= gpl_gfx * (1 + GFX_FIRST_RUN_HARD_LIMIT)
    lines.append(("GFX", "INFO" if within else "FAIL",
                  f"graphics blocked s, gen first run {gen_gfx:.2f} vs gpl first run {gpl_gfx:.2f},"
                  f" fails past + {GFX_FIRST_RUN_HARD_LIMIT:.0%} (a trade for less compute blocked)"))
    lines += memory_lines(groups)
    print()
    print(f"Gate, gen first run against gpl arm: {gpl_label}")
    for code, result, detail in lines:
        print(f"{code:<5} {result:<9} {detail}")
    return 0


def watch_row(report, module_id):
    """What one run says about one module id, as the text after the id. The generated row comes
    first, then the late_modules entry, then an inline compile, and a module in none of them is not
    seen. A warm take is added to any of them. Ids are matched as lowercase text."""
    block = generation_block(report)
    generated = next((row for row in block.get("generated_modules", [])
                      if hash_text(row.get("id")).lower() == module_id), None)
    late = next((row for row in block.get("runtime_misses", {}).get("late_modules", [])
                 if hash_text(row.get("id")).lower() == module_id), None)
    creates = [row.get("create_ms", 0.0) for row in report.get("compiles", [])
               if "id" in row and hash_text(row["id"]).lower() == module_id]
    take = next((row for row in report.get("warm_takes", [])
                 if hash_text(row.get("id")).lower() == module_id), None)
    if generated is not None:
        wide = {True: "yes", False: "no"}.get(generated.get("wide_tail"), "-")
        text = (f"tier {fmt_count(generated.get('tier'))} wide_tail {wide}"
                f" posted {fmt_num(generated.get('posted_ms'), 1)} ready {fmt_num(generated.get('ready_ms'), 1)}"
                f" first_use {fmt_num(generated.get('first_use_ms'), 1)}"
                f" lead {fmt_num(generated.get('lead_ms'), 1)} how {generated.get('how', '-')}")
    elif late is not None:
        # A refused row's delivered_ms is when the drain refused it, and the worker's post time was
        # not recorded, so it must not read as a post time.
        if late.get("via") == "refused":
            delivery = f"refused {fmt_num(late.get('delivered_ms'), 1)} (drain refusal, post not recorded)"
        else:
            delivery = f"posted {fmt_num(late.get('delivered_ms'), 1)}"
        text = (f"late miss {fmt_num(late.get('miss_ms'), 1)} {delivery}"
                f" create {fmt_num(late.get('create_ms'), 1)} misses {fmt_count(late.get('misses'))}")
    elif creates:
        text = f"inline compile create {fmt_num(max(creates), 1)}"
    else:
        text = "not seen"
    if take is not None:
        text += f"; take {take.get('how', '-')} stopped {fmt_num(take.get('stopped_ms'), 1)}"
    return text


def lead_of(report, module_id):
    """The lead_ms of a module's generated row in one run, or None when it has none."""
    block = generation_block(report)
    row = next((row for row in block.get("generated_modules", [])
                if hash_text(row.get("id")).lower() == module_id), None)
    return None if row is None else row.get("lead_ms")


def print_watch(watch_arms, watch_ids):
    """The Watched modules section, for information only and with no gate. It lists one row per run
    and id for each arm, then the median lead of each id across the arm's runs. A run without a
    generation block adds nothing, so a set of old reports prints no section at all."""
    if not any(watch_arms.values()):
        return
    print()
    print("Watched modules (information only, times in ms from the run start, lead = first use - ready)")
    for name in sorted(watch_arms):
        runs = watch_arms[name]
        for number, report in enumerate(runs, 1):
            for module_id in watch_ids:
                print(f"  {name} run {number} {module_id}: {watch_row(report, module_id.lower())}")
        for module_id in watch_ids:
            leads = [lead_of(report, module_id.lower()) for report in runs]
            median_lead = median_of(leads)
            present = sum(1 for lead in leads if lead is not None)
            print(f"  {name} {module_id}: median lead {fmt_num(median_lead, 1)}"
                  f" over {present} of {len(runs)} runs")


def main(patterns, gate_arm=None, watch=()):
    """Prints the tables for the reports. Watch is a list of module ids to list under Watched modules,
    and the gate arm names the baseline for the pass criteria."""
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
    tier_table(new_arms)
    if watch:
        watch_arms = {}
        for report, _ in records:
            if generation_block(report):
                watch_arms.setdefault(arm_name(report, common), []).append(report)
        print_watch(watch_arms, list(watch))
    audio_table(new_arms)
    contention_table(new_arms)
    lane_table(new_arms)
    drain_cut_section(new_arms)
    underrun_section(new_arms)
    return gate(records, common, gate_arm) if gate_arm else 0


def parse_watch_ids(text):
    """Splits a comma-separated list of module ids, or returns None when any id is not 16 hex digits.
    The ids stay text as given, so leading zeros are kept."""
    ids = text.split(",")
    if not all(MODULE_ID_PATTERN.fullmatch(module_id) for module_id in ids):
        return None
    # A repeated id would print its rows twice, so each is kept once in its first position.
    return list(dict.fromkeys(ids))


def parse_arguments(arguments):
    """Splits out --gate and --watch with their values. Returns the patterns, the gate arm and the
    watched ids (empty with no --watch), or None when a value is missing or is not valid, or there
    are no patterns."""
    patterns = []
    gate_arm = None
    watch = []
    index = 0
    while index < len(arguments):
        if arguments[index] in ("--gate", "--watch"):
            if index + 1 >= len(arguments):
                return None
            if arguments[index] == "--gate":
                gate_arm = arguments[index + 1]
            else:
                ids = parse_watch_ids(arguments[index + 1])
                if ids is None:
                    return None
                watch += [module_id for module_id in ids if module_id not in watch]
            index += 2
        else:
            patterns.append(arguments[index])
            index += 1
    return (patterns, gate_arm, watch) if patterns else None


if __name__ == "__main__":
    parsed = parse_arguments(sys.argv[1:])
    if parsed is None:
        print(__doc__.strip())
        sys.exit(2)
    sys.exit(main(parsed[0], parsed[1], parsed[2]))
