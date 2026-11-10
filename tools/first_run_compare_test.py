#!/usr/bin/env python3
"""Tests for first_run_compare.py. Run as `python -I tools/first_run_compare_test.py`.

The fixtures in first_run_compare_fixtures are small hand-made reports. Where a test says a variant
differs from the baseline only in one behaviour, the variant is the fixture with that one edit made
in memory, so a changed line in the output can only come from the edit.

Fixture numbers worth knowing:
  The gen first run generated for 10000 ms (finished_ms), starting 700 ms after the run start, so it
  finished at 10700 ms on the run's clock (finished_at_ms). It has underruns at 500..900, 10300 and
  15000 ms. The gpl first run has underruns at 1000, 2000, 30000 and 40000 ms. Cut at 10700 ms that
  is gen 6 and 1 against gpl 2 and 2, so the before window fails and the after window passes. Cut at
  the duration of 10000 ms it would be gen 5 and 2, and uncut the totals (7 and 4) would give a
  different verdict again.
  The gpl second boot holds audio 5 s and the gen second boot 6 s. The gen first run holds 8 s, so a
  baseline taken from the gen first run would pass where the gpl second boot fails.
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest

# Importing the script by path would otherwise leave a __pycache__ folder in tools.
sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "first_run_compare_fixtures")
GPL_ARM = "--deferred-pipeline-draws --graphics-pipeline-library"
GEN_ARM = "--deferred-pipeline-draws --generate-shader-store --graphics-pipeline-library"
NEW_FILES = ["new_gpl_first.json", "new_gpl_second.json", "new_gen_first.json", "new_gen_second.json"]


def load_script():
    """Imports the script by path, since -I leaves its folder off sys.path."""
    spec = importlib.util.spec_from_file_location("first_run_compare", os.path.join(HERE, "first_run_compare.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = load_script()


def read_fixture(name):
    """Reads one fixture report as a dict."""
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as handle:
        return json.load(handle)


def run_script(paths, gate=None):
    """Runs main on the paths and returns the exit code, stdout and stderr."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = SCRIPT.main(paths, gate)
    return code, out.getvalue(), err.getvalue()


def run_new(edits=None, gate=GPL_ARM, extra=None):
    """Runs the four new fixtures, each optionally edited in memory by a function keyed on its file
    name. Extra is a list of (file name, report) added as further runs."""
    edits = edits or {}
    with tempfile.TemporaryDirectory() as folder:
        paths = []
        reports = [(name, read_fixture(name)) for name in NEW_FILES] + list(extra or [])
        for name, report in reports:
            if name in edits:
                edits[name](report)
            path = os.path.join(folder, name)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(report, handle)
            paths.append(path)
        code, out, err = run_script(paths, gate)
    assert code == 0, err
    return out


def gate_line(output, code):
    """The gate line that starts with the code, with its columns squeezed to single spaces."""
    for line in output.splitlines():
        if line.split(" ")[0] == code:
            return " ".join(line.split())
    raise AssertionError(f"no gate line {code} in\n{output}")


def gate_lines(output, code):
    """Every gate line that starts with the code."""
    return [" ".join(line.split()) for line in output.splitlines() if line.split(" ")[0] == code]


def table_rows(output, title):
    """The lines of the table whose title line starts with the title."""
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(title):
            rows = []
            for row in lines[index + 3:]:
                if not row.strip() or not row.startswith("--"):
                    break
                rows.append(row)
            return rows
    return []


def row_cells(output, title, arm):
    """The cells of an arm's row in a table, or None when the table has no row for the arm. The
    arm label is followed by two spaces, which a longer label of the same start is not."""
    for row in table_rows(output, title):
        if row.startswith(arm + "  "):
            return row[len(arm):].split()
    return None


class OldReports(unittest.TestCase):
    """Reports from before the new fields print exactly as the old script printed them."""

    def test_old_reports_print_byte_identical(self):
        paths = [os.path.join(FIXTURES, "old_baseline.json"), os.path.join(FIXTURES, "old_gpl.json")]
        with open(os.path.join(FIXTURES, "old_expected.txt"), encoding="utf-8") as handle:
            expected = handle.read()
        code, out, _ = run_script(paths)
        self.assertEqual(code, 0)
        self.assertEqual(out, expected)

    def test_old_reports_print_no_new_table(self):
        # The identical text above already rules a table out. This names the three titles so a
        # failure says which table leaked.
        paths = [os.path.join(FIXTURES, "old_baseline.json"), os.path.join(FIXTURES, "old_gpl.json")]
        _, out, _ = run_script(paths)
        for title in ("Generation", "Audio", "Contention"):
            self.assertEqual(table_rows(out, title), [], title)
            self.assertNotIn(title, out)

    def test_old_report_beside_new_ones_adds_no_row(self):
        # The old gpl report is its own arm and must not get a row in a table that its data can't fill.
        old = read_fixture("old_gpl.json")
        old["flags"]["--deferred-pipeline-draws"] = True
        out = run_new(extra=[("old_extra.json", old)], gate=None)
        for title in ("Generation", "Audio", "Contention"):
            for row in table_rows(out, title):
                self.assertNotIn("--background-warm-up", row)


class NewTables(unittest.TestCase):
    """A new report prints the three tables, and each only for an arm with the data."""

    def test_generation_row(self):
        cells = row_cells(run_new(gate=None), "Generation", GEN_ARM)
        self.assertEqual(cells, ["2", "1", "10.00", "200", "1.50", "1", "0", "2", "1", "1", "1", "2", "3.00",
                                 "100.0", "MB", "(working_set)"])

    def test_generation_duration_is_labelled_as_a_duration(self):
        # The figure is finished_ms, a duration from the start of generation, and not a time of the run.
        header = next(line for line in run_new(gate=None).splitlines() if "gen duration s" in line)
        self.assertIn("gen duration s", header)
        self.assertNotIn("finished s", header)

    def test_generation_not_printed_for_an_arm_without_the_block(self):
        out = run_new(gate=None)
        self.assertIsNone(row_cells(out, "Generation", GPL_ARM))
        without = run_new({"new_gen_first.json": lambda report: report.pop("shader_store_generation")},
                          gate=None)
        self.assertIsNone(row_cells(without, "Generation", GEN_ARM))

    def test_miss_classes_are_labelled_as_pipeline_create_time(self):
        out = run_new(gate=None)
        self.assertIn("median pipeline create ms, not translation time", out)
        self.assertIn("before_prediction             2      60.00", out)
        self.assertNotIn("translate ms", out.split("Generation")[1].split("Audio")[0])

    def test_audio_row(self):
        cells = row_cells(run_new(gate=None), "Audio", GEN_ARM)
        self.assertEqual(cells, ["8.00", "20", "7", "6", "1", "1", "1", "500.0", "600.0", "600.0", "2", "1.20"])

    def test_audio_split_follows_finished_at_ms(self):
        # Baseline above splits at 10700. Moving only finished_at_ms moves the split, and moving only
        # finished_ms leaves it where it was.
        def move_at(report):
            report["shader_store_generation"]["supervisor"]["finished_at_ms"] = 400.0
        def move_duration(report):
            report["shader_store_generation"]["supervisor"]["finished_ms"] = 400.0
        self.assertEqual(row_cells(run_new({"new_gen_first.json": move_at}, gate=None), "Audio", GEN_ARM)[2:5],
                         ["7", "0", "7"])
        self.assertEqual(row_cells(run_new({"new_gen_first.json": move_duration}, gate=None), "Audio", GEN_ARM)[2:5],
                         ["7", "6", "1"])

    def test_audio_split_is_blank_without_finished_at_ms(self):
        def drop(report):
            del report["shader_store_generation"]["supervisor"]["finished_at_ms"]
        cells = row_cells(run_new({"new_gen_first.json": drop}, gate=None), "Audio", GEN_ARM)
        self.assertEqual(cells[2:5], ["7", "-", "-"])

    def test_audio_row_needs_audio_underruns(self):
        without = run_new({"new_gen_first.json": lambda report: report.pop("audio_underruns")}, gate=None)
        self.assertIsNone(row_cells(without, "Audio", GEN_ARM))
        self.assertIsNotNone(row_cells(without, "Audio", GPL_ARM))

    def test_contention_row(self):
        cells = row_cells(run_new(gate=None), "Contention", GEN_ARM)
        self.assertEqual(cells, ["4", "1.20", "450.0", "3", "/", "0.85", "1", "/", "0.35", "1", "/", "0.35",
                                 "1", "/", "0.15", "2", "/", "0.70", "4096.0", "MB", "(peak_commit_bytes)"])

    def test_contention_needs_worker_rows_with_context(self):
        def strip_context(report):
            for row in report["compiles"]:
                row.pop("at_ms", None)
        without = run_new({"new_gen_first.json": strip_context}, gate=None)
        self.assertIsNone(row_cells(without, "Contention", GEN_ARM))
        self.assertIsNotNone(row_cells(without, "Contention", GPL_ARM))

    def test_machine_line(self):
        self.assertIn("Test CPU, 16 logical cores, 4 warm workers, 32.0 GB RAM", run_new(gate=None))


class Contention(unittest.TestCase):
    """Contention is attributed from graphics worker rows only."""

    def test_warm_compute_rows_do_not_change_the_figure(self):
        def drop_compute(report):
            report["compiles"] = [row for row in report["compiles"] if row["kind"] != "compute"]
        with_compute = row_cells(run_new(gate=None), "Contention", GEN_ARM)
        without = row_cells(run_new({"new_gen_first.json": drop_compute}, gate=None), "Contention", GEN_ARM)
        self.assertEqual(with_compute, without)
        # The compute row is 7 s, so any sum near 8 s means it was counted.
        self.assertEqual(with_compute[:2], ["4", "1.20"])

    def test_a_graphics_row_does_change_the_figure(self):
        # The baseline above would also hold if every row were ignored, so one graphics row is added.
        def add_graphics(report):
            report["compiles"].append({"kind": "graphics", "vs": "0000000000000009", "ps": "000000000000000a",
                                       "create_ms": 1000.0, "phase": "play", "thread": "worker", "flip": 9,
                                       "at_ms": 4000.0, "warm_running": 0, "child_alive": False})
        cells = row_cells(run_new({"new_gen_first.json": add_graphics}, gate=None), "Contention", GEN_ARM)
        self.assertEqual(cells[:2], ["5", "2.20"])


class GatePass5(unittest.TestCase):
    """P5 cuts both arms at the gen arm's median finished_ms."""

    def test_cut_at_gen_finished_at(self):
        out = run_new()
        self.assertEqual(gate_line(out, "P5a"),
                         "P5a FAIL underrun events before cut 10700 ms, gen 6 vs gpl 2, limit 4"
                         " (submits, gen 6 vs gpl 2)")
        self.assertEqual(gate_line(out, "P5b"),
                         "P5b PASS underrun events after cut 10700 ms, gen 1 vs gpl 2, limit 4"
                         " (submits, gen 1 vs gpl 2)")

    def test_moving_the_finish_moves_the_cut(self):
        # Baseline for the cut. Only finished_at_ms differs, and the window counts follow it.
        def finish_early(report):
            report["shader_store_generation"]["supervisor"]["finished_at_ms"] = 400.0
        out = run_new({"new_gen_first.json": finish_early})
        self.assertEqual(gate_line(out, "P5a"),
                         "P5a PASS underrun events before cut 400 ms, gen 0 vs gpl 0, limit 2"
                         " (submits, gen 0 vs gpl 0)")
        self.assertEqual(gate_line(out, "P5b"),
                         "P5b FAIL underrun events after cut 400 ms, gen 7 vs gpl 4, limit 6"
                         " (submits, gen 7 vs gpl 4)")

    def test_the_duration_is_not_the_cut(self):
        # finished_ms is 700 ms short of the run-clock time. Changing it alone changes nothing in P5,
        # and a cut taken from it would give 5 and 2 here.
        def other_duration(report):
            report["shader_store_generation"]["supervisor"]["finished_ms"] = 400.0
        base = run_new()
        out = run_new({"new_gen_first.json": other_duration})
        for code in ("P5a", "P5b"):
            self.assertEqual(gate_line(out, code), gate_line(base, code))

    def test_limit_is_the_larger_of_two_and_a_quarter(self):
        # With 12 gpl underruns before the cut a quarter is 3, which beats the flat 2.
        def many(report):
            # Spaced beyond the event gap, so each submit is its own event.
            report["audio_underruns"]["at_ms"] = [index * 100.0 for index in range(12)]
        out = run_new({"new_gpl_first.json": many})
        self.assertIn("gpl 12, limit 15", gate_line(out, "P5a"))

    def test_without_finished_at_ms_p5_is_not_judged(self):
        # An older report has finished_ms and no finished_at_ms. The duration must not stand in.
        def no_finish(report):
            del report["shader_store_generation"]["supervisor"]["finished_at_ms"]
        out = run_new({"new_gen_first.json": no_finish})
        line = gate_line(out, "P5")
        self.assertEqual(line.split(" ")[1], "N/A")
        self.assertIn("no supervisor.finished_at_ms", line)
        self.assertEqual(gate_lines(out, "P5a"), [])


class GatePass8(unittest.TestCase):
    """P8 measures the gen second boot against the gpl second boot."""

    def test_p8_fails_against_the_gpl_second_boot(self):
        out = run_new()
        self.assertEqual(gate_line(out, "P8a"),
                         "P8a FAIL audio_held s, gen second boot 6.00 vs gpl second boot 5.00")
        self.assertEqual(gate_line(out, "P8b"),
                         "P8b FAIL window_held max ms, gen second boot 200.00 vs gpl second boot 150.00")
        self.assertEqual(gate_line(out, "P8c").split(" ")[1], "PASS")

    def test_p8_passes_when_the_gen_second_boot_is_no_worse(self):
        def better(report):
            report["audio_held_ms"] = 4000.0
            report["deferred_draws"]["resolves"][1]["window_held_ms"] = 100.0
        out = run_new({"new_gen_second.json": better})
        self.assertEqual(gate_line(out, "P8a").split(" ")[1], "PASS")
        self.assertEqual(gate_line(out, "P8b").split(" ")[1], "PASS")

    def test_p8_ignores_the_gen_first_run(self):
        # A gen first run that is far better or far worse leaves P8 as it was.
        def worse_first(report):
            report["audio_held_ms"] = 90000.0
            report["deferred_draws"]["resolves"][4]["window_held_ms"] = 90000.0
            report["play"]["blocked_ms"]["graphics"] = 1.0
        def better_first(report):
            report["audio_held_ms"] = 1.0
            report["deferred_draws"]["resolves"][4]["window_held_ms"] = 1.0
        base = run_new()
        for edit in (worse_first, better_first):
            out = run_new({"new_gen_first.json": edit})
            for code in ("P8a", "P8b", "P8c"):
                self.assertEqual(gate_line(out, code), gate_line(base, code))

    def test_p8_needs_both_second_boots(self):
        out = run_new(extra=None, edits={"new_gpl_second.json": lambda report: report["boot"].update(modules=0)})
        self.assertEqual(gate_line(out, "P8").split(" ")[1], "N/A")


    def test_a_warm_gpl_second_boot_says_why_p8_is_not_judged(self):
        # A second boot without --cold-pipelines gets a (warm) label, so it is another arm. P8 is N/A
        # and says so. The baseline is the same report with the flag, which P8 judges as before.
        def warm(report):
            report["cold"] = False
        line = gate_line(run_new({"new_gpl_second.json": warm}), "P8")
        self.assertEqual(line.split(" ")[1], "N/A")
        self.assertIn("--cold-pipelines", line)
        self.assertIn("warm arm", line)
        self.assertEqual(gate_lines(run_new({"new_gpl_second.json": warm}), "P8a"), [])
        self.assertEqual(len(gate_lines(run_new(), "P8a")), 1)


class GateDrain(unittest.TestCase):
    """P6 gates on the calls over 1 ms, with at most 5, and on the longest call."""

    def edit_drain(self, **figures):
        def edit(report):
            report["shader_store_generation"]["drain"].update(figures)
        return run_new({"new_gen_first.json": edit})

    def test_baseline_passes(self):
        self.assertEqual(gate_line(run_new(), "P6"),
                         "P6 PASS drain max 1.50 ms, over 1 ms 1 (at most 5), over 4 ms 0")

    def test_six_calls_over_1ms_fail_with_a_short_max(self):
        out = self.edit_drain(over_1ms=6)
        self.assertEqual(gate_line(out, "P6"),
                         "P6 FAIL drain max 1.50 ms, over 1 ms 6 (at most 5), over 4 ms 0")

    def test_five_calls_over_1ms_pass(self):
        self.assertEqual(gate_line(self.edit_drain(over_1ms=5), "P6").split(" ")[1], "PASS")

    def test_calls_over_4ms_alone_do_not_fail(self):
        # Six calls over 4 ms with one over 1 ms is not possible in a real run, but the gate reads the
        # 1 ms counter, and this keeps the two apart.
        self.assertEqual(gate_line(self.edit_drain(over_4ms=6), "P6").split(" ")[1], "PASS")

    def test_a_slow_call_fails_and_over_8ms_is_hard(self):
        self.assertEqual(gate_line(self.edit_drain(max_ms=3.0), "P6").split(" ")[1], "FAIL")
        self.assertEqual(" ".join(gate_line(self.edit_drain(max_ms=9.0), "P6").split(" ")[1:3]), "HARD FAIL")


class GateWaitedTakes(unittest.TestCase):
    """P4a counts waited takes over 250 ms only. Inline takes are shown and not gated."""

    def test_baseline(self):
        out = run_new()
        self.assertEqual(gate_line(out, "P4a"), "P4a PASS waited takes over 250 ms, gen 2.0 vs gpl 1.0 + 2")
        self.assertEqual(gate_line(out, "P4d"),
                         "P4d INFO inline takes over 250 ms, gen 0.0 vs gpl 0.0 (not part of P4a)")

    def test_inline_takes_do_not_count_toward_p4a(self):
        # Two more slow takes, both inline. Waited stays 2 against a limit of 3, and a count that
        # included them would be 4.
        def add_inline(report):
            for flip, take_id in ((9, "0000000000000003"), (10, "00000000000000ee")):
                report["warm_takes"].append({"id": take_id, "how": "inline", "flip": flip,
                                             "stopped_ms": 400.0, "at_ms": 9000.0})
        out = run_new({"new_gen_first.json": add_inline})
        self.assertEqual(gate_line(out, "P4a"), "P4a PASS waited takes over 250 ms, gen 2.0 vs gpl 1.0 + 2")
        self.assertEqual(gate_line(out, "P4d"),
                         "P4d INFO inline takes over 250 ms, gen 2.0 vs gpl 0.0 (not part of P4a)")

    def test_the_limit_is_two_more_than_gpl(self):
        # Gpl has 1 waited take, so 3 pass and 4 fail. The baseline has 2.
        def add_waited(count):
            def edit(report):
                for flip in range(count):
                    report["warm_takes"].append({"id": "00000000000000ee", "how": "waited", "flip": 9 + flip,
                                                 "stopped_ms": 400.0, "at_ms": 9000.0})
            return edit
        self.assertEqual(gate_line(run_new({"new_gen_first.json": add_waited(1)}), "P4a").split(" ")[1], "PASS")
        self.assertEqual(gate_line(run_new({"new_gen_first.json": add_waited(2)}), "P4a").split(" ")[1], "FAIL")

    def test_the_table_keeps_inline_in_its_slow_columns(self):
        def add_inline(report):
            report["warm_takes"].append({"id": "00000000000000ee", "how": "inline", "flip": 9,
                                         "stopped_ms": 400.0, "at_ms": 9000.0})
        cells = row_cells(run_new({"new_gen_first.json": add_inline}, gate=None), "Audio", GEN_ARM)
        self.assertEqual(cells[5:7], ["1", "2"])


class GateGraphics(unittest.TestCase):
    """The graphics gate compares first runs, with 15 percent of room."""

    def test_within_the_allowance(self):
        out = run_new()
        self.assertEqual(gate_line(out, "GFX"),
                         "GFX PASS graphics blocked s, gen first run 11.00 vs gpl first run 10.00 + 15%")

    def test_over_the_allowance(self):
        def over(report):
            report["play"]["blocked_ms"]["graphics"] = 12000.0
        out = run_new({"new_gen_first.json": over})
        self.assertEqual(gate_line(out, "GFX").split(" ")[1], "FAIL")

    def test_second_boot_graphics_do_not_move_it(self):
        def huge(report):
            report["play"]["blocked_ms"]["graphics"] = 99000.0
        out = run_new({"new_gen_second.json": huge})
        self.assertEqual(gate_line(out, "GFX").split(" ")[1], "PASS")
        self.assertEqual(gate_line(out, "P8c").split(" ")[1], "FAIL")

    def test_p3_p95_is_gated_with_a_quarter_of_room(self):
        # Gen 600 against gpl 500 was a FAIL before the allowance. The limit is now 625, so it passes.
        out = run_new()
        self.assertEqual(gate_line(out, "P3b"),
                         "P3b PASS window_held p95 ms pooled over all runs, gen 600.00 (5 windows)"
                         " vs gpl 500.00 (5 windows), limit 625.00 (gpl x 1.25)")
        self.assertEqual(gate_line(out, "P3c").split(" ")[1], "PASS")

    def test_unknown_gpl_arm_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [os.path.join(FIXTURES, name) for name in NEW_FILES]
            code, _, err = run_script(paths, "--no-such-arm")
        self.assertEqual(code, 1)
        self.assertIn("no single gpl arm", err)


def set_windows(values):
    """An edit that gives a report these held times as its resolve windows."""
    def edit(report):
        report["deferred_draws"]["resolves"] = [
            {"flip": index, "window_held_ms": value, "window_hold_count_change": 1}
            for index, value in enumerate(values)]
    return edit


class GateP3(unittest.TestCase):
    """P3b pools the windows of every run of an arm, and P3d is information only."""

    def verdict_of(self, edits=None, extra=None):
        return gate_line(run_new(edits, extra=extra), "P3b").split(" ")[1]

    def test_boundary_of_the_allowance(self):
        # The gpl p95 is 500, so 625 is exactly on the limit and passes, and a hair over fails.
        self.assertEqual(self.verdict_of({"new_gen_first.json": set_windows([625.0])}), "PASS")
        self.assertEqual(self.verdict_of({"new_gen_first.json": set_windows([625.5])}), "FAIL")

    def test_pooled_windows_pass_where_the_median_of_run_p95s_fails(self):
        # Three gen runs. Two have 10 windows with one 1000 ms outlier, so their own p95 is 1000.
        # The third has 80 windows at 100 ms. Pooled that is 100 windows with 2 outliers, and the
        # p95 (rank 95) is 100, while the median of the three per-run p95s is 1000.
        outlier_run = [100.0] * 9 + [1000.0]
        extra = [("gen_b.json", read_fixture("new_gen_first.json")),
                 ("gen_c.json", read_fixture("new_gen_first.json"))]
        for _, report in extra:
            set_windows(outlier_run)(report)
        edits = {"new_gen_first.json": set_windows([100.0] * 80)}
        out = run_new(edits, extra=extra)
        self.assertEqual(gate_line(out, "P3b"),
                         "P3b PASS window_held p95 ms pooled over all runs, gen 100.00 (100 windows)"
                         " vs gpl 500.00 (5 windows), limit 625.00 (gpl x 1.25)")

    def test_pooled_windows_fail_when_enough_outliers_are_pooled(self):
        # The baseline of the test above, with the big run holding four outliers instead of none, so
        # 6 of 100 windows are 1000 ms and rank 95 is an outlier. Only the outlier count differs.
        extra = [("gen_b.json", read_fixture("new_gen_first.json")),
                 ("gen_c.json", read_fixture("new_gen_first.json"))]
        for _, report in extra:
            set_windows([100.0] * 9 + [1000.0])(report)
        edits = {"new_gen_first.json": set_windows([100.0] * 76 + [1000.0] * 4)}
        self.assertEqual(self.verdict_of(edits, extra), "FAIL")

    def test_hold_mean_is_information_only(self):
        # A gen mean far above gpl's, and one far below, both print INFO.
        def high(report):
            for row in report["deferred_draws"]["resolves"]:
                row["window_hold_count_change"] = 99
        def low(report):
            for row in report["deferred_draws"]["resolves"]:
                row["window_hold_count_change"] = 0
        for edit in (high, low):
            line = gate_line(run_new({"new_gen_first.json": edit}), "P3d")
            self.assertEqual(line.split(" ")[1], "INFO")
        self.assertIn("gen 99.00 vs gpl 3.00", gate_line(run_new({"new_gen_first.json": high}), "P3d"))


def set_stopped(gen_ms=None, gpl_ms=None):
    """Edits that set the longest stopped_ms of the gen first run and of the gpl first run. Only the
    first take of each is changed, and every other take stays below the new figure."""
    edits = {}
    if gen_ms is not None:
        edits["new_gen_first.json"] = lambda report: report["warm_takes"][0].update(stopped_ms=gen_ms)
    if gpl_ms is not None:
        edits["new_gpl_first.json"] = lambda report: report["warm_takes"][0].update(stopped_ms=gpl_ms)
    return edits


class GateP4b(unittest.TestCase):
    """P4b passes up to the larger of 1.25 x gpl and 500 ms, and P4e shows like for like create times."""

    def verdict_of(self, gen_ms, gpl_ms):
        return gate_line(run_new(set_stopped(gen_ms, gpl_ms)), "P4b").split(" ")[1]

    def test_the_floor_lets_a_small_gpl_through(self):
        # With gpl at 0 the ratio alone would allow nothing. The other gen takes are 300 and below.
        self.assertEqual(self.verdict_of(400.0, 0.0), "PASS")
        self.assertEqual(self.verdict_of(500.0, 0.0), "PASS")
        self.assertEqual(self.verdict_of(600.0, 0.0), "FAIL")

    def test_the_ratio_applies_above_the_floor(self):
        self.assertEqual(self.verdict_of(1250.0, 1000.0), "PASS")
        self.assertEqual(self.verdict_of(1251.0, 1000.0), "FAIL")

    def like_for_like(self, gpl_rows, extra=None):
        def add_rows(report):
            report["compiles"].extend(gpl_rows)
        return gate_lines(run_new({"new_gpl_first.json": add_rows}, extra=extra), "P4e")

    def test_like_for_like_shows_the_gpl_create_time_of_the_same_id(self):
        row = {"kind": "compute", "id": "0000000000000001", "create_ms": 80.0, "thread": "gpu"}
        self.assertEqual(self.like_for_like([row]),
                         ["P4e INFO like for like, gen take 0000000000000001 stopped 500.0 ms,"
                          " gpl create_ms for the same id 80.0 ms"])

    def test_like_for_like_is_n_a_without_a_gpl_row(self):
        # Baseline of the test above, the row left out. Another id's row must not stand in.
        other = {"kind": "compute", "id": "0000000000000002", "create_ms": 80.0, "thread": "gpu"}
        self.assertEqual(self.like_for_like([other]),
                         ["P4e INFO like for like, gen take 0000000000000001 stopped 500.0 ms,"
                          " gpl create_ms for the same id n/a"])

    def test_like_for_like_matches_ids_as_text(self):
        # The id 1 is the same number as 0000000000000001 and a different text, so its row is no match.
        # Parsed as numbers the 999 would win as the longest.
        rows = [{"kind": "compute", "id": "0000000000000001", "create_ms": 80.0, "thread": "gpu"},
                {"kind": "compute", "id": "1", "create_ms": 999.0, "thread": "gpu"}]
        self.assertIn("same id 80.0 ms", self.like_for_like(rows)[0])

    def test_like_for_like_takes_the_median_over_gpl_runs(self):
        row = lambda create_ms: {"kind": "compute", "id": "0000000000000001", "create_ms": create_ms,
                                 "thread": "gpu"}
        second = read_fixture("new_gpl_first.json")
        second["compiles"].append(row(100.0))
        lines = self.like_for_like([row(80.0)], extra=[("gpl_b.json", second)])
        self.assertIn("same id 90.0 ms", lines[0])


class GateP5Events(unittest.TestCase):
    """P5 counts events. A submit no more than 50 ms after the previous one joins its event, so a gap
    of exactly 50 ms is the same event and anything longer is a new one."""

    def counts(self, at_ms, code="P5a"):
        """The event and submit figures of the gen arm in the window, read from the line."""
        line = gate_line(run_new({"new_gen_first.json": lambda report: report["audio_underruns"].update(at_ms=at_ms)}),
                         code)
        gen_events = line.split(" gen ")[1].split(" vs")[0]
        gen_submits = line.split("(submits, gen ")[1].split(" vs")[0]
        return int(gen_events), int(gen_submits)

    def test_three_submits_within_half_a_millisecond_are_one_event(self):
        self.assertEqual(self.counts([100.0, 100.2, 100.5]), (1, 3))

    def test_submits_spaced_out_are_separate_events(self):
        # Baseline of the test above. The same three submits, 100 ms apart.
        self.assertEqual(self.counts([100.0, 200.0, 300.0]), (3, 3))

    def test_two_submits_60_ms_apart_are_two_events(self):
        self.assertEqual(self.counts([100.0, 160.0]), (2, 2))

    def test_a_gap_of_exactly_50_ms_joins_the_event(self):
        self.assertEqual(self.counts([100.0, 150.0]), (1, 2))
        self.assertEqual(self.counts([100.0, 150.5]), (2, 2))

    def test_a_chain_of_short_gaps_is_one_event(self):
        # Each gap is 40 ms from the previous submit, though the burst spans 120 ms in all.
        self.assertEqual(self.counts([100.0, 140.0, 180.0, 220.0]), (1, 4))

    def test_events_are_counted_after_the_cut_too(self):
        self.assertEqual(self.counts([11000.0, 11000.1, 11000.3], "P5b"), (1, 3))
        self.assertEqual(self.counts([11000.0, 11100.0, 11200.0], "P5b"), (3, 3))

    def test_a_burst_across_the_cut_counts_once_on_each_side(self):
        # The cut is 10700 ms. The submits are 20 ms apart and are split at the cut first.
        self.assertEqual(self.counts([10690.0, 10710.0], "P5a"), (1, 1))
        self.assertEqual(self.counts([10690.0, 10710.0], "P5b"), (1, 1))


class GateP7(unittest.TestCase):
    """P7 passes when the decode maximum is under 50 ms."""

    def line(self, decode_ms):
        def edit(report):
            report["shader_store_generation"]["supervisor"]["decode_ms_max"] = decode_ms
        return gate_line(run_new({"new_gen_first.json": edit}), "P7")

    def test_just_under_passes(self):
        self.assertEqual(self.line(49.9), "P7 PASS decode max 49.90 ms")

    def test_exactly_50_fails(self):
        self.assertEqual(self.line(50.0), "P7 FAIL decode max 50.00 ms")


TIER_TITLE = "Generated by tier"


def module(module_id, code_hash, posted, adopted=0.0, how="done", ready=None, build=None):
    """A generated module row. Ready and build are left out for a module that was never built."""
    row = {"id": module_id, "code_hash": code_hash, "posted_ms": posted, "adopted_ms": adopted, "how": how}
    if ready is not None:
        row["ready_ms"] = ready
    if build is not None:
        row["build_ms"] = build
    return row


def set_modules(rows):
    """An edit that gives the gen first run exactly these generated modules."""
    def edit(report):
        report["shader_store_generation"]["generated_modules"] = list(rows)
    return edit


def tier_note(output, text):
    """The note line of the gen first run's arm that contains the text."""
    return next(line.strip() for line in output.splitlines()
                if line.startswith(f"  {GEN_ARM}: ") and text in line)


def tier_cells(rows, key="tier 0", extra=None):
    """The cells of the gen first run's tier row for modules, in an otherwise unchanged run."""
    out = run_new({"new_gen_first.json": set_modules(rows)}, gate=None, extra=extra)
    return row_cells(out, TIER_TITLE, f"{GEN_ARM} {key}")


class VariantTiers(unittest.TestCase):
    """The tier is the rank among modules of one code hash in one run, in posted order."""

    def test_ranked_by_posted_time(self):
        rows = [module("a", "h1", 300.0), module("b", "h1", 100.0), module("c", "h1", 200.0)]
        self.assertEqual(SCRIPT.variant_tiers(rows), [2, 0, 1])

    def test_a_posted_tie_goes_to_the_earlier_adoption(self):
        # Baseline, the same two modules posted apart. Only the posted times differ.
        tie = [module("a", "h1", 100.0, adopted=20.0), module("b", "h1", 100.0, adopted=10.0)]
        apart = [module("a", "h1", 100.0, adopted=20.0), module("b", "h1", 101.0, adopted=10.0)]
        self.assertEqual(SCRIPT.variant_tiers(tie), [1, 0])
        self.assertEqual(SCRIPT.variant_tiers(apart), [0, 1])

    def test_a_tie_in_both_goes_to_the_id_as_text(self):
        rows = [module("b", "h1", 100.0, 10.0), module("a", "h1", 100.0, 10.0)]
        self.assertEqual(SCRIPT.variant_tiers(rows), [1, 0])
        # As text 10 sorts before 9, and as numbers it would not.
        rows = [module("9", "h1", 100.0, 10.0), module("10", "h1", 100.0, 10.0)]
        self.assertEqual(SCRIPT.variant_tiers(rows), [1, 0])

    def test_the_order_is_not_the_id_or_the_adoption_alone(self):
        # Posted order runs against both the id order and the adopted order.
        rows = [module("a", "h1", 200.0, adopted=1.0), module("b", "h1", 100.0, adopted=2.0)]
        self.assertEqual(SCRIPT.variant_tiers(rows), [1, 0])

    def test_a_single_variant_program_is_tier_zero(self):
        rows = [module("a", "h1", 500.0), module("b", "h2", 100.0), module("c", "h2", 200.0)]
        self.assertEqual(SCRIPT.variant_tiers(rows), [0, 0, 1])

    def test_tiers_are_per_run_and_not_counted_across_runs(self):
        # The same modules in two runs. Counted across runs the second run's would be tier 1.
        rows = [module("a", "h1", 100.0, how="done"), module("b", "h1", 200.0, how="unused")]
        second = read_fixture("new_gen_first.json")
        set_modules(rows)(second)
        cells = tier_cells(rows, "tier 1+", extra=[("new_gen_first_b.json", second)])
        self.assertEqual(cells[:4], ["1", "0", "1", "1"])
        self.assertEqual(tier_cells(rows, "tier 0", extra=[("new_gen_first_b.json", second)])[:4],
                         ["1", "1", "0", "1"])


class TierTable(unittest.TestCase):
    """The table counts used, unused and not built modules, and the start delay."""

    def test_start_delay_is_ready_minus_build_minus_adopted(self):
        # Delays 150-30-100=20 and 70-10-0=60, median 40. Not subtracting the build would give 60.
        rows = [module("a", "h1", 1.0, adopted=100.0, ready=150.0, build=30.0),
                module("b", "h2", 1.0, adopted=0.0, ready=70.0, build=10.0)]
        self.assertEqual(tier_cells(rows), ["2", "2", "0", "0", "0.00", "40.0"])

    def test_a_not_built_row_is_counted_and_left_out_of_the_delay(self):
        # The baseline above plus a used row with no ready_ms. Modules and not built move, and the
        # delay, used and unused counts do not, so it is not unused and not in the median.
        rows = [module("a", "h1", 1.0, adopted=100.0, ready=150.0, build=30.0),
                module("b", "h2", 1.0, adopted=0.0, ready=70.0, build=10.0),
                module("c", "h3", 1.0, adopted=0.0, build=500.0)]
        self.assertEqual(tier_cells(rows), ["3", "3", "0", "1", "0.00", "40.0"])

    def test_a_row_missing_build_ms_is_not_built(self):
        rows = [module("a", "h1", 1.0, adopted=0.0, ready=70.0)]
        self.assertEqual(tier_cells(rows), ["1", "1", "0", "1", "0.00", "-"])

    def test_unused_build_is_summed_over_unused_rows(self):
        rows = [module("a", "h1", 1.0, how="unused", ready=2000.0, build=1500.0),
                module("b", "h2", 1.0, how="unused", build=500.0),
                module("c", "h3", 1.0, how="done", ready=100.0, build=900.0)]
        self.assertEqual(tier_cells(rows)[:5], ["3", "1", "2", "1", "2.00"])

    def test_the_two_tiers_are_split(self):
        rows = [module("a", "h1", 1.0), module("b", "h1", 2.0, how="unused", build=700.0)]
        self.assertEqual(tier_cells(rows, "tier 0")[:5], ["1", "1", "0", "1", "0.00"])
        self.assertEqual(tier_cells(rows, "tier 1+")[:5], ["1", "0", "1", "1", "0.70"])

    def test_no_table_without_the_block_or_for_old_reports(self):
        without = run_new({"new_gen_first.json": lambda report: report.pop("shader_store_generation")}, gate=None)
        self.assertNotIn(TIER_TITLE, without)
        old = [os.path.join(FIXTURES, "old_baseline.json"), os.path.join(FIXTURES, "old_gpl.json")]
        self.assertNotIn(TIER_TITLE, run_script(old)[1])
        # The baseline with the block does print it.
        self.assertIn(TIER_TITLE, run_new(gate=None))

    def test_the_table_follows_the_generation_table(self):
        out = run_new(gate=None)
        self.assertLess(out.index("Generation"), out.index(TIER_TITLE))
        self.assertLess(out.index(TIER_TITLE), out.index("Audio"))


def shaped_modules(tier0_used, tier1_used, unused_tier0_build=0.0, unused_tier1_build=1500.0):
    """Modules for the go boundary. Each tier 0 module is its own program, tier 1 modules are second
    variants of the first programs, and one unused tier 1 module carries the unused build time."""
    rows = [module(f"t0_{index}", f"h{index}", 1.0) for index in range(tier0_used)]
    rows += [module(f"t1_{index}", f"h{index}", 2.0) for index in range(tier1_used)]
    rows.append(module("u1", "h0", 3.0, how="unused", build=unused_tier1_build))
    if unused_tier0_build:
        rows.append(module("u0", "hu", 1.0, how="unused", build=unused_tier0_build))
    return rows


class M1PreCheck(unittest.TestCase):
    """Go needs 90 percent of used modules at tier 0 and most unused build time at tier 1+."""

    def check(self, rows):
        return tier_note(run_new({"new_gen_first.json": set_modules(rows)}, gate=None), "M1 pre-check")

    def test_exactly_90_percent_is_go(self):
        out = run_new({"new_gen_first.json": set_modules(shaped_modules(90, 10))}, gate=None)
        self.assertEqual(tier_note(out, "used modules at tier 0"),
                         f"{GEN_ARM}: used modules at tier 0 (runs pooled): 90.0%")
        self.assertTrue(self.check(shaped_modules(90, 10)).split(": ", 1)[1].startswith("M1 pre-check: go (INFO)"))

    def test_just_under_90_percent_is_no_go(self):
        self.assertTrue(self.check(shaped_modules(89, 11)).split(": ", 1)[1].startswith("M1 pre-check: no-go (INFO)"))
        self.assertIn("used at tier 0 89.0%", self.check(shaped_modules(89, 11)))

    def test_most_of_the_unused_build_must_be_tier_1_plus(self):
        # Half is not most. Tier 1+ at 500 against 500 is 50 percent, and 501 is over it.
        self.assertIn("M1 pre-check: no-go", self.check(shaped_modules(90, 10, 500.0, 500.0)))
        self.assertIn("M1 pre-check: go", self.check(shaped_modules(90, 10, 500.0, 501.0)))

    def test_the_unused_share_line(self):
        out = run_new({"new_gen_first.json": set_modules(shaped_modules(90, 10, 500.0, 1500.0))}, gate=None)
        self.assertEqual(tier_note(out, "unused build time at tier 1+"),
                         f"{GEN_ARM}: unused build time at tier 1+ (runs pooled): 75.0%")

    def test_the_shares_are_pooled_over_the_runs(self):
        # Three runs with 100, 100 and 20 percent of used modules at tier 0. The median of those is
        # 100 and would say go, while pooled it is 4 of 12 (33.3 percent) and says no-go.
        # The third run has two programs with 4 later used variants each, then the usual unused one.
        busy = [module(f"t0_{index}", f"h{index}", 1.0) for index in range(2)]
        busy += [module(f"v{index}_{rank}", f"h{index}", 10.0 + rank) for index in range(2) for rank in range(4)]
        busy.append(module("u1", "h0", 99.0, how="unused", build=1500.0))
        extra = []
        for name, rows in (("gen_b.json", shaped_modules(1, 0)), ("gen_c.json", busy)):
            report = read_fixture("new_gen_first.json")
            set_modules(rows)(report)
            extra.append((name, report))
        out = run_new({"new_gen_first.json": set_modules(shaped_modules(1, 0))}, gate=None, extra=extra)
        self.assertEqual(tier_note(out, "used modules at tier 0"),
                         f"{GEN_ARM}: used modules at tier 0 (runs pooled): 33.3%")
        self.assertIn("M1 pre-check: no-go", tier_note(out, "M1 pre-check"))
        # Unused build pooled is 3 runs of 1500 at tier 1+ and none at tier 0, so 100 percent.
        self.assertIn("(runs pooled): 100.0%", tier_note(out, "unused build time at tier 1+"))

    def test_no_used_modules_is_not_judged(self):
        rows = [module("a", "h1", 1.0, how="unused", build=5.0)]
        self.assertIn("M1 pre-check: n/a", self.check(rows))


class ProgramTiers(unittest.TestCase):
    """The per-program line shows which tiers were used, with the hash as text."""

    def lines(self, rows):
        out = run_new({"new_gen_first.json": set_modules(rows)}, gate=None)
        return [line.strip() for line in out.splitlines() if "used tiers" in line]

    def test_used_tiers_are_listed(self):
        rows = [module("a", "00000000000000aa", 1.0), module("b", "00000000000000aa", 2.0),
                module("c", "1e00000000000005", 1.0, how="unused"),
                module("d", "1e00000000000005", 2.0, how="done"),
                module("e", "1e00000000000005", 3.0, how="unused"),
                module("f", "single", 1.0)]
        self.assertEqual(self.lines(rows),
                         [f"{GEN_ARM}: 00000000000000aa: used tiers 0,1 of 2",
                          f"{GEN_ARM}: 1e00000000000005: used tiers 1 of 3"])

    def test_leading_zeros_are_kept_in_the_line(self):
        rows = [module("a", "0000000000000a1b", 1.0), module("b", "0000000000000a1b", 2.0)]
        self.assertEqual(self.lines(rows), [f"{GEN_ARM}: 0000000000000a1b: used tiers 0,1 of 2"])

    def test_hashes_that_differ_only_in_leading_zeros_are_two_programs(self):
        # Read as numbers 0001 and 1 are one program of 4 variants. As text each has 2.
        rows = [module("a", "0001", 1.0), module("b", "0001", 2.0),
                module("c", "1", 1.0), module("d", "1", 2.0, how="unused")]
        self.assertEqual(self.lines(rows), [f"{GEN_ARM}: 0001: used tiers 0,1 of 2",
                                            f"{GEN_ARM}: 1: used tiers 0 of 2"])
        self.assertEqual(SCRIPT.variant_tiers(rows), [0, 1, 0, 1])

    def test_a_program_with_no_used_variant_says_none(self):
        rows = [module("a", "h1", 1.0, how="unused"), module("b", "h1", 2.0, how="unused")]
        self.assertEqual(self.lines(rows), [f"{GEN_ARM}: h1: used tiers none of 2"])

    def test_single_variant_programs_have_no_line(self):
        self.assertEqual(self.lines([module("a", "h1", 1.0), module("b", "h2", 1.0)]), [])


class PeakSources(unittest.TestCase):
    """Peaks from different sources are never compared or combined."""

    def test_sources_that_differ_are_not_compared(self):
        out = run_new()
        self.assertEqual(gate_line(out, "MEM")[:20], "MEM INFO child peak ")
        mem = gate_lines(out, "MEM")
        self.assertEqual(mem[0], "MEM INFO child peak first run vs second boot n/a (sources differ)")
        self.assertEqual(mem[1], "MEM INFO emulator peak gen vs gpl n/a (sources differ)")
        self.assertNotIn("4096.0 MB vs", out)

    def test_same_source_is_compared(self):
        # Baseline for the refusal. Only the source names change, and the figures then appear.
        def same_child(report):
            report["shader_store_generation"]["supervisor"]["peak_source"] = "working_set"
        def same_emulator(report):
            report["process"] = {"peak_working_set_bytes": 4294967296}
        out = run_new({"new_gen_second.json": same_child, "new_gen_first.json": same_emulator})
        mem = gate_lines(out, "MEM")
        self.assertEqual(mem[0], "MEM INFO child peak first run vs second boot 100.0 MB vs 90.0 MB (working_set)")
        self.assertEqual(mem[1], "MEM INFO emulator peak gen vs gpl 4096.0 MB vs 2048.0 MB (peak_working_set_bytes)")

    def test_runs_of_one_arm_with_different_sources_are_not_merged(self):
        def other_source(report):
            report["shader_store_generation"]["supervisor"]["peak_source"] = "commit"
        second_run = read_fixture("new_gen_first.json")
        other_source(second_run)
        out = run_new(extra=[("new_gen_first_b.json", second_run)], gate=None)
        row = next(r for r in table_rows(out, "Generation") if r.startswith(GEN_ARM + "  "))
        self.assertTrue(row.rstrip().endswith("n/a (sources differ)"))
        same = read_fixture("new_gen_first.json")
        out = run_new(extra=[("new_gen_first_b.json", same)], gate=None)
        row = next(r for r in table_rows(out, "Generation") if r.startswith(GEN_ARM + "  "))
        self.assertTrue(row.rstrip().endswith("100.0 MB (working_set)"))

    def test_the_source_is_printed_beside_every_figure(self):
        out = run_new(gate=None)
        self.assertIn("100.0 MB (working_set)", out)
        self.assertIn("90.0 MB (commit)", out)
        self.assertIn("2048.0 MB (peak_working_set_bytes)", out)


class HashText(unittest.TestCase):
    """Hash fields are text and are never read as numbers."""

    def order_row(self, order):
        def set_order(report):
            report["shader_store_generation"]["worker"]["order"] = order
        return run_new({"new_gen_first.json": set_order}, gate=None)

    def test_digit_only_order_keeps_its_zeros(self):
        out = self.order_row("0000000000000012")
        self.assertIn(f"{GEN_ARM}: worker order 0000000000000012\n", out)

    def test_order_that_reads_as_a_float_stays_text(self):
        out = self.order_row("1e00000000000001")
        self.assertIn(f"{GEN_ARM}: worker order 1e00000000000001\n", out)

    def test_ids_join_as_text(self):
        # Module 0000000000000001 is generated, so its slow take counts as generated. The player's
        # id 00000000000000aa is not in the module list. Baseline, the id edited to another text.
        out = run_new(gate=None)
        self.assertEqual(row_cells(out, "Audio", GEN_ARM)[5:7], ["1", "1"])
        def other_id(report):
            report["warm_takes"][0]["id"] = "0000000000000099"
        out = run_new({"new_gen_first.json": other_id}, gate=None)
        self.assertEqual(row_cells(out, "Audio", GEN_ARM)[5:7], ["0", "2"])


if __name__ == "__main__":
    unittest.main()
