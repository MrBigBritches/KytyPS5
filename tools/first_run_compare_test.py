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
                         "P5a FAIL underruns before cut 10700 ms, gen 6 vs gpl 2, limit 4")
        self.assertEqual(gate_line(out, "P5b"),
                         "P5b PASS underruns after cut 10700 ms, gen 1 vs gpl 2, limit 4")

    def test_moving_the_finish_moves_the_cut(self):
        # Baseline for the cut. Only finished_at_ms differs, and the window counts follow it.
        def finish_early(report):
            report["shader_store_generation"]["supervisor"]["finished_at_ms"] = 400.0
        out = run_new({"new_gen_first.json": finish_early})
        self.assertEqual(gate_line(out, "P5a"),
                         "P5a PASS underruns before cut 400 ms, gen 0 vs gpl 0, limit 2")
        self.assertEqual(gate_line(out, "P5b"),
                         "P5b FAIL underruns after cut 400 ms, gen 7 vs gpl 4, limit 6")

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
            report["audio_underruns"]["at_ms"] = [float(index) for index in range(12)]
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

    def test_p3_p95_is_gated(self):
        out = run_new()
        self.assertEqual(gate_line(out, "P3b"), "P3b FAIL window_held p95 ms, gen 600.00 vs gpl 500.00")
        self.assertEqual(gate_line(out, "P3c").split(" ")[1], "PASS")

    def test_unknown_gpl_arm_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [os.path.join(FIXTURES, name) for name in NEW_FILES]
            code, _, err = run_script(paths, "--no-such-arm")
        self.assertEqual(code, 1)
        self.assertIn("no single gpl arm", err)


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
