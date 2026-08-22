from __future__ import annotations

import contextlib
import io
import unittest

from smart_cane.app import RunSummary, run_demo
from smart_cane.cli import main
from smart_cane.config import RuntimeConfig


class AppTests(unittest.TestCase):
    def test_demo_runs_without_external_services(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assessments = run_demo(RuntimeConfig(display=False))

        self.assertEqual(len(assessments), 3)
        self.assertIn("Most urgent: Danger: chair ahead.", output.getvalue())

    def test_cli_demo_returns_success(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["--demo", "--no-display"])

        self.assertEqual(result, 0)
        self.assertIn("deterministic demo", output.getvalue())

    def test_run_summary_calculates_fps(self) -> None:
        summary = RunSummary(frames=30, detections=5, alerts=1, elapsed_seconds=2.0)
        self.assertEqual(summary.average_fps, 15.0)


if __name__ == "__main__":
    unittest.main()
