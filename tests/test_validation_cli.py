from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_lidar_validation import load_records, main


class ValidationCliTests(unittest.TestCase):
    def test_loads_multiple_metadata_only_jsonl_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.jsonl"
            second = root / "second.jsonl"
            first.write_text('{"record_type":"frame","frame_id":1}\n\n', encoding="utf-8")
            second.write_text('{"record_type":"speech","alert_id":"a1"}\n', encoding="utf-8")

            records = load_records([first, second])

        self.assertEqual([record["record_type"] for record in records], ["frame", "speech"])

    def test_main_reports_invalid_json_without_creating_a_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "broken.jsonl"
            output = root / "report.md"
            source.write_text("not-json\n", encoding="utf-8")
            error = io.StringIO()

            with contextlib.redirect_stderr(error):
                result = main([str(source), "--output", str(output)])

            self.assertEqual(result, 2)
            self.assertIn("broken.jsonl:1: invalid JSON", error.getvalue())
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
