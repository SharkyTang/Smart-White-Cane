#!/usr/bin/env python3
"""Render an acceptance report from Smart Cane LiDAR metadata JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from smart_cane.validation import render_validation_markdown, summarize_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="metadata-only JSONL file(s)")
    parser.add_argument("--output", type=Path, help="write Markdown here instead of stdout")
    return parser


def load_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: record must be a JSON object")
                records.append(value)
    return records


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = load_records(args.inputs)
    except (OSError, ValueError) as error:
        print(f"analyze-lidar-validation: {error}", file=sys.stderr)
        return 2

    report = render_validation_markdown(summarize_validation(records))
    if args.output is None:
        print(report, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
