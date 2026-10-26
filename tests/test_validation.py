from __future__ import annotations

import unittest

from smart_cane.validation import render_validation_markdown, summarize_validation


def frame(
    frame_id: int,
    *,
    measured: float | None,
    truth: float = 1.0,
    timestamp: float = 0.0,
    rtt_ms: float = 20.0,
    expected_alert: bool = True,
    emit_alert: bool = True,
    alert_id: str = "alert-1",
    validation_group: str = "ordinary",
    session_id: str = "session-1",
    run_id: str = "run-1",
    expected_alert_event_id: str | None = "box-1m-alert",
) -> dict[str, object]:
    assessment = {
        "track_key": "person:1",
        "label": "person",
        "direction": "center",
        "level": "high",
        "distance_meters": measured,
        "distance_status": "valid" if measured is not None else "insufficient_samples",
        "risk_source": "lidar" if measured is not None else "area_fallback",
    }
    return {
        "record_type": "frame",
        "session_id": session_id,
        "run_id": run_id,
        "scenario_id": "box-1m",
        "validation_group": validation_group,
        "ground_truth_meters": truth,
        "frame_id": frame_id,
        "response_frame_id": frame_id,
        "capture_timestamp_seconds": timestamp,
        "rtt_ms": rtt_ms,
        "expected_alert": expected_alert,
        "expected_alert_event_id": expected_alert_event_id,
        "assessments": [assessment],
        "alert": {
            "alert_id": alert_id,
            "should_emit": emit_alert,
            "track_key": "person:1",
            "label": "person",
            "direction": "center",
            "level": "high",
            "risk_source": assessment["risk_source"],
            "distance_status": assessment["distance_status"],
        },
    }


class ValidationSummaryTests(unittest.TestCase):
    def test_calculates_acceptance_metrics_from_metadata_only_records(self) -> None:
        records = []
        for run_number in range(1, 4):
            session_id = f"session-{run_number}"
            run_id = f"run-{run_number}"
            records.extend(
                frame(
                    index,
                    measured=(0.90, 1.05, 1.10)[index % 3],
                    timestamp=(index - 1) * 0.125,
                    expected_alert=run_number == 1,
                    emit_alert=run_number == 1 and index in {1, 26},
                    alert_id="alert-1" if index == 1 else "alert-2",
                    session_id=session_id,
                    run_id=run_id,
                )
                for index in range(1, 31)
            )
        records.extend(
            [
                {
                    "record_type": "speech",
                    "session_id": "session-1",
                    "frame_id": 1,
                    "alert_id": "alert-1",
                    "speech_start_latency_ms": 410.0,
                    "spoken_distance_meters": 0.9,
                    "distance_status": "valid",
                    "risk_source": "lidar",
                    "track_key": "person:1",
                    "label": "person",
                    "direction": "center",
                    "level": "high",
                },
                {
                    "record_type": "speech",
                    "session_id": "session-1",
                    "frame_id": 26,
                    "alert_id": "alert-2",
                    "speech_start_latency_ms": 450.0,
                    "spoken_distance_meters": 1.1,
                    "distance_status": "valid",
                    "risk_source": "lidar",
                    "track_key": "person:1",
                    "label": "person",
                    "direction": "center",
                    "level": "high",
                },
            ]
        )

        summary = summarize_validation(records)

        self.assertEqual(summary.frame_count, 90)
        self.assertEqual(summary.valid_lidar_count, 90)
        self.assertAlmostEqual(summary.median_absolute_error_m.value, 0.1)
        self.assertAlmostEqual(summary.p95_absolute_error_m.value, 0.1)
        self.assertEqual(summary.distance_coverage.value, 1.0)
        self.assertEqual(summary.effective_hz.value, 8.0)
        self.assertEqual(summary.cooldown_violations.value, 0)
        self.assertEqual(summary.scenarios_below_30_samples.value, 0)
        self.assertEqual(summary.scenarios_below_3_runs.value, 0)
        self.assertTrue(summary.passed)
        report = render_validation_markdown(summary)
        self.assertIn("Metadata gates: **PASS**", report)
        self.assertIn("NOT DETERMINED BY THIS REPORT ALONE", report)

    def test_reports_failures_and_missing_measurements_without_inventing_values(self) -> None:
        bad = frame(1, measured=None, expected_alert=True)
        bad["response_frame_id"] = 99
        bad["alert"] = {"should_emit": False}
        records = [
            bad,
            {
                "record_type": "speech",
                "alert_id": "alert-1",
                "speech_start_latency_ms": 900.0,
                "spoken_distance_meters": 1.0,
                "distance_status": "stale",
                "risk_source": "lidar",
                "level": "high",
            },
        ]

        summary = summarize_validation(records)

        self.assertIsNone(summary.median_absolute_error_m.value)
        self.assertEqual(summary.distance_coverage.value, 0.0)
        self.assertEqual(summary.frame_id_mismatches.value, 1)
        self.assertEqual(summary.invalid_numeric_speeches.value, 1)
        self.assertEqual(summary.expected_alert_misses.value, 1)
        self.assertFalse(summary.passed)
        self.assertIn("NOT MEASURED", render_validation_markdown(summary))

    def test_boundary_surfaces_are_excluded_from_required_accuracy_gate(self) -> None:
        boundary = frame(
            1,
            measured=3.0,
            truth=1.0,
            validation_group="boundary",
            expected_alert=False,
        )

        summary = summarize_validation([boundary])

        self.assertIsNone(summary.distance_coverage.value)
        self.assertIsNone(summary.median_absolute_error_m.value)
        self.assertFalse(summary.passed)
        self.assertEqual(summary.scenarios_below_3_runs.value, 1)
        self.assertEqual(summary.scenarios["box-1m"]["validation_group"], "boundary")

    def test_does_not_merge_short_runs_or_accept_unnamed_expected_events(self) -> None:
        records = []
        for run_number in (1, 2):
            records.extend(
                frame(
                    index,
                    measured=1.0,
                    timestamp=(index - 1) * 0.125,
                    expected_alert_event_id=None,
                    emit_alert=False,
                    session_id=f"session-{run_number}",
                    run_id=f"run-{run_number}",
                )
                for index in range(1, 16)
            )
        for record in records:
            assessments = record["assessments"]
            assert isinstance(assessments, list)
            record["assessments"] = [*assessments, *assessments]

        summary = summarize_validation(records)

        self.assertEqual(summary.scenarios_below_30_samples.value, 2)
        self.assertEqual(summary.scenarios_below_3_runs.value, 1)
        self.assertEqual(summary.expected_alerts_missing_event_id.value, 30)
        self.assertFalse(summary.passed)

    def test_counts_distinct_expected_events_in_the_same_run(self) -> None:
        first = frame(
            1,
            measured=1.0,
            expected_alert_event_id="threshold-2m-enter",
            alert_id="alert-a",
        )
        second = frame(
            2,
            measured=0.8,
            expected_alert_event_id="threshold-1m-enter",
            alert_id="alert-b",
        )
        speech = {
            "record_type": "speech",
            "session_id": "session-1",
            "alert_id": "alert-a",
            "speech_start_latency_ms": 300.0,
            "distance_meters": 1.0,
            "distance_status": "valid",
            "risk_source": "lidar",
            "level": "high",
        }

        summary = summarize_validation([first, second, speech])

        self.assertEqual(summary.expected_alert_misses.value, 1)


if __name__ == "__main__":
    unittest.main()
