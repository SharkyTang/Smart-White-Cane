from __future__ import annotations

import math
import unittest

from smart_cane.metrics import FrameMetrics, MetricsCollector, STAGE_NAMES


def frame(
    *,
    capture: float = 0.001,
    inference: float = 0.010,
    decision: float = 0.002,
    render: float = 0.003,
    total: float = 0.020,
) -> FrameMetrics:
    return FrameMetrics(
        capture_seconds=capture,
        inference_seconds=inference,
        decision_seconds=decision,
        render_seconds=render,
        total_seconds=total,
    )


class FrameMetricsTests(unittest.TestCase):
    def test_normalizes_numeric_durations_to_float(self) -> None:
        result = FrameMetrics(1, 2, 3, 4, 5)

        self.assertEqual(
            (
                result.capture_seconds,
                result.inference_seconds,
                result.decision_seconds,
                result.render_seconds,
                result.total_seconds,
            ),
            (1.0, 2.0, 3.0, 4.0, 5.0),
        )

    def test_rejects_negative_non_finite_and_boolean_durations(self) -> None:
        invalid_values = (-0.001, math.inf, -math.inf, math.nan, True)

        for invalid in invalid_values:
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "capture_seconds"):
                    frame(capture=invalid)


class MetricsCollectorTests(unittest.TestCase):
    def test_add_accepts_frame_or_duration_keywords(self) -> None:
        collector = MetricsCollector()
        existing = frame()

        returned_existing = collector.add(existing)
        returned_keywords = collector.add(
            capture_seconds=0.002,
            inference_seconds=0.012,
            decision_seconds=0.003,
            render_seconds=0.004,
            total_seconds=0.025,
        )

        self.assertIs(returned_existing, existing)
        self.assertIsInstance(returned_keywords, FrameMetrics)
        self.assertEqual(returned_keywords.inference_seconds, 0.012)
        self.assertEqual(collector.frame_count, 2)

    def test_add_rejects_incomplete_mixed_and_wrong_input_forms(self) -> None:
        collector = MetricsCollector()

        with self.assertRaisesRegex(TypeError, "missing required duration"):
            collector.add(capture_seconds=0.1)
        with self.assertRaisesRegex(TypeError, "either metrics or duration"):
            collector.add(
                frame(),
                capture_seconds=0.1,
                inference_seconds=0.1,
                decision_seconds=0.1,
                render_seconds=0.1,
                total_seconds=0.5,
            )
        with self.assertRaisesRegex(TypeError, "FrameMetrics"):
            collector.add("not metrics")  # type: ignore[arg-type]

        self.assertEqual(collector.frame_count, 0)

    def test_excludes_leading_warmup_from_fps_and_stage_statistics(self) -> None:
        collector = MetricsCollector(warmup_frames=2)
        collector.add(frame(capture=9.0, inference=9.0, decision=9.0, render=9.0, total=9.0))
        collector.add(frame(capture=8.0, inference=8.0, decision=8.0, render=8.0, total=8.0))
        collector.add(
            frame(capture=0.001, inference=0.010, decision=0.002, render=0.003, total=0.020)
        )
        collector.add(
            frame(capture=0.003, inference=0.030, decision=0.004, render=0.005, total=0.030)
        )

        summary = collector.summary()

        self.assertEqual(summary.total_frames, 4)
        self.assertEqual(summary.warmup_frames, 2)
        self.assertEqual(summary.steady_frames, 2)
        self.assertAlmostEqual(summary.steady_average_fps, 40.0)
        self.assertEqual(summary.stage("capture").samples, 2)
        self.assertAlmostEqual(summary.stage("capture").mean_ms, 2.0)
        self.assertAlmostEqual(summary.stage("capture").median_ms, 2.0)
        self.assertAlmostEqual(summary.stage("capture").p95_ms, 3.0)
        self.assertAlmostEqual(summary.stage("inference").mean_ms, 20.0)
        self.assertAlmostEqual(summary.stage("total").mean_ms, 25.0)

    def test_p95_uses_deterministic_nearest_rank_without_interpolation(self) -> None:
        collector = MetricsCollector()
        for milliseconds in range(1, 21):
            seconds = milliseconds / 1000.0
            collector.add(
                capture_seconds=seconds,
                inference_seconds=seconds,
                decision_seconds=seconds,
                render_seconds=seconds,
                total_seconds=seconds,
            )

        capture = collector.summary().stage("capture")

        self.assertEqual(capture.samples, 20)
        self.assertAlmostEqual(capture.mean_ms, 10.5)
        self.assertAlmostEqual(capture.median_ms, 10.5)
        self.assertAlmostEqual(capture.p95_ms, 19.0)

    def test_empty_summary_has_all_stages_and_zero_values(self) -> None:
        summary = MetricsCollector(warmup_frames=5).summary()

        self.assertEqual(
            (summary.total_frames, summary.warmup_frames, summary.steady_frames),
            (0, 0, 0),
        )
        self.assertEqual(summary.steady_average_fps, 0.0)
        self.assertEqual(tuple(summary.stages), STAGE_NAMES)
        for statistics in summary.stages.values():
            self.assertEqual(
                (statistics.samples, statistics.mean_ms, statistics.median_ms, statistics.p95_ms),
                (0, 0.0, 0.0, 0.0),
            )

    def test_only_warmup_frames_produce_a_stable_zero_steady_summary(self) -> None:
        collector = MetricsCollector(warmup_frames=3)
        collector.add(frame(total=0.0))
        collector.add(frame(total=0.0))

        summary = collector.summary()

        self.assertEqual(summary.total_frames, 2)
        self.assertEqual(summary.warmup_frames, 2)
        self.assertEqual(summary.steady_frames, 0)
        self.assertEqual(summary.steady_average_fps, 0.0)
        self.assertTrue(all(stage.samples == 0 for stage in summary.stages.values()))

    def test_zero_total_steady_time_returns_zero_fps_without_division_error(self) -> None:
        collector = MetricsCollector()
        collector.add(frame(capture=0.0, inference=0.0, decision=0.0, render=0.0, total=0.0))

        summary = collector.summary()

        self.assertEqual(summary.steady_frames, 1)
        self.assertEqual(summary.steady_average_fps, 0.0)
        self.assertEqual(summary.stage("total").mean_ms, 0.0)

    def test_summary_is_a_snapshot_and_stage_mapping_is_read_only(self) -> None:
        collector = MetricsCollector()
        collector.add(frame())
        first = collector.summary()
        collector.add(frame())

        self.assertEqual(first.total_frames, 1)
        self.assertEqual(collector.summary().total_frames, 2)
        with self.assertRaises(TypeError):
            first.stages["capture"] = first.stage("capture")  # type: ignore[index]

    def test_rejects_invalid_warmup_count_and_unknown_stage(self) -> None:
        for invalid in (-1, 1.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "warmup_frames"):
                    MetricsCollector(invalid)  # type: ignore[arg-type]

        with self.assertRaisesRegex(KeyError, "unknown stage"):
            MetricsCollector().summary().stage("audio")


if __name__ == "__main__":
    unittest.main()
