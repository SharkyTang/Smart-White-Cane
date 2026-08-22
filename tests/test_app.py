from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from smart_cane.app import RunSummary, evaluate_detections, run_demo, run_live
from smart_cane.cli import main
from smart_cane.config import RuntimeConfig
from smart_cane.decision.risk_engine import RiskEngine
from smart_cane.decision.temporal_filter import TemporalFilter, TemporalFilterConfig
from smart_cane.feedback.alert_policy import AlertPolicy
from smart_cane.metrics import MetricsCollector
from smart_cane.models import Detection
from smart_cane.perception.camera import CameraReadError


class FakeLiveSource:
    def __init__(self, frame_count: int) -> None:
        self.frames = [object() for _ in range(frame_count)]
        self.released = False

    def read(self) -> object:
        if not self.frames:
            raise CameraReadError("end")
        return self.frames.pop(0)

    def release(self) -> None:
        self.released = True

    def __enter__(self) -> FakeLiveSource:
        return self

    def __exit__(self, *args: object) -> bool:
        self.release()
        return False


class FakeLiveDetector:
    device = "cpu"

    @staticmethod
    def _detections() -> list[Detection]:
        return [Detection("chair", 0.9, (20, 10, 80, 70), 100, 100)]

    def warmup(self, frame: object) -> list[Detection]:
        return self._detections()

    def __call__(self, frame: object) -> list[Detection]:
        return self._detections()


class RejectingFeedback:
    def __init__(self) -> None:
        self.calls = 0

    def notify(self, message: str, *, force: bool = False) -> bool:
        self.calls += 1
        return False


class MutableClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class AppTests(unittest.TestCase):
    def test_demo_runs_without_external_services(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assessments = run_demo(RuntimeConfig(display=False))

        self.assertEqual(len(assessments), 3)
        self.assertIn("Most urgent: Danger: chair ahead.", output.getvalue())

    def test_custom_high_risk_area_changes_confirmation_and_risk_rule_together(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            assessments = run_demo(RuntimeConfig(display=False, high_risk_area_ratio=0.4))

        chair = next(item for item in assessments if item.label == "chair")
        self.assertEqual(chair.level, "medium")
        self.assertIn("Caution: chair ahead.", output.getvalue())

    def test_cli_demo_returns_success(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(["--demo", "--no-display"])

        self.assertEqual(result, 0)
        self.assertIn("deterministic demo", output.getvalue())

    def test_run_summary_calculates_fps(self) -> None:
        summary = RunSummary(frames=30, detections=5, alerts=1, elapsed_seconds=2.0)
        self.assertEqual(summary.average_fps, 15.0)

    def test_live_loop_integrates_warmup_stability_alerts_and_metrics(self) -> None:
        source = FakeLiveSource(frame_count=3)
        detector = FakeLiveDetector()
        config = RuntimeConfig(
            display=False,
            max_frames=3,
            warmup_frames=1,
            confirmation_window=1,
            minimum_confirmations=1,
        )

        with (
            mock.patch("smart_cane.app._load_cv2", return_value=object()),
            mock.patch("smart_cane.app.CameraSource", return_value=source),
            mock.patch("smart_cane.app.YoloDetector", return_value=detector),
        ):
            summary = run_live(config)

        self.assertTrue(source.released)
        self.assertEqual(summary.frames, 3)
        self.assertEqual(summary.detections, 3)
        self.assertEqual(summary.stable_detections, 3)
        self.assertEqual(summary.alerts, 1)
        self.assertEqual(summary.spoken_alerts, 0)
        self.assertEqual(summary.device, "cpu")
        self.assertIsNotNone(summary.metrics)
        assert summary.metrics is not None
        self.assertEqual(summary.metrics.total_frames, 3)
        self.assertEqual(summary.metrics.warmup_frames, 1)
        self.assertEqual(summary.metrics.steady_frames, 2)
        self.assertEqual(summary.metrics.stage("inference").samples, 2)

    def test_cooling_urgent_target_does_not_starve_second_target(self) -> None:
        clock = MutableClock()
        temporal_filter = TemporalFilter(
            TemporalFilterConfig(
                confirmation_window=1,
                minimum_confirmations=1,
            )
        )
        alert_policy = AlertPolicy(cooldown_seconds=3.0, clock=clock)
        detections = (
            Detection("chair", 0.9, (20, 10, 80, 70), 100, 100),
            Detection("person", 0.9, (0, 10, 20, 70), 100, 100),
        )

        first = evaluate_detections(
            detections,
            temporal_filter=temporal_filter,
            risk_engine=RiskEngine(),
            alert_policy=alert_policy,
        )
        second = evaluate_detections(
            detections,
            temporal_filter=temporal_filter,
            risk_engine=RiskEngine(),
            alert_policy=alert_policy,
        )

        self.assertIsNotNone(first.alert)
        self.assertIsNotNone(second.alert)
        assert first.alert is not None and second.alert is not None
        self.assertTrue(first.alert.should_emit)
        self.assertEqual(first.alert.assessment.label, "chair")
        self.assertTrue(second.alert.should_emit)
        self.assertEqual(second.alert.assessment.label, "person")

    def test_live_loop_propagates_first_frame_failure(self) -> None:
        source = FakeLiveSource(frame_count=0)

        with (
            mock.patch("smart_cane.app._load_cv2", return_value=object()),
            mock.patch("smart_cane.app.CameraSource", return_value=source),
            mock.patch("smart_cane.app.YoloDetector", return_value=FakeLiveDetector()),
            self.assertRaises(CameraReadError),
        ):
            run_live(RuntimeConfig(display=False))

        self.assertTrue(source.released)

    def test_failed_audio_delivery_does_not_consume_alert_cooldown(self) -> None:
        source = FakeLiveSource(frame_count=3)
        feedback = RejectingFeedback()
        config = RuntimeConfig(
            audio_enabled=True,
            display=False,
            max_frames=3,
            warmup_frames=0,
            confirmation_window=1,
            minimum_confirmations=1,
        )

        with (
            mock.patch("smart_cane.app._load_cv2", return_value=object()),
            mock.patch("smart_cane.app.CameraSource", return_value=source),
            mock.patch("smart_cane.app.YoloDetector", return_value=FakeLiveDetector()),
            mock.patch("smart_cane.app.AudioFeedback", return_value=feedback),
        ):
            summary = run_live(config)

        self.assertEqual(feedback.calls, 3)
        self.assertEqual(summary.alerts, 0)
        self.assertEqual(summary.spoken_alerts, 0)

    def test_cli_labels_an_all_warmup_run_as_having_no_steady_samples(self) -> None:
        collector = MetricsCollector(warmup_frames=1)
        collector.add(
            capture_seconds=0.01,
            inference_seconds=0.02,
            decision_seconds=0.001,
            render_seconds=0.001,
            total_seconds=0.032,
        )
        summary = RunSummary(
            frames=1,
            detections=1,
            alerts=0,
            elapsed_seconds=0.5,
            cold_inference_seconds=0.02,
            device="cpu",
            metrics=collector.summary(),
        )
        output = io.StringIO()

        with (
            mock.patch("smart_cane.cli.run_live", return_value=summary),
            contextlib.redirect_stdout(output),
        ):
            result = main(["--no-display", "--max-frames", "1"])

        self.assertEqual(result, 0)
        self.assertIn("no steady-state samples", output.getvalue())

    def test_cli_passes_milestone_02_tuning_options_to_runtime_config(self) -> None:
        summary = RunSummary(
            frames=0,
            detections=0,
            alerts=0,
            elapsed_seconds=0.0,
        )
        output = io.StringIO()

        with (
            mock.patch("smart_cane.cli.run_live", return_value=summary) as runner,
            contextlib.redirect_stdout(output),
        ):
            result = main(
                [
                    "--no-display",
                    "--device",
                    "0",
                    "--image-size",
                    "320",
                    "--warmup-frames",
                    "2",
                    "--confirmation-window",
                    "7",
                    "--minimum-confirmations",
                    "4",
                    "--high-risk-confirmations",
                    "3",
                    "--high-risk-area-ratio",
                    "0.25",
                    "--max-missing-frames",
                    "4",
                    "--direction-hysteresis",
                    "0.04",
                    "--audio-cooldown",
                    "1.5",
                ]
            )

        self.assertEqual(result, 0)
        config = runner.call_args.args[0]
        self.assertEqual(config.device, "0")
        self.assertEqual(config.image_size, 320)
        self.assertEqual(config.warmup_frames, 2)
        self.assertEqual((config.minimum_confirmations, config.confirmation_window), (4, 7))
        self.assertEqual(config.high_risk_consecutive_frames, 3)
        self.assertEqual(config.high_risk_area_ratio, 0.25)
        self.assertEqual(config.max_missing_frames, 4)
        self.assertEqual(config.direction_hysteresis, 0.04)
        self.assertEqual(config.alert_cooldown_seconds, 1.5)


if __name__ == "__main__":
    unittest.main()
