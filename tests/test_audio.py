from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smart_cane.feedback.audio import AudioFeedback  # noqa: E402


class MutableClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class AudioFeedbackTests(unittest.TestCase):
    def test_disabled_feedback_does_not_start_a_process(self) -> None:
        runner = Mock()
        feedback = AudioFeedback(enabled=False, runner=runner)

        self.assertFalse(feedback.speak("obstacle ahead"))
        runner.assert_not_called()

    def test_speak_uses_argument_list_without_a_shell(self) -> None:
        runner = Mock(return_value=subprocess.CompletedProcess([], 0))
        feedback = AudioFeedback(
            enabled=True,
            voice="Samantha",
            rate=190,
            runner=runner,
        )

        self.assertTrue(feedback.speak("warning; $(touch /tmp/not-run)"))

        runner.assert_called_once_with(
            ["say", "-v", "Samantha", "-r", "190", "warning; $(touch /tmp/not-run)"],
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def test_cooldown_suppresses_then_allows_an_alert(self) -> None:
        clock = MutableClock(10.0)
        runner = Mock(return_value=subprocess.CompletedProcess([], 0))
        feedback = AudioFeedback(
            enabled=True,
            cooldown_seconds=2.5,
            clock=clock,
            runner=runner,
        )

        self.assertTrue(feedback.speak("first"))
        clock.now = 12.4
        self.assertFalse(feedback.speak("too soon"))
        clock.now = 12.5
        self.assertTrue(feedback.speak("allowed"))
        self.assertEqual(runner.call_count, 2)

    def test_force_bypasses_cooldown(self) -> None:
        clock = MutableClock(5.0)
        runner = Mock(return_value=subprocess.CompletedProcess([], 0))
        feedback = AudioFeedback(
            enabled=True,
            cooldown_seconds=60.0,
            clock=clock,
            runner=runner,
        )

        self.assertTrue(feedback.speak("first"))
        self.assertTrue(feedback.speak("urgent", force=True))
        self.assertEqual(runner.call_count, 2)

    def test_process_failure_does_not_consume_cooldown(self) -> None:
        clock = MutableClock(3.0)
        runner = Mock(
            side_effect=[
                OSError("say is unavailable"),
                subprocess.CompletedProcess([], 0),
            ]
        )
        feedback = AudioFeedback(
            enabled=True,
            cooldown_seconds=10.0,
            clock=clock,
            runner=runner,
        )

        self.assertFalse(feedback.speak("first attempt"))
        self.assertTrue(feedback.speak("second attempt"))
        self.assertEqual(runner.call_count, 2)

    def test_nonzero_exit_does_not_report_success(self) -> None:
        runner = Mock(return_value=subprocess.CompletedProcess([], 1))
        feedback = AudioFeedback(enabled=True, runner=runner)

        self.assertFalse(feedback.notify("warning"))
        self.assertIsNone(feedback.last_spoken_at)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            AudioFeedback(cooldown_seconds=-0.1)
        with self.assertRaises(ValueError):
            AudioFeedback(rate=0)
        with self.assertRaises(ValueError):
            AudioFeedback(voice="   ")


if __name__ == "__main__":
    unittest.main()
