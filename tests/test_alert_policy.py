from __future__ import annotations

import unittest

from smart_cane.decision.risk_engine import RiskAssessment
from smart_cane.feedback.alert_policy import AlertPolicy


class MutableClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def assessment(
    *,
    label: str = "person",
    direction: str = "center",
    level: str = "medium",
    should_alert: bool = True,
) -> RiskAssessment:
    return RiskAssessment(
        label=label,
        direction=direction,  # type: ignore[arg-type]
        level=level,  # type: ignore[arg-type]
        should_alert=should_alert,
        message=f"{level}: {label} {direction}",
        confidence=0.9,
        area_ratio=0.2,
    )


class AlertPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = MutableClock(10.0)
        self.policy = AlertPolicy(clock=self.clock)

    def test_repeated_signature_is_suppressed_during_cooldown(self) -> None:
        first = self.policy.evaluate("track-1", assessment())
        repeated = self.policy.evaluate("track-1", assessment())

        self.assertTrue(first.should_emit)
        self.assertEqual(first.reason, "first_alert")
        self.assertFalse(repeated.should_emit)
        self.assertEqual(repeated.reason, "cooldown_active")
        self.assertEqual(repeated.cooldown_remaining, 3.0)

    def test_uncommitted_candidate_remains_eligible_after_delivery_failure(self) -> None:
        first_attempt = self.policy.propose("track-1", assessment())
        retry = self.policy.propose("track-1", assessment())

        self.assertTrue(first_attempt.should_emit)
        self.assertTrue(retry.should_emit)
        self.assertEqual(retry.reason, "first_alert")

        self.assertTrue(self.policy.commit("track-1", retry))
        suppressed = self.policy.propose("track-1", assessment())
        self.assertFalse(suppressed.should_emit)
        self.assertEqual(suppressed.reason, "cooldown_active")

    def test_uncommitted_upgrade_keeps_its_force_reason_for_retry(self) -> None:
        self.policy.evaluate("track-1", assessment(level="medium"))
        first_attempt = self.policy.propose("track-1", assessment(level="high"))
        retry = self.policy.propose("track-1", assessment(level="high"))

        self.assertEqual(first_attempt.reason, "risk_upgraded")
        self.assertTrue(retry.should_emit)
        self.assertEqual(retry.reason, "risk_upgraded")

    def test_same_signature_emits_when_cooldown_has_elapsed(self) -> None:
        self.policy.evaluate(1, assessment())
        self.clock.now = 12.999
        self.assertFalse(self.policy.evaluate(1, assessment()).should_emit)

        self.clock.now = 13.0
        decision = self.policy.evaluate(1, assessment())

        self.assertTrue(decision.should_emit)
        self.assertEqual(decision.reason, "cooldown_elapsed")

    def test_medium_to_high_upgrade_bypasses_cooldown(self) -> None:
        self.policy.evaluate("track-1", assessment(level="medium"))
        self.clock.now = 10.1

        decision = self.policy.evaluate("track-1", assessment(level="high"))

        self.assertTrue(decision.should_emit)
        self.assertEqual(decision.reason, "risk_upgraded")

    def test_direction_change_bypasses_cooldown(self) -> None:
        self.policy.evaluate("track-1", assessment(direction="left"))
        self.clock.now = 10.1

        decision = self.policy.evaluate("track-1", assessment(direction="right"))

        self.assertTrue(decision.should_emit)
        self.assertEqual(decision.reason, "direction_changed")

    def test_disappearing_track_is_forgotten_and_can_alert_on_reappearance(self) -> None:
        self.policy.evaluate("track-1", assessment())
        self.assertEqual(self.policy.tracked_count, 1)

        removed = self.policy.retain_active([])
        reappeared = self.policy.evaluate("track-1", assessment())

        self.assertEqual(removed, 1)
        self.assertTrue(reappeared.should_emit)
        self.assertEqual(reappeared.reason, "first_alert")

    def test_retain_active_preserves_tracks_still_visible(self) -> None:
        self.policy.evaluate("keep", assessment())
        self.policy.evaluate("remove", assessment())

        removed = self.policy.retain_active(["keep"])
        repeated = self.policy.evaluate("keep", assessment())

        self.assertEqual(removed, 1)
        self.assertEqual(self.policy.tracked_count, 1)
        self.assertFalse(repeated.should_emit)

    def test_assessment_that_declines_alert_is_never_emitted(self) -> None:
        decision = self.policy.evaluate(
            "track-1",
            assessment(level="low", should_alert=False),
        )

        self.assertFalse(decision.should_emit)
        self.assertEqual(decision.reason, "not_alertable")
        self.assertEqual(decision.message, "low: person center")

    def test_label_is_part_of_the_cooldown_signature(self) -> None:
        self.policy.evaluate("track-1", assessment(label="person"))

        decision = self.policy.evaluate("track-1", assessment(label="chair"))

        self.assertTrue(decision.should_emit)
        self.assertEqual(decision.reason, "signature_changed")

    def test_configuration_and_inputs_are_validated(self) -> None:
        for invalid in (-1, float("inf"), float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                AlertPolicy(invalid)
        with self.assertRaises(TypeError):
            AlertPolicy(True)
        with self.assertRaises(TypeError):
            AlertPolicy(clock=None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.policy.evaluate([], assessment())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            self.policy.retain_active([["not-hashable"]])


if __name__ == "__main__":
    unittest.main()
