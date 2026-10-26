from __future__ import annotations

import unittest

from smart_cane.decision.risk_engine import DistanceRiskRules, DistanceRiskState, RiskEngine
from smart_cane.decision.temporal_filter import StableDetection
from smart_cane.depth import DistanceEstimate, FusedDetection
from smart_cane.models import Detection


def fused(
    distance_m: float | None,
    *,
    track_key: str = "person:1",
    bbox: tuple[float, float, float, float] = (40, 0, 60, 20),
    confidence: float = 0.9,
    stale: bool = False,
    direction: str = "center",
) -> FusedDetection:
    stable = StableDetection(
        detection=Detection("person", confidence, bbox, 100, 100),
        track_key=track_key,
        direction=direction,  # type: ignore[arg-type]
        confirmed_frames=3,
    )
    valid = distance_m is not None and not stale
    estimate = DistanceEstimate(
        distance_m=distance_m,
        percentile_20_m=distance_m if valid else None,
        candidate_median_m=distance_m if valid else None,
        sample_count=20 if valid else 0,
        roi_pixel_count=40 if valid else 0,
        coverage_ratio=0.5 if valid else 0.0,
        is_valid=valid,
        is_stale=stale,
        missed_frames=1 if stale else 0,
    )
    return FusedDetection(stable, estimate)


class DistanceRiskStateTests(unittest.TestCase):
    def test_requires_two_frames_to_enter_and_three_to_exit(self) -> None:
        state = DistanceRiskState()

        self.assertEqual(state.update("track", 0.8), "low")
        self.assertEqual(state.update("track", 0.8), "high")
        self.assertEqual(state.update("track", 1.20), "high")
        self.assertEqual(state.update("track", 1.20), "high")
        self.assertEqual(state.update("track", 1.20), "medium")

    def test_applies_point_fifteen_meter_exit_hysteresis(self) -> None:
        state = DistanceRiskState()
        state.update("track", 1.5)
        self.assertEqual(state.update("track", 1.5), "medium")

        self.assertEqual(state.update("track", 2.10), "medium")
        self.assertEqual(state.update("track", 2.16), "medium")
        self.assertEqual(state.update("track", 2.16), "medium")
        self.assertEqual(state.update("track", 2.16), "low")

    def test_missing_observation_breaks_a_pending_transition(self) -> None:
        state = DistanceRiskState()
        self.assertEqual(state.update("track", 0.8), "low")
        state.observe_missing("track")

        self.assertEqual(state.update("track", 0.8), "low")
        self.assertEqual(state.update("track", 0.8), "high")

    def test_rules_and_track_lifecycle_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            DistanceRiskRules(high_distance_m=2.0, medium_distance_m=1.0)
        state = DistanceRiskState()
        state.update("keep", 3.0)
        state.update("remove", 3.0)

        self.assertEqual(state.retain_active(["keep"]), 1)
        self.assertEqual(state.tracked_count, 1)
        state.reset()
        self.assertEqual(state.tracked_count, 0)


class FusedRiskEngineTests(unittest.TestCase):
    def test_valid_lidar_distance_takes_priority_over_box_area(self) -> None:
        engine = RiskEngine()
        large_far = fused(3.0, bbox=(0, 0, 100, 100))

        result = engine.assess(large_far)

        self.assertEqual(result.level, "low")
        self.assertEqual(result.distance_source, "lidar")
        self.assertEqual(result.distance_m, 3.0)
        self.assertTrue(result.distance_valid)
        self.assertEqual(result.distance_status, "valid")
        self.assertIn("3.0 meters", result.message)

    def test_metric_threshold_uses_two_frame_entry_and_numeric_message(self) -> None:
        engine = RiskEngine()
        near = fused(0.8)

        first = engine.assess(near)
        second = engine.assess(near)

        self.assertEqual(first.level, "low")
        self.assertFalse(first.should_alert)
        self.assertEqual(second.level, "high")
        self.assertTrue(second.should_alert)
        self.assertEqual(second.message, "Danger: person ahead, 0.8 meters.")

    def test_invalid_or_stale_distance_falls_back_to_area_without_number(self) -> None:
        engine = RiskEngine()
        invalid = fused(None, bbox=(0, 0, 100, 100))
        stale = fused(0.7, bbox=(0, 0, 100, 100), stale=True)

        invalid_result = engine.assess(invalid)
        stale_result = engine.assess(stale)

        self.assertEqual(invalid_result.level, "high")
        self.assertEqual(stale_result.level, "high")
        self.assertEqual(invalid_result.distance_source, "area_fallback")
        self.assertEqual(invalid_result.distance_status, "unavailable")
        self.assertEqual(stale_result.distance_status, "stale")
        self.assertIsNone(stale_result.distance_m)
        self.assertNotIn("meter", invalid_result.message)
        self.assertNotIn("meter", stale_result.message)

    def test_yolo_confidence_still_suppresses_metric_alert(self) -> None:
        engine = RiskEngine()
        near = fused(0.5, confidence=0.2)

        result = engine.assess(near)

        self.assertEqual(result.level, "low")
        self.assertFalse(result.should_alert)
        self.assertEqual(result.distance_source, "lidar")

    def test_low_confidence_breaks_a_pending_metric_transition(self) -> None:
        engine = RiskEngine()
        self.assertEqual(engine.assess(fused(0.5)).level, "low")
        self.assertEqual(engine.assess(fused(0.5, confidence=0.2)).level, "low")

        self.assertEqual(engine.assess(fused(0.5)).level, "low")
        self.assertEqual(engine.assess(fused(0.5)).level, "high")

    def test_engine_exposes_metric_track_cleanup_and_reset(self) -> None:
        engine = RiskEngine()
        engine.assess(fused(3.0, track_key="keep"))
        engine.assess(fused(3.0, track_key="remove"))

        self.assertEqual(engine.retain_active(["keep"]), 1)
        self.assertEqual(engine.distance_state.tracked_count, 1)
        engine.reset()
        self.assertEqual(engine.distance_state.tracked_count, 0)

    def test_center_target_wins_before_nearer_side_target_at_same_risk(self) -> None:
        engine = RiskEngine(distance_rules=DistanceRiskRules(enter_frames=1))
        farther = fused(1.8, track_key="person:1")
        nearer = fused(
            1.2,
            track_key="person:2",
            bbox=(0, 0, 20, 20),
            direction="left",
        )

        result = engine.most_urgent([farther, nearer])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.distance_m, 1.8)

    def test_nearest_metric_target_wins_within_the_same_direction_and_risk(self) -> None:
        engine = RiskEngine(distance_rules=DistanceRiskRules(enter_frames=1))
        farther = fused(1.8, track_key="person:1")
        nearer = fused(1.2, track_key="person:2")

        result = engine.most_urgent([farther, nearer])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.distance_m, 1.2)


if __name__ == "__main__":
    unittest.main()
