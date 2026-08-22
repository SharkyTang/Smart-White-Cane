from __future__ import annotations

import unittest

from smart_cane.decision.risk_engine import AreaThresholds, RiskEngine, RiskRules
from smart_cane.models import Detection


def detection(
    *,
    label: str = "person",
    confidence: float = 0.9,
    bbox: tuple[float, float, float, float],
) -> Detection:
    return Detection(
        label=label,
        confidence=confidence,
        bbox=bbox,
        frame_width=100,
        frame_height=100,
    )


class DetectionTests(unittest.TestCase):
    def test_calculates_center_and_visible_area_ratio(self) -> None:
        result = Detection(
            label="person",
            confidence=0.8,
            bbox=(100, 50, 300, 250),
            frame_width=400,
            frame_height=400,
        )

        self.assertEqual(result.center, (200.0, 150.0))
        self.assertEqual(result.normalized_center, (0.5, 0.375))
        self.assertAlmostEqual(result.area_ratio, 0.25)

    def test_clips_out_of_frame_box_when_calculating_area(self) -> None:
        result = detection(bbox=(-20, -10, 60, 50))

        self.assertAlmostEqual(result.area_ratio, 0.30)

    def test_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "confidence"):
            detection(confidence=1.1, bbox=(0, 0, 10, 10))
        with self.assertRaisesRegex(ValueError, "maximum"):
            detection(bbox=(20, 0, 10, 10))
        with self.assertRaisesRegex(ValueError, "frame_width"):
            Detection("person", 0.8, (0, 0, 1, 1), 0, 100)


class RiskEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RiskEngine(
            RiskRules(
                minimum_confidence=0.5,
                left_boundary=0.3,
                right_boundary=0.7,
                default_area_thresholds=AreaThresholds(medium=0.10, high=0.30),
            )
        )

    def test_classifies_left_center_and_right(self) -> None:
        self.assertEqual(self.engine.assess(detection(bbox=(0, 0, 20, 20))).direction, "left")
        self.assertEqual(
            self.engine.assess(detection(bbox=(40, 0, 60, 20))).direction,
            "center",
        )
        self.assertEqual(
            self.engine.assess(detection(bbox=(80, 0, 100, 20))).direction,
            "right",
        )

    def test_temporal_direction_can_override_raw_box_direction(self) -> None:
        result = self.engine.assess(
            detection(bbox=(28, 0, 32, 20)),
            direction_override="center",
        )

        self.assertEqual(result.direction, "center")
        self.assertIn("ahead", result.message)
        with self.assertRaisesRegex(ValueError, "direction_override"):
            self.engine.assess(
                detection(bbox=(28, 0, 32, 20)),
                direction_override="up",  # type: ignore[arg-type]
            )

    def test_area_thresholds_produce_all_risk_levels(self) -> None:
        low = self.engine.assess(detection(bbox=(40, 0, 60, 20)))
        medium = self.engine.assess(detection(bbox=(30, 0, 70, 40)))
        high = self.engine.assess(detection(bbox=(20, 0, 80, 60)))

        self.assertEqual((low.level, medium.level, high.level), ("low", "medium", "high"))
        self.assertEqual(
            (low.should_alert, medium.should_alert, high.should_alert),
            (False, True, True),
        )
        self.assertEqual(low.message, "Person detected ahead.")
        self.assertEqual(medium.message, "Caution: person ahead.")
        self.assertEqual(high.message, "Danger: person ahead.")

    def test_low_confidence_suppresses_an_otherwise_high_risk_box(self) -> None:
        result = self.engine.assess(detection(confidence=0.49, bbox=(0, 0, 100, 100)))

        self.assertEqual(result.level, "low")
        self.assertFalse(result.should_alert)

    def test_label_specific_thresholds_are_case_insensitive(self) -> None:
        engine = RiskEngine(
            RiskRules(
                default_area_thresholds=AreaThresholds(medium=0.20, high=0.40),
                per_label_area_thresholds={"CAR": AreaThresholds(medium=0.05, high=0.10)},
            )
        )

        result = engine.assess(detection(label="car", bbox=(30, 0, 70, 40)))

        self.assertEqual(result.level, "high")

    def test_most_urgent_prefers_risk_then_center_position(self) -> None:
        result = self.engine.most_urgent(
            [
                detection(label="chair", bbox=(0, 0, 50, 60)),
                detection(label="person", bbox=(20, 0, 80, 60)),
            ]
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.label, "person")
        self.assertEqual(result.direction, "center")
        self.assertIsNone(self.engine.most_urgent([]))

    def test_rules_validate_inconsistent_boundaries_and_thresholds(self) -> None:
        with self.assertRaisesRegex(ValueError, "direction boundaries"):
            RiskRules(left_boundary=0.8, right_boundary=0.2)
        with self.assertRaisesRegex(ValueError, "area thresholds"):
            AreaThresholds(medium=0.5, high=0.4)


if __name__ == "__main__":
    unittest.main()
