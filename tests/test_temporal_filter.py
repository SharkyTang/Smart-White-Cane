from __future__ import annotations

import unittest

from smart_cane.decision.temporal_filter import TemporalFilter, TemporalFilterConfig
from smart_cane.models import Detection


def detection(
    *,
    label: str = "person",
    bbox: tuple[float, float, float, float] = (20, 20, 40, 40),
    confidence: float = 0.9,
) -> Detection:
    return Detection(
        label=label,
        confidence=confidence,
        bbox=bbox,
        frame_width=100,
        frame_height=100,
    )


class TemporalFilterTests(unittest.TestCase):
    def test_ordinary_target_is_returned_after_three_of_five_frames(self) -> None:
        temporal = TemporalFilter()

        self.assertEqual(temporal.update([detection()]), ())
        self.assertEqual(temporal.update([]), ())
        self.assertEqual(temporal.update([detection(bbox=(21, 20, 41, 40))]), ())
        self.assertEqual(temporal.update([]), ())
        stable = temporal.update([detection(bbox=(22, 20, 42, 40))])

        self.assertEqual(len(stable), 1)
        self.assertEqual(stable[0].confirmed_frames, 3)
        self.assertEqual(stable[0].track_key, "person:1")

    def test_one_frame_false_positive_never_becomes_stable(self) -> None:
        temporal = TemporalFilter()

        self.assertEqual(temporal.update([detection(label="chair")]), ())
        self.assertEqual(temporal.update([]), ())
        self.assertEqual(temporal.update([]), ())
        self.assertEqual(temporal.update([]), ())
        self.assertEqual(temporal.track_count, 0)

    def test_high_area_target_fast_confirms_after_two_consecutive_frames(self) -> None:
        temporal = TemporalFilter()
        large = detection(bbox=(20, 20, 70, 70))

        self.assertEqual(temporal.update([large]), ())
        stable = temporal.update([detection(bbox=(21, 20, 71, 70))])

        self.assertEqual(len(stable), 1)
        self.assertEqual(stable[0].confirmed_frames, 2)
        self.assertGreaterEqual(large.area_ratio, 0.18)

    def test_high_area_fast_path_resets_after_a_missing_frame(self) -> None:
        temporal = TemporalFilter()
        large = detection(bbox=(20, 20, 70, 70))

        self.assertEqual(temporal.update([large]), ())
        self.assertEqual(temporal.update([]), ())
        self.assertEqual(temporal.update([large]), ())
        stable = temporal.update([large])

        self.assertEqual(len(stable), 1)
        self.assertEqual(stable[0].confirmed_frames, 3)

    def test_brief_dropout_keeps_confirmed_target_then_resumes_same_track(self) -> None:
        temporal = TemporalFilter()
        for _ in range(3):
            stable = temporal.update([detection()])

        track_key = stable[0].track_key
        first_miss = temporal.update([])
        second_miss = temporal.update([])
        resumed = temporal.update([detection(bbox=(21, 20, 41, 40))])

        self.assertEqual(first_miss[0].missed_frames, 1)
        self.assertEqual(second_miss[0].missed_frames, 2)
        self.assertEqual(resumed[0].track_key, track_key)
        self.assertEqual(resumed[0].missed_frames, 0)
        self.assertEqual(resumed[0].confirmed_frames, 4)

    def test_confirmed_target_is_cleared_on_third_consecutive_miss(self) -> None:
        temporal = TemporalFilter()
        for _ in range(3):
            temporal.update([detection()])

        self.assertEqual(len(temporal.update([])), 1)
        self.assertEqual(len(temporal.update([])), 1)
        self.assertEqual(temporal.update([]), ())
        self.assertEqual(temporal.track_count, 0)

    def test_bbox_uses_configurable_exponential_smoothing(self) -> None:
        temporal = TemporalFilter(
            TemporalFilterConfig(
                minimum_confirmations=2,
                bbox_smoothing=0.25,
                high_risk_consecutive_frames=3,
            )
        )
        temporal.update([detection(bbox=(20, 20, 40, 40))])
        stable = temporal.update([detection(bbox=(24, 24, 44, 44))])

        self.assertEqual(stable[0].detection.bbox, (21.0, 21.0, 41.0, 41.0))

    def test_direction_hysteresis_prevents_boundary_flapping(self) -> None:
        temporal = TemporalFilter(
            TemporalFilterConfig(
                minimum_confirmations=2,
                bbox_smoothing=1.0,
                high_risk_consecutive_frames=3,
            )
        )

        temporal.update([detection(bbox=(23, 20, 43, 40))])  # centre x = 0.33 -> left
        stable = temporal.update([detection(bbox=(25, 20, 45, 40))])  # x = 0.35
        self.assertEqual(stable[0].direction, "left")

        stable = temporal.update([detection(bbox=(28, 20, 48, 40))])  # x = 0.38
        self.assertEqual(stable[0].direction, "left")

        stable = temporal.update([detection(bbox=(31, 20, 51, 40))])  # x = 0.41
        self.assertEqual(stable[0].direction, "center")

        stable = temporal.update([detection(bbox=(40, 20, 60, 40))])  # x = 0.50
        self.assertEqual(stable[0].direction, "center")

        stable = temporal.update([detection(bbox=(49, 20, 69, 40))])  # x = 0.59
        self.assertEqual(stable[0].direction, "center")

        stable = temporal.update([detection(bbox=(59, 20, 79, 40))])  # x = 0.69
        self.assertEqual(stable[0].direction, "center")

        stable = temporal.update([detection(bbox=(61, 20, 81, 40))])  # x = 0.71
        self.assertEqual(stable[0].direction, "right")

        stable = temporal.update([detection(bbox=(57, 20, 77, 40))])  # x = 0.67
        self.assertEqual(stable[0].direction, "right")

        stable = temporal.update([detection(bbox=(49, 20, 69, 40))])  # x = 0.59
        self.assertEqual(stable[0].direction, "center")

    def test_tracks_multiple_same_label_targets_by_iou(self) -> None:
        temporal = TemporalFilter(
            TemporalFilterConfig(
                minimum_confirmations=2,
                high_risk_consecutive_frames=3,
                bbox_smoothing=1.0,
            )
        )
        temporal.update(
            [
                detection(label="person", bbox=(5, 10, 25, 30)),
                detection(label="person", bbox=(75, 10, 95, 30)),
            ]
        )
        stable = temporal.update(
            [
                detection(label="person", bbox=(73, 10, 93, 30)),
                detection(label="person", bbox=(7, 10, 27, 30)),
            ]
        )

        self.assertEqual(len(stable), 2)
        self.assertEqual(
            [(item.track_key, item.direction) for item in stable],
            [("person:1", "left"), ("person:2", "right")],
        )

    def test_different_labels_do_not_share_tracks(self) -> None:
        temporal = TemporalFilter(
            TemporalFilterConfig(
                minimum_confirmations=2,
                high_risk_consecutive_frames=3,
            )
        )
        temporal.update([detection(label="person")])
        temporal.update([detection(label="chair")])

        self.assertEqual(temporal.track_count, 2)

    def test_configuration_and_inputs_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            TemporalFilterConfig(confirmation_window=2, minimum_confirmations=3)
        with self.assertRaisesRegex(ValueError, "direction_hysteresis"):
            TemporalFilterConfig(direction_hysteresis=0.20)
        with self.assertRaisesRegex(ValueError, "bbox_smoothing"):
            TemporalFilterConfig(bbox_smoothing=0.0)
        with self.assertRaisesRegex(ValueError, "iou_threshold"):
            TemporalFilterConfig(iou_threshold=float("nan"))

        temporal = TemporalFilter()
        with self.assertRaisesRegex(TypeError, "Detection"):
            temporal.update([object()])  # type: ignore[list-item]
        with self.assertRaisesRegex(ValueError, "frame dimensions"):
            temporal.update(
                [
                    detection(),
                    Detection("chair", 0.9, (20, 20, 40, 40), 200, 100),
                ]
            )


if __name__ == "__main__":
    unittest.main()
