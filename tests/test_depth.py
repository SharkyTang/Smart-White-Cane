from __future__ import annotations

import math
import unittest

from smart_cane.decision.temporal_filter import StableDetection
from smart_cane.depth import DepthFrame, DistanceFusion
from smart_cane.models import Detection


def stable(
    *,
    track_key: str = "person:1",
    bbox: tuple[float, float, float, float] = (0, 0, 10, 10),
    missed_frames: int = 0,
    frame_width: int = 10,
    frame_height: int = 10,
) -> StableDetection:
    return StableDetection(
        detection=Detection("person", 0.9, bbox, frame_width, frame_height),
        track_key=track_key,
        direction="center",
        confirmed_frames=3,
        missed_frames=missed_frames,
    )


def frame(
    value: float,
    *,
    confidence: int = 2,
    frame_id: str = "frame-1",
    depth_source: str = "raw_scene_depth",
) -> DepthFrame:
    return DepthFrame(
        distances_m=[[value] * 10 for _ in range(10)],
        confidence=[[confidence] * 10 for _ in range(10)],
        frame_width=10,
        frame_height=10,
        frame_id=frame_id,
        depth_source=depth_source,  # type: ignore[arg-type]
    )


class DepthFrameTests(unittest.TestCase):
    def test_normalizes_rectangular_grids_and_exposes_depth_dimensions(self) -> None:
        depth = DepthFrame(
            distances_m=[[1, 2], [3, math.nan]],
            confidence=[[0, 1], [2, 2]],
            frame_width=20,
            frame_height=10,
            frame_id=7,
            captured_at=1.25,
        )

        self.assertEqual((depth.depth_width, depth.depth_height), (2, 2))
        self.assertEqual(depth.distances_m[0], (1.0, 2.0))
        self.assertTrue(math.isnan(depth.distances_m[1][1]))
        self.assertEqual(depth.depth_source, "raw_scene_depth")

    def test_validates_depth_source_without_comparing_capture_timestamps(self) -> None:
        raw = frame(1.0, frame_id="newer")
        smoothed = DepthFrame(
            [[1.0]],
            [[2]],
            10,
            10,
            frame_id="older",
            captured_at=0.1,
            depth_source="smoothed_scene_depth",
        )

        self.assertEqual(raw.depth_source, "raw_scene_depth")
        self.assertEqual(smoothed.depth_source, "smoothed_scene_depth")
        with self.assertRaisesRegex(ValueError, "depth_source"):
            DepthFrame([[1.0]], [[2]], 10, 10, depth_source="estimated")  # type: ignore[arg-type]

    def test_rejects_misaligned_or_invalid_grids(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            DepthFrame([[1.0]], [[2, 2]], 10, 10)
        with self.assertRaisesRegex(ValueError, "rectangular"):
            DepthFrame([[1.0], [2.0, 3.0]], [[2], [2]], 10, 10)
        with self.assertRaisesRegex(ValueError, "low.*medium.*high"):
            DepthFrame([[1.0]], [[3]], 10, 10)


class DistanceFusionTests(unittest.TestCase):
    def test_insets_box_and_uses_p20_while_exposing_candidate_median(self) -> None:
        values = [[0.25] * 10 for _ in range(10)]
        confidence = [[2] * 10 for _ in range(10)]
        # The 15% inset maps the full 10x10 box to rows/columns 1 through 8.
        interior_positions = [(row, column) for row in range(1, 9) for column in range(1, 9)]
        for row, column in interior_positions:
            values[row][column] = 3.0
        for row, column in interior_positions[:13]:
            values[row][column] = 0.8
        depth = DepthFrame(values, confidence, 10, 10, frame_id="rgb-1")

        result = DistanceFusion().update([stable()], depth)[0].distance

        self.assertTrue(result.usable)
        self.assertEqual(result.sample_count, 64)
        self.assertEqual(result.roi_pixel_count, 64)
        self.assertEqual(result.percentile_20_m, 0.8)
        self.assertEqual(result.candidate_median_m, 3.0)
        self.assertEqual(result.candidate_center_median_m, 3.0)
        self.assertEqual(result.distance_m, 0.8)
        self.assertEqual(result.frame_id, "rgb-1")
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.candidate_status, "valid")
        self.assertEqual(result.depth_source, "raw_scene_depth")
        self.assertEqual(result.quality, "good")
        self.assertEqual(result.freshness_frames, 0)

    def test_requires_medium_confidence_range_sample_count_and_coverage(self) -> None:
        values = [[math.nan] * 10 for _ in range(10)]
        confidence = [[0] * 10 for _ in range(10)]
        positions = [(row, column) for row in range(1, 9) for column in range(1, 9)]
        for row, column in positions[:11]:
            values[row][column] = 1.0
            confidence[row][column] = 2
        values[8][8] = 0.0
        values[8][7] = 5.1
        confidence[8][8] = confidence[8][7] = 2
        fusion = DistanceFusion()

        insufficient = fusion.update([stable()], DepthFrame(values, confidence, 10, 10))[0]
        values[7][8] = 1.2
        confidence[7][8] = 1
        low_coverage = fusion.update([stable()], DepthFrame(values, confidence, 10, 10))[0]
        values[7][7] = 1.3
        confidence[7][7] = 1
        accepted = fusion.update([stable()], DepthFrame(values, confidence, 10, 10))[0]

        self.assertFalse(insufficient.distance.is_valid)
        self.assertEqual(insufficient.distance.sample_count, 11)
        self.assertEqual(insufficient.distance.status, "insufficient_samples")
        self.assertEqual(insufficient.distance.candidate_status, "insufficient_samples")
        self.assertEqual(insufficient.distance.quality, "degraded")
        self.assertEqual(insufficient.distance.freshness_frames, 0)
        self.assertFalse(low_coverage.distance.is_valid)
        self.assertEqual(low_coverage.distance.sample_count, 12)
        self.assertEqual(low_coverage.distance.status, "low_coverage")
        self.assertEqual(low_coverage.distance.candidate_status, "low_coverage")
        self.assertEqual(low_coverage.distance.freshness_frames, 0)
        self.assertTrue(accepted.distance.is_valid)
        self.assertEqual(accepted.distance.sample_count, 13)
        self.assertGreaterEqual(accepted.distance.coverage_ratio, 0.20)

    def test_track_distance_is_median_of_latest_three_candidates(self) -> None:
        fusion = DistanceFusion()

        results = [fusion.update([stable()], frame(value))[0].distance for value in (3, 1, 2, 4)]

        self.assertEqual([item.distance_m for item in results], [3.0, 2.0, 2.0, 2.0])
        self.assertEqual(results[-1].percentile_20_m, 4.0)

    def test_invalid_or_missed_depth_is_stale_twice_then_cleared(self) -> None:
        fusion = DistanceFusion(max_stale_frames=2)
        fusion.update([stable()], frame(1.4))
        invalid = frame(1.4, confidence=0)

        first = fusion.update([stable()], invalid)[0].distance
        second = fusion.update([stable(missed_frames=1)], invalid)[0].distance
        third = fusion.update([stable(missed_frames=2)], invalid)[0].distance

        self.assertEqual((first.distance_m, first.is_stale, first.missed_frames), (1.4, True, 1))
        self.assertEqual((second.distance_m, second.is_stale, second.missed_frames), (1.4, True, 2))
        self.assertEqual((third.distance_m, third.is_stale, third.missed_frames), (None, False, 3))
        self.assertEqual(first.status, "stale")
        self.assertEqual(first.candidate_status, "insufficient_samples")
        self.assertEqual(first.quality, "degraded")
        self.assertEqual(first.freshness_frames, 1)
        self.assertEqual(second.freshness_frames, 2)
        self.assertEqual(third.status, "unavailable")
        self.assertEqual(third.quality, "unavailable")
        self.assertEqual(third.freshness_frames, 3)
        self.assertFalse(first.usable)

    def test_smoothed_source_is_retained_while_distance_is_stale(self) -> None:
        fusion = DistanceFusion()
        current = fusion.update(
            [stable()],
            frame(1.3, depth_source="smoothed_scene_depth"),
        )[0].distance
        stale = fusion.update([stable()], None)[0].distance

        self.assertEqual(current.depth_source, "smoothed_scene_depth")
        self.assertEqual(stale.depth_source, "smoothed_scene_depth")
        self.assertEqual(stale.quality, "degraded")
        self.assertEqual(stale.candidate_status, "unavailable")

    def test_none_depth_marks_tracks_missing_and_absent_tracks_are_forgotten(self) -> None:
        fusion = DistanceFusion()
        fusion.update([stable()], frame(2.0))

        missing = fusion.update([stable()], None)[0].distance
        empty = fusion.update([], None)

        self.assertTrue(missing.is_stale)
        self.assertEqual(empty, ())
        self.assertEqual(fusion.tracked_count, 0)

    def test_rejects_unregistered_frame_dimensions(self) -> None:
        depth = DepthFrame([[1.0] * 5 for _ in range(5)], [[2] * 5 for _ in range(5)], 20, 20)

        with self.assertRaisesRegex(ValueError, "registered"):
            DistanceFusion().update([stable()], depth)

    def test_maps_clipped_landscape_rgb_box_into_lower_resolution_depth(self) -> None:
        depth = DepthFrame(
            distances_m=[[2.0] * 256 for _ in range(192)],
            confidence=[[2] * 256 for _ in range(192)],
            frame_width=640,
            frame_height=480,
        )
        clipped = stable(
            bbox=(-80, -40, 720, 520),
            frame_width=640,
            frame_height=480,
        )

        estimate = DistanceFusion().update([clipped], depth)[0].distance

        # Clipping gives the full RGB frame; 15% inset maps to x=38..217,
        # y=28..163 in the 256x192 registered depth grid.
        self.assertEqual(estimate.roi_pixel_count, 180 * 136)
        self.assertEqual(estimate.sample_count, 180 * 136)
        self.assertEqual(estimate.distance_m, 2.0)

    def test_portrait_registered_coordinates_are_not_rotated_in_domain_layer(self) -> None:
        values = [[0.8 if row < 128 else 3.0 for _ in range(192)] for row in range(256)]
        depth = DepthFrame(
            distances_m=values,
            confidence=[[2] * 192 for _ in range(256)],
            frame_width=480,
            frame_height=640,
            depth_source="smoothed_scene_depth",
        )
        upper_portrait_box = stable(
            bbox=(120, 64, 360, 256),
            frame_width=480,
            frame_height=640,
        )

        estimate = DistanceFusion().update([upper_portrait_box], depth)[0].distance

        self.assertEqual(estimate.distance_m, 0.8)
        self.assertEqual(estimate.candidate_center_median_m, 0.8)
        self.assertEqual(estimate.depth_source, "smoothed_scene_depth")


if __name__ == "__main__":
    unittest.main()
