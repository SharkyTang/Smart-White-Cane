from __future__ import annotations

import json
import struct
import unittest

import numpy as np

from smart_cane.rgbd_protocol import (
    HEADER,
    MAX_REQUEST_BYTES,
    RGBDProtocolError,
    parse_rgbd_request,
)


JPEG_BYTES = b"\xff\xd8rgb\xff\xd9"
IOS_FIXED_METADATA_JSON = (
    b'{"depth_height":1,"depth_source":"raw_scene_depth","depth_width":1,'
    b'"frame_id":0,"orientation":"portrait_up","rgb_height":1,"rgb_width":1,'
    b'"scenario_id":null,"schema_version":1,"session_id":"s",'
    b'"source":"arkit_scene_depth","timestamp_seconds":1.25}'
)


def packet(
    *,
    metadata: dict[str, object] | None = None,
    raw_metadata: bytes | None = None,
    jpeg: bytes = JPEG_BYTES,
    depth: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    header: struct.Struct = HEADER,
) -> bytes:
    values = metadata or {
        "schema_version": 1,
        "session_id": "session-1",
        "frame_id": 7,
        "timestamp_seconds": 12.5,
        "rgb_width": 4,
        "rgb_height": 2,
        "depth_width": 2,
        "depth_height": 1,
        "source": "arkit_scene_depth",
        "depth_source": "raw_scene_depth",
        "orientation": "portrait_up",
        "scenario_id": None,
    }
    metadata_bytes = raw_metadata or json.dumps(values, separators=(",", ":")).encode()
    depth_bytes = (
        np.asarray([[1.25, 2.5]], dtype="<f2") if depth is None else depth.astype("<f2")
    ).tobytes()
    confidence_bytes = (
        np.asarray([[2, 1]], dtype=np.uint8) if confidence is None else confidence.astype(np.uint8)
    ).tobytes()
    return b"".join(
        (
            header.pack(
                b"SCF1",
                len(metadata_bytes),
                len(jpeg),
                len(depth_bytes),
                len(confidence_bytes),
            ),
            metadata_bytes,
            jpeg,
            depth_bytes,
            confidence_bytes,
        )
    )


class RGBDProtocolTests(unittest.TestCase):
    def test_parses_exact_ios_fixed_vector(self) -> None:
        """Keep the Python parser locked to the Swift XCTest SCF1 vector."""

        jpeg = b"\xff\xd8\xff\xd9"
        depth = b"\x00\x3e"  # Float16(1.5), little-endian
        confidence = b"\x02"
        body = b"".join(
            (
                HEADER.pack(
                    b"SCF1",
                    len(IOS_FIXED_METADATA_JSON),
                    len(jpeg),
                    len(depth),
                    len(confidence),
                ),
                IOS_FIXED_METADATA_JSON,
                jpeg,
                depth,
                confidence,
            )
        )

        parsed = parse_rgbd_request(body)

        self.assertEqual(parsed.session_id, "s")
        self.assertEqual(parsed.frame_id, 0)
        np.testing.assert_array_equal(parsed.depth_meters, [[1.5]])
        np.testing.assert_array_equal(parsed.confidence, [[2]])

    def test_parses_network_order_lengths_and_little_endian_depth(self) -> None:
        body = packet()
        parsed = parse_rgbd_request(body)

        self.assertEqual(body[:4], b"SCF1")
        self.assertGreater(struct.unpack(">I", body[4:8])[0], 0)
        self.assertEqual(parsed.session_id, "session-1")
        self.assertEqual(parsed.frame_id, 7)
        self.assertEqual((parsed.rgb_width, parsed.rgb_height), (4, 2))
        self.assertEqual((parsed.depth_width, parsed.depth_height), (2, 1))
        np.testing.assert_array_equal(parsed.depth_meters, [[1.25, 2.5]])
        np.testing.assert_array_equal(parsed.confidence, [[2, 1]])
        self.assertFalse(parsed.depth_meters.flags.writeable)
        self.assertEqual(parsed.metadata["source"], "arkit_scene_depth")

    def test_rejects_wrong_magic_truncation_trailing_bytes_and_little_endian_lengths(self) -> None:
        valid = packet()
        cases = (
            b"BAD!" + valid[4:],
            valid[: HEADER.size - 1],
            valid[:-1],
            valid + b"trailing",
            packet(header=struct.Struct("<4sIIII")),
        )
        for body in cases:
            with self.subTest(size=len(body)):
                with self.assertRaises(RGBDProtocolError):
                    parse_rgbd_request(body)

    def test_rejects_oversized_body_before_parsing(self) -> None:
        with self.assertRaisesRegex(RGBDProtocolError, "exceeds") as caught:
            parse_rgbd_request(b"x" * (MAX_REQUEST_BYTES + 1))
        self.assertEqual(caught.exception.code, "frame_too_large")

    def test_rejects_invalid_json_ids_and_dimensions(self) -> None:
        valid = {
            "schema_version": 1,
            "session_id": "session-1",
            "frame_id": 1,
            "timestamp_seconds": 12.5,
            "rgb_width": 4,
            "rgb_height": 2,
            "depth_width": 2,
            "depth_height": 1,
            "source": "arkit_scene_depth",
            "depth_source": "raw_scene_depth",
            "orientation": "portrait_up",
            "scenario_id": None,
        }
        invalid_values = (
            {**valid, "session_id": " "},
            {**valid, "frame_id": True},
            {**valid, "frame_id": -1},
            {**valid, "depth_width": 0},
            {**valid, "rgb_height": 2.0},
        )
        for metadata in invalid_values:
            with self.subTest(metadata=metadata):
                with self.assertRaises(RGBDProtocolError):
                    parse_rgbd_request(packet(metadata=metadata))

        malformed = (b"[]", b'{"frame_id":NaN}', b'{"frame_id":1,"frame_id":2}')
        for raw_metadata in malformed:
            with self.subTest(raw_metadata=raw_metadata):
                with self.assertRaisesRegex(RGBDProtocolError, "JSON object"):
                    parse_rgbd_request(packet(raw_metadata=raw_metadata))

    def test_rejects_aspect_ratio_and_payload_dimension_mismatches(self) -> None:
        wrong_aspect = {
            "session_id": "session-1",
            "frame_id": 1,
            "rgb_width": 4,
            "rgb_height": 3,
            "depth_width": 2,
            "depth_height": 1,
        }
        with self.assertRaisesRegex(RGBDProtocolError, "aspect ratio"):
            parse_rgbd_request(packet(metadata=wrong_aspect))

        wrong_depth_shape = np.asarray([[1.0]], dtype="<f2")
        with self.assertRaisesRegex(RGBDProtocolError, "Float16"):
            parse_rgbd_request(packet(depth=wrong_depth_shape))

        wrong_confidence_shape = np.asarray([[1]], dtype=np.uint8)
        with self.assertRaisesRegex(RGBDProtocolError, "UInt8"):
            parse_rgbd_request(packet(confidence=wrong_confidence_shape))

    def test_rejects_non_finite_depth_and_invalid_jpeg_markers(self) -> None:
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                depth = np.asarray([[value, 1.0]], dtype="<f2")
                with self.assertRaisesRegex(RGBDProtocolError, "NaN or infinity"):
                    parse_rgbd_request(packet(depth=depth))

        with self.assertRaisesRegex(RGBDProtocolError, "JPEG"):
            parse_rgbd_request(packet(jpeg=b"not-a-jpeg"))

    def test_requires_fixed_arkit_metadata_contract(self) -> None:
        base = {
            "schema_version": 1,
            "session_id": "session-1",
            "frame_id": 1,
            "timestamp_seconds": 12.5,
            "rgb_width": 4,
            "rgb_height": 2,
            "depth_width": 2,
            "depth_height": 1,
            "source": "arkit_scene_depth",
            "depth_source": "raw_scene_depth",
            "orientation": "portrait_up",
            "scenario_id": None,
        }
        invalid = (
            {**base, "schema_version": 2},
            {**base, "timestamp_seconds": float("inf")},
            {**base, "source": "other"},
            {**base, "depth_source": "estimated"},
            {**base, "orientation": "landscape_left"},
            {key: value for key, value in base.items() if key != "scenario_id"},
            {**base, "scenario_id": " "},
        )
        for metadata in invalid:
            with self.subTest(metadata=metadata):
                with self.assertRaises(RGBDProtocolError):
                    parse_rgbd_request(packet(metadata=metadata))

        parsed = parse_rgbd_request(
            packet(
                metadata={
                    **base,
                    "depth_source": "smoothed_scene_depth",
                    "scenario_id": "indoor-static-01",
                }
            )
        )
        self.assertEqual(parsed.metadata["depth_source"], "smoothed_scene_depth")
        self.assertEqual(parsed.metadata["scenario_id"], "indoor-static-01")


if __name__ == "__main__":
    unittest.main()
