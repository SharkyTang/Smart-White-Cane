"""Strict parser for the Smart Cane RGB-D frame wire format.

Version 1 is intentionally small and self-delimiting::

    SCF1 | json_len | jpeg_len | depth_len | confidence_len | payloads...

The four lengths are unsigned 32-bit integers in network byte order.  Depth
samples are the sole little-endian field: one Float16 metre value per depth
pixel.  Confidence contains one UInt8 value per depth pixel.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


MAGIC = b"SCF1"
HEADER = struct.Struct(">4sIIII")
MAX_REQUEST_BYTES = 2_000_000
MAX_METADATA_BYTES = 64 * 1024
MAX_DIMENSION = 8192


class RGBDProtocolError(ValueError):
    """Raised when an RGB-D request violates the version-1 wire contract."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class ParsedRGBDRequest:
    """Validated transport values from one SCF1 request.

    This is a transport DTO, not the depth-fusion domain model.  The service
    adapts these arrays to ``smart_cane.depth.DepthFrame`` at its boundary.
    """

    session_id: str
    frame_id: int
    rgb_width: int
    rgb_height: int
    depth_width: int
    depth_height: int
    metadata: Mapping[str, Any]
    jpeg_bytes: bytes
    depth_meters: np.ndarray
    confidence: np.ndarray


def parse_rgbd_request(
    body: bytes,
    *,
    max_request_bytes: int = MAX_REQUEST_BYTES,
) -> ParsedRGBDRequest:
    """Parse and validate one complete SCF1 request body."""

    if not isinstance(body, bytes):
        raise TypeError("body must be bytes")
    if (
        isinstance(max_request_bytes, bool)
        or not isinstance(max_request_bytes, int)
        or max_request_bytes <= 0
    ):
        raise ValueError("max_request_bytes must be a positive integer")
    if len(body) > max_request_bytes:
        raise RGBDProtocolError(
            "frame_too_large",
            f"RGB-D body exceeds {max_request_bytes} bytes",
        )
    if len(body) < HEADER.size:
        raise RGBDProtocolError("invalid_header", "RGB-D body is shorter than the SCF1 header")

    magic, metadata_length, jpeg_length, depth_length, confidence_length = HEADER.unpack_from(body)
    if magic != MAGIC:
        raise RGBDProtocolError("invalid_magic", "RGB-D body must start with SCF1")
    if metadata_length == 0 or metadata_length > MAX_METADATA_BYTES:
        raise RGBDProtocolError(
            "invalid_metadata_length",
            f"metadata length must be between 1 and {MAX_METADATA_BYTES} bytes",
        )
    if jpeg_length == 0 or depth_length == 0 or confidence_length == 0:
        raise RGBDProtocolError(
            "empty_payload",
            "JPEG, depth, and confidence payloads must all be non-empty",
        )

    declared_size = HEADER.size + metadata_length + jpeg_length + depth_length + confidence_length
    if declared_size != len(body):
        raise RGBDProtocolError(
            "length_mismatch",
            "SCF1 payload lengths do not match the request body",
        )

    metadata_start = HEADER.size
    jpeg_start = metadata_start + metadata_length
    depth_start = jpeg_start + jpeg_length
    confidence_start = depth_start + depth_length

    metadata = _parse_metadata(body[metadata_start:jpeg_start])
    jpeg_bytes = body[jpeg_start:depth_start]
    if not jpeg_bytes.startswith(b"\xff\xd8") or not jpeg_bytes.endswith(b"\xff\xd9"):
        raise RGBDProtocolError("invalid_jpeg", "JPEG payload is missing its SOI or EOI marker")

    rgb_width = _required_positive_int(metadata, "rgb_width")
    rgb_height = _required_positive_int(metadata, "rgb_height")
    depth_width = _required_positive_int(metadata, "depth_width")
    depth_height = _required_positive_int(metadata, "depth_height")
    if rgb_width * depth_height != depth_width * rgb_height:
        raise RGBDProtocolError(
            "aspect_ratio_mismatch",
            "RGB and depth dimensions must have the same aspect ratio",
        )

    depth_pixels = depth_width * depth_height
    expected_depth_length = depth_pixels * np.dtype("<f2").itemsize
    if depth_length != expected_depth_length:
        raise RGBDProtocolError(
            "depth_size_mismatch",
            f"depth payload must contain exactly {depth_pixels} Float16 samples",
        )
    if confidence_length != depth_pixels:
        raise RGBDProtocolError(
            "confidence_size_mismatch",
            f"confidence payload must contain exactly {depth_pixels} UInt8 samples",
        )

    depth_meters = np.frombuffer(
        body[depth_start:confidence_start],
        dtype=np.dtype("<f2"),
    ).reshape((depth_height, depth_width))
    if not np.isfinite(depth_meters).all():
        raise RGBDProtocolError("non_finite_depth", "depth payload contains NaN or infinity")
    confidence = np.frombuffer(
        body[confidence_start:],
        dtype=np.uint8,
    ).reshape((depth_height, depth_width))
    if np.any(confidence > 2):
        raise RGBDProtocolError(
            "invalid_confidence",
            "confidence samples must use ARKit values 0, 1, or 2",
        )
    _validate_fixed_metadata(metadata)
    timestamp = metadata["timestamp_seconds"]
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, (int, float))
        or not isfinite(float(timestamp))
        or float(timestamp) < 0.0
    ):
        raise RGBDProtocolError(
            "invalid_timestamp",
            "timestamp_seconds must be a finite non-negative number when present",
        )
    depth_meters.setflags(write=False)
    confidence.setflags(write=False)

    return ParsedRGBDRequest(
        session_id=_required_session_id(metadata),
        frame_id=_required_frame_id(metadata),
        rgb_width=rgb_width,
        rgb_height=rgb_height,
        depth_width=depth_width,
        depth_height=depth_height,
        metadata=MappingProxyType(metadata),
        jpeg_bytes=jpeg_bytes,
        depth_meters=depth_meters,
        confidence=confidence,
    )


def _validate_fixed_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("schema_version") != 1 or isinstance(metadata.get("schema_version"), bool):
        raise RGBDProtocolError("unsupported_schema", "schema_version must be integer 1")
    if metadata.get("source") != "arkit_scene_depth":
        raise RGBDProtocolError("unsupported_source", "source must be arkit_scene_depth")
    if metadata.get("depth_source") not in (
        "raw_scene_depth",
        "smoothed_scene_depth",
    ):
        raise RGBDProtocolError(
            "unsupported_depth_source",
            "depth_source must be raw_scene_depth or smoothed_scene_depth",
        )
    if metadata.get("orientation") != "portrait_up":
        raise RGBDProtocolError("unsupported_orientation", "orientation must be portrait_up")
    if "scenario_id" not in metadata:
        raise RGBDProtocolError("invalid_scenario_id", "scenario_id key must be present")
    scenario_id = metadata["scenario_id"]
    if scenario_id is not None and (not isinstance(scenario_id, str) or not scenario_id.strip()):
        raise RGBDProtocolError(
            "invalid_scenario_id",
            "scenario_id must be null or a non-blank string",
        )
    if "timestamp_seconds" not in metadata:
        raise RGBDProtocolError("invalid_timestamp", "timestamp_seconds key must be present")


def _parse_metadata(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RGBDProtocolError("invalid_metadata", "metadata must be UTF-8 JSON") from error

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate metadata key {key!r}")
            result[key] = value
        return result

    try:
        metadata = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise RGBDProtocolError(
            "invalid_metadata", "metadata must be a valid JSON object"
        ) from error
    if not isinstance(metadata, dict):
        raise RGBDProtocolError("invalid_metadata", "metadata must be a JSON object")
    return metadata


def _required_session_id(metadata: Mapping[str, Any]) -> str:
    value = metadata.get("session_id")
    if not isinstance(value, str) or not value.strip() or len(value) > 128:
        raise RGBDProtocolError(
            "invalid_session_id",
            "session_id must be a non-blank string of at most 128 characters",
        )
    if value != value.strip():
        raise RGBDProtocolError("invalid_session_id", "session_id must not have outer whitespace")
    return value


def _required_frame_id(metadata: Mapping[str, Any]) -> int:
    value = metadata.get("frame_id")
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= (2**63 - 1):
        raise RGBDProtocolError(
            "invalid_frame_id",
            "frame_id must be an integer between 0 and 2^63-1",
        )
    return value


def _required_positive_int(metadata: Mapping[str, Any], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > MAX_DIMENSION:
        raise RGBDProtocolError(
            "invalid_dimensions",
            f"{key} must be an integer between 1 and {MAX_DIMENSION}",
        )
    return value
