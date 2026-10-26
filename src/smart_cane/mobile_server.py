"""Small HTTP(S) server for the iPhone camera prototype.

The server deliberately uses only the Python standard library for transport.
JPEG bytes are decoded in memory and are never written to disk.  Model
inference and the stateful decision pipeline are serialized so concurrent HTTP
requests cannot race the detector, temporal tracker, or alert cooldown.
"""

from __future__ import annotations

import json
import ssl
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from os import PathLike
from typing import Any, TypeAlias
from urllib.parse import urlsplit

import numpy as np

from smart_cane.app import evaluate_detections
from smart_cane.config import RuntimeConfig
from smart_cane.decision.risk_engine import (
    AreaThresholds,
    DistanceRiskRules,
    RiskEngine,
    RiskRules,
)
from smart_cane.decision.temporal_filter import TemporalFilter, TemporalFilterConfig
from smart_cane.depth import DepthFrame, DistanceFusion, FusedDetection
from smart_cane.feedback.alert_policy import AlertDecision, AlertPolicy
from smart_cane.models import Detection
from smart_cane.perception.detector import YoloDetector
from smart_cane.rgbd_protocol import ParsedRGBDRequest, RGBDProtocolError, parse_rgbd_request


DEFAULT_MAX_REQUEST_BYTES = 2_000_000
DEFAULT_STATIC_PACKAGE = "smart_cane.mobile"
RGBD_CONTENT_TYPE = "application/vnd.smart-cane.rgbd-frame"
RGBD_CONTENT_VERSION = "1"

_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
}

JsonObject: TypeAlias = dict[str, Any]
Delivery: TypeAlias = Callable[[JsonObject], None]


class FrameDecodeError(ValueError):
    """Raised when a request body is not a decodable image."""


class RGBDStateError(ValueError):
    """Raised when an RGB-D frame or acknowledgement is stale or unknown."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(slots=True)
class _PendingRGBDAlert:
    alert_id: str
    key: tuple[str, str, str, str, str]
    track_key: str
    decision: AlertDecision
    frame_id: int


class MobilePipeline:
    """Decode JPEG frames and run the existing smart-cane decision pipeline."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        detector: Any | None = None,
        cv2_module: Any | None = None,
        temporal_filter: TemporalFilter | None = None,
        risk_engine: RiskEngine | None = None,
        alert_policy: AlertPolicy | None = None,
        distance_fusion: DistanceFusion | None = None,
    ) -> None:
        if not isinstance(config, RuntimeConfig):
            raise TypeError("config must be a RuntimeConfig")

        self.config = config
        self.detector = detector if detector is not None else _build_detector(config)
        self.cv2 = cv2_module if cv2_module is not None else _load_cv2()
        self.temporal_filter = (
            temporal_filter if temporal_filter is not None else _build_temporal_filter(config)
        )
        self.risk_engine = risk_engine if risk_engine is not None else _build_risk_engine(config)
        self.alert_policy = (
            alert_policy if alert_policy is not None else AlertPolicy(config.alert_cooldown_seconds)
        )
        self.distance_fusion = (
            distance_fusion if distance_fusion is not None else _build_distance_fusion(config)
        )
        self._session_id = uuid.uuid4().hex
        self._last_rgbd_frame_id = -1
        self._pending_rgbd_alerts: dict[str, _PendingRGBDAlert] = {}
        self._pending_rgbd_alert_ids: dict[tuple[str, str, str, str, str], str] = {}
        self._acknowledged_rgbd_alerts: dict[str, int] = {}
        self._lock = threading.Lock()

    @property
    def session_id(self) -> str:
        """Current RGB-D session identifier returned by the last reset."""

        with self._lock:
            return self._session_id

    def deliver_frame(self, jpeg_bytes: bytes, delivery: Delivery) -> JsonObject:
        """Process one in-memory JPEG and commit an alert after delivery succeeds.

        ``delivery`` normally writes the HTTP response.  Keeping it inside the
        lock makes inference, state updates, delivery, and alert commit one
        ordered transaction.  If delivery raises (for example, because the
        phone disconnected), the alert remains uncommitted and is eligible on
        the next frame.
        """

        if not isinstance(jpeg_bytes, bytes):
            raise TypeError("jpeg_bytes must be bytes")
        if not callable(delivery):
            raise TypeError("delivery must be callable")

        with self._lock:
            result, alert_to_commit = self._process_frame(jpeg_bytes)
            delivery(result)
            if alert_to_commit is not None:
                track_key, alert = alert_to_commit
                self.alert_policy.commit(track_key, alert)
            return result

    def deliver_rgbd_frame(
        self,
        request: ParsedRGBDRequest,
        delivery: Delivery,
    ) -> JsonObject:
        """Process one ordered RGB-D request without committing its alert.

        RGB-D alerts use explicit client acknowledgement.  A candidate keeps
        the same ``alert_id`` across subsequent frames until acknowledged.
        """

        if not isinstance(request, ParsedRGBDRequest):
            raise TypeError("request must be a ParsedRGBDRequest")
        if not callable(delivery):
            raise TypeError("delivery must be callable")

        with self._lock:
            self._validate_rgbd_order(request.session_id, request.frame_id)
            result = self._process_rgbd_frame(request)
            self._last_rgbd_frame_id = request.frame_id
            delivery(result)
            return result

    def acknowledge_alert(self, session_id: str, frame_id: int, alert_id: str) -> JsonObject:
        """Commit one RGB-D alert, with idempotency for a repeated ACK."""

        if not isinstance(session_id, str) or not session_id:
            raise RGBDStateError("invalid_session_id", "session_id must be a non-empty string")
        if isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0:
            raise RGBDStateError("invalid_frame_id", "frame_id must be a non-negative integer")
        if not isinstance(alert_id, str) or not alert_id:
            raise RGBDStateError("invalid_alert_id", "alert_id must be a non-empty string")

        with self._lock:
            if session_id != self._session_id:
                raise RGBDStateError("stale_session", "session_id is not current")

            acknowledged_frame = self._acknowledged_rgbd_alerts.get(alert_id)
            if acknowledged_frame is not None:
                if acknowledged_frame != frame_id:
                    raise RGBDStateError("stale_frame", "alert was acknowledged for another frame")
                return {
                    "status": "ok",
                    "session_id": session_id,
                    "frame_id": frame_id,
                    "alert_id": alert_id,
                    "acknowledged": True,
                    "already_acknowledged": True,
                }

            pending = self._pending_rgbd_alerts.get(alert_id)
            if pending is None:
                raise RGBDStateError("unknown_alert", "alert_id is not pending")
            if pending.frame_id != frame_id:
                raise RGBDStateError("stale_frame", "alert_id does not belong to this frame")

            self.alert_policy.commit(pending.track_key, pending.decision)
            self._acknowledged_rgbd_alerts[alert_id] = frame_id
            del self._pending_rgbd_alerts[alert_id]
            self._pending_rgbd_alert_ids.pop(pending.key, None)
            return {
                "status": "ok",
                "session_id": session_id,
                "frame_id": frame_id,
                "alert_id": alert_id,
                "acknowledged": True,
                "already_acknowledged": False,
            }

    def reset_session(self) -> JsonObject:
        """Clear tracking and alert history for the single-phone prototype."""

        with self._lock:
            temporal_tracks = self.temporal_filter.track_count
            alert_tracks = self.alert_policy.tracked_count
            distance_tracks = self.distance_fusion.tracked_count
            distance_risk_tracks = self.risk_engine.distance_state.tracked_count
            self.temporal_filter.reset()
            self.distance_fusion.reset()
            self.risk_engine.reset()
            self.alert_policy.retain_active(())
            self._session_id = uuid.uuid4().hex
            self._last_rgbd_frame_id = -1
            self._pending_rgbd_alerts.clear()
            self._pending_rgbd_alert_ids.clear()
            self._acknowledged_rgbd_alerts.clear()
            session_id = self._session_id
        return {
            "status": "ok",
            "reset": True,
            "session_id": session_id,
            "model_name": self.config.model,
            "cleared": {
                "temporal_tracks": temporal_tracks,
                "alert_tracks": alert_tracks,
                "distance_tracks": distance_tracks,
                "distance_risk_tracks": distance_risk_tracks,
            },
        }

    def _validate_rgbd_order(self, session_id: str, frame_id: int) -> None:
        if session_id != self._session_id:
            raise RGBDStateError("stale_session", "session_id is not current")
        if frame_id <= self._last_rgbd_frame_id:
            raise RGBDStateError(
                "stale_frame",
                "frame_id must be greater than the last accepted frame",
            )

    def _process_rgbd_frame(self, request: ParsedRGBDRequest) -> JsonObject:
        total_started = time.perf_counter()

        decode_started = time.perf_counter()
        frame = self._decode_jpeg(request.jpeg_bytes)
        self._validate_decoded_dimensions(frame, request.rgb_width, request.rgb_height)
        decode_seconds = time.perf_counter() - decode_started

        inference_started = time.perf_counter()
        detections = tuple(self.detector(frame))
        inference_seconds = time.perf_counter() - inference_started

        decision_started = time.perf_counter()
        stable = self.temporal_filter.update(detections)
        depth_frame = DepthFrame(
            distances_m=request.depth_meters,
            confidence=request.confidence,
            frame_width=request.rgb_width,
            frame_height=request.rgb_height,
            frame_id=request.frame_id,
            captured_at=request.metadata.get("timestamp_seconds"),
            depth_source=request.metadata["depth_source"],
        )
        fused = self.distance_fusion.update(stable, depth_frame)
        self.risk_engine.retain_active(item.track_key for item in fused)
        assessments = tuple(self.risk_engine.assess(item) for item in fused)
        self.alert_policy.retain_active(item.track_key for item in fused)

        alert = None
        alert_track_key = None
        for index in _urgency_order(assessments):
            track_key = fused[index].track_key
            candidate = self.alert_policy.propose(track_key, assessments[index])
            if alert is None:
                alert = candidate
                alert_track_key = track_key
            if candidate.should_emit:
                alert = candidate
                alert_track_key = track_key
                break
        decision_seconds = time.perf_counter() - decision_started
        total_seconds = time.perf_counter() - total_started

        alert_json = None
        if alert is not None:
            alert_json = _alert_to_json(alert, alert_track_key)
            if alert.should_emit:
                assert alert_track_key is not None
                pending = self._pending_for(alert_track_key, alert, request.frame_id)
                alert_json["alert_id"] = pending.alert_id
                alert_json["alert_frame_id"] = pending.frame_id

        return {
            "session_id": request.session_id,
            "frame_id": request.frame_id,
            "model_name": self.config.model,
            "detections": [_detection_to_json(item) for item in detections],
            "assessments": [
                _fused_assessment_to_json(item, assessment)
                for item, assessment in zip(fused, assessments, strict=True)
            ],
            "alert": alert_json,
            "timing": {
                "decode_ms": decode_seconds * 1000.0,
                "inference_ms": inference_seconds * 1000.0,
                "decision_ms": decision_seconds * 1000.0,
                "total_ms": total_seconds * 1000.0,
            },
        }

    def _pending_for(
        self,
        track_key: str,
        decision: AlertDecision,
        frame_id: int,
    ) -> _PendingRGBDAlert:
        assessment = decision.assessment
        key = (
            track_key,
            assessment.label.casefold(),
            assessment.direction,
            assessment.level,
            decision.reason,
        )
        pending_id = self._pending_rgbd_alert_ids.get(key)
        pending = self._pending_rgbd_alerts.get(pending_id) if pending_id is not None else None
        if pending is None:
            pending = _PendingRGBDAlert(
                alert_id=uuid.uuid4().hex,
                key=key,
                track_key=track_key,
                decision=decision,
                frame_id=frame_id,
            )
            self._pending_rgbd_alerts[pending.alert_id] = pending
            self._pending_rgbd_alert_ids[key] = pending.alert_id
        else:
            pending.decision = decision
        return pending

    @staticmethod
    def _validate_decoded_dimensions(frame: Any, width: int, height: int) -> None:
        shape = getattr(frame, "shape", None)
        if not isinstance(shape, tuple) or len(shape) < 2:
            raise FrameDecodeError("decoded JPEG does not expose image dimensions")
        if (int(shape[1]), int(shape[0])) != (width, height):
            raise FrameDecodeError("decoded JPEG dimensions do not match RGB metadata")

    def _process_frame(
        self, jpeg_bytes: bytes
    ) -> tuple[JsonObject, tuple[str, AlertDecision] | None]:
        total_started = time.perf_counter()

        decode_started = time.perf_counter()
        frame = self._decode_jpeg(jpeg_bytes)
        decode_seconds = time.perf_counter() - decode_started

        inference_started = time.perf_counter()
        detections = tuple(self.detector(frame))
        inference_seconds = time.perf_counter() - inference_started

        decision_started = time.perf_counter()
        decision = evaluate_detections(
            detections,
            temporal_filter=self.temporal_filter,
            risk_engine=self.risk_engine,
            alert_policy=self.alert_policy,
            commit_alert=False,
        )
        decision_seconds = time.perf_counter() - decision_started
        total_seconds = time.perf_counter() - total_started

        alert_json = None
        alert_to_commit = None
        if decision.alert is not None:
            alert_json = _alert_to_json(decision.alert, decision.alert_track_key)
            if decision.alert.should_emit:
                assert decision.alert_track_key is not None
                alert_to_commit = (decision.alert_track_key, decision.alert)

        result: JsonObject = {
            "model_name": self.config.model,
            "detections": [_detection_to_json(item) for item in detections],
            "assessments": [
                _assessment_to_json(stable, assessment)
                for stable, assessment in zip(
                    decision.stable_detections,
                    decision.assessments,
                    strict=True,
                )
            ],
            "alert": alert_json,
            "timing": {
                "decode_ms": decode_seconds * 1000.0,
                "inference_ms": inference_seconds * 1000.0,
                "decision_ms": decision_seconds * 1000.0,
                "total_ms": total_seconds * 1000.0,
            },
        }
        return result, alert_to_commit

    def _decode_jpeg(self, jpeg_bytes: bytes) -> Any:
        if not jpeg_bytes:
            raise FrameDecodeError("JPEG body cannot be empty")
        encoded = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        try:
            frame = self.cv2.imdecode(encoded, self.cv2.IMREAD_COLOR)
        except Exception as error:
            raise FrameDecodeError("request body is not a decodable JPEG") from error
        if frame is None:
            raise FrameDecodeError("request body is not a decodable JPEG")
        return frame


class MobileHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying immutable resources and one shared pipeline."""

    daemon_threads = True
    allow_reuse_address = True

    pipeline: MobilePipeline
    static_package: str
    max_request_bytes: int


class MobileRequestHandler(BaseHTTPRequestHandler):
    """Serve the mobile page and its two JSON API endpoints."""

    protocol_version = "HTTP/1.1"
    server_version = "SmartCaneMobile/0.4"
    sys_version = ""

    server: MobileHTTPServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        path = urlsplit(self.path).path
        if path == "/healthz":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return

        static_file = _STATIC_FILES.get(path)
        if static_file is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", "resource not found")
            return

        resource_name, content_type = static_file
        try:
            body = resources.files(self.server.static_package).joinpath(resource_name).read_bytes()
        except (FileNotFoundError, ModuleNotFoundError, TypeError):
            self._send_error_json(HTTPStatus.NOT_FOUND, "not_found", "resource not found")
            return
        self._send_bytes(HTTPStatus.OK, body, content_type)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if not self._request_origin_is_allowed():
            self._send_error_json(
                HTTPStatus.FORBIDDEN,
                "origin_not_allowed",
                "request origin does not match this server",
                close=True,
            )
            return
        path = urlsplit(self.path).path
        if path == "/api/frame":
            self._handle_frame()
            return
        if path == "/api/rgbd-frame":
            self._handle_rgbd_frame()
            return
        if path == "/api/alert/ack":
            self._handle_alert_ack()
            return
        if path == "/api/session/reset":
            if not self._consume_optional_empty_body():
                return
            self._send_json(HTTPStatus.OK, self.server.pipeline.reset_session())
            return
        self._send_error_json(
            HTTPStatus.NOT_FOUND,
            "not_found",
            "endpoint not found",
            close=True,
        )

    def _request_origin_is_allowed(self) -> bool:
        """Reject browser cross-origin writes while retaining curl/test access."""

        origin = self.headers.get("Origin")
        if origin is None:
            return True
        host = self.headers.get("Host")
        if host is None:
            return False
        parsed = urlsplit(origin)
        return parsed.scheme in {"http", "https"} and parsed.netloc.casefold() == host.casefold()

    def _handle_frame(self) -> None:
        if self.headers.get_content_type().casefold() != "image/jpeg":
            self._send_error_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "Content-Type must be image/jpeg",
                close=True,
            )
            return

        payload = self._read_required_body()
        if payload is None:
            return

        try:
            self.server.pipeline.deliver_frame(
                payload,
                lambda result: self._send_json(HTTPStatus.OK, result),
            )
        except FrameDecodeError as error:
            self._send_error_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "invalid_image",
                str(error),
            )
        except (BrokenPipeError, ConnectionError, OSError):
            self.close_connection = True
        except Exception as error:  # detector failures are reported without leaking internals
            self.log_error("frame processing failed: %s", error)
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "processing_failed",
                "frame processing failed",
            )

    def _handle_rgbd_frame(self) -> None:
        content_type = self.headers.get_content_type().casefold()
        version = self.headers.get_param("version", header="content-type")
        if content_type != RGBD_CONTENT_TYPE or version != RGBD_CONTENT_VERSION:
            self._send_error_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "Content-Type must be application/vnd.smart-cane.rgbd-frame;version=1",
                close=True,
            )
            return

        payload = self._read_required_body()
        if payload is None:
            return

        try:
            request = parse_rgbd_request(
                payload,
                max_request_bytes=self.server.max_request_bytes,
            )
            self.server.pipeline.deliver_rgbd_frame(
                request,
                lambda result: self._send_json(HTTPStatus.OK, result),
            )
        except RGBDProtocolError as error:
            if error.code == "frame_too_large":
                status = HTTPStatus.REQUEST_ENTITY_TOO_LARGE
            elif error.code == "invalid_jpeg":
                status = HTTPStatus.UNPROCESSABLE_ENTITY
            else:
                status = HTTPStatus.BAD_REQUEST
            self._send_error_json(status, error.code, str(error))
        except RGBDStateError as error:
            self._send_error_json(HTTPStatus.CONFLICT, error.code, str(error))
        except FrameDecodeError as error:
            self._send_error_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                "invalid_image",
                str(error),
            )
        except (BrokenPipeError, ConnectionError, OSError):
            self.close_connection = True
        except Exception as error:  # detector and fusion failures hide internals
            self.log_error("RGB-D frame processing failed: %s", error)
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                "processing_failed",
                "RGB-D frame processing failed",
            )

    def _handle_alert_ack(self) -> None:
        if self.headers.get_content_type().casefold() != "application/json":
            self._send_error_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                "unsupported_media_type",
                "Content-Type must be application/json",
                close=True,
            )
            return
        body = self._read_required_body()
        if body is None:
            return
        try:
            payload = _parse_json_object(body)
            result = self.server.pipeline.acknowledge_alert(
                payload.get("session_id"),
                payload.get("frame_id"),
                payload.get("alert_id"),
            )
            self._send_json(HTTPStatus.OK, result)
        except RGBDProtocolError as error:
            self._send_error_json(HTTPStatus.BAD_REQUEST, error.code, str(error))
        except RGBDStateError as error:
            status = (
                HTTPStatus.CONFLICT
                if error.code in {"stale_session", "stale_frame", "unknown_alert"}
                else HTTPStatus.BAD_REQUEST
            )
            self._send_error_json(status, error.code, str(error))
        except (BrokenPipeError, ConnectionError, OSError):
            self.close_connection = True

    def _read_required_body(self) -> bytes | None:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            self._send_error_json(
                HTTPStatus.LENGTH_REQUIRED,
                "length_required",
                "Content-Length is required",
                close=True,
            )
            return None
        try:
            length = int(content_length, 10)
        except ValueError:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "invalid_content_length",
                "Content-Length must be a non-negative integer",
                close=True,
            )
            return None
        if length < 0:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "invalid_content_length",
                "Content-Length must be a non-negative integer",
                close=True,
            )
            return None
        if length > self.server.max_request_bytes:
            self._send_error_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "frame_too_large",
                f"request body exceeds {self.server.max_request_bytes} bytes",
                close=True,
            )
            return None
        body = self.rfile.read(length)
        if len(body) != length:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "incomplete_body",
                "request body ended before Content-Length bytes were received",
                close=True,
            )
            return None
        return body

    def _consume_optional_empty_body(self) -> bool:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            return True
        try:
            length = int(content_length, 10)
        except ValueError:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST,
                "invalid_content_length",
                "Content-Length must be a non-negative integer",
                close=True,
            )
            return False
        if length < 0 or length > self.server.max_request_bytes:
            status = HTTPStatus.BAD_REQUEST if length < 0 else HTTPStatus.REQUEST_ENTITY_TOO_LARGE
            self._send_error_json(
                status,
                "invalid_content_length" if length < 0 else "body_too_large",
                "invalid reset request body length",
                close=True,
            )
            return False
        if length:
            body = self.rfile.read(length)
            if len(body) != length:
                self._send_error_json(
                    HTTPStatus.BAD_REQUEST,
                    "incomplete_body",
                    "request body ended before Content-Length bytes were received",
                    close=True,
                )
                return False
        return True

    def _send_json(
        self,
        status: HTTPStatus,
        payload: JsonObject,
        *,
        close: bool = False,
    ) -> None:
        body = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        self._send_bytes(
            status,
            body,
            "application/json; charset=utf-8",
            close=close,
        )

    def _send_error_json(
        self,
        status: HTTPStatus,
        code: str,
        message: str,
        *,
        close: bool = False,
    ) -> None:
        self._send_json(
            status,
            {"error": {"code": code, "message": message}},
            close=close,
        )

    def _send_bytes(
        self,
        status: HTTPStatus,
        body: bytes,
        content_type: str,
        *,
        close: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; "
            "connect-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        )
        self.send_header(
            "Permissions-Policy",
            "camera=(self), microphone=(), geolocation=()",
        )
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()


def create_mobile_server(
    config: RuntimeConfig,
    host: str = "0.0.0.0",
    port: int = 8443,
    *,
    certfile: str | PathLike[str] | None = None,
    keyfile: str | PathLike[str] | None = None,
    detector: Any | None = None,
    cv2_module: Any | None = None,
    distance_fusion: DistanceFusion | None = None,
    static_package: str = DEFAULT_STATIC_PACKAGE,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> MobileHTTPServer:
    """Create a configured HTTP or HTTPS server without starting its loop.

    Passing both ``certfile`` and ``keyfile`` enables TLS.  They must be passed
    together; the CLI can therefore make HTTPS the default while tests and
    explicit desktop-only runs use plain HTTP.
    """

    if (certfile is None) != (keyfile is None):
        raise ValueError("certfile and keyfile must be provided together")
    if not isinstance(static_package, str) or not static_package.strip():
        raise ValueError("static_package cannot be blank")
    if (
        isinstance(max_request_bytes, bool)
        or not isinstance(max_request_bytes, int)
        or max_request_bytes <= 0
    ):
        raise ValueError("max_request_bytes must be a positive integer")

    pipeline = MobilePipeline(
        config,
        detector=detector,
        cv2_module=cv2_module,
        distance_fusion=distance_fusion,
    )
    server = MobileHTTPServer((host, port), MobileRequestHandler)
    server.pipeline = pipeline
    server.static_package = static_package.strip()
    server.max_request_bytes = max_request_bytes

    if certfile is not None and keyfile is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            context.load_cert_chain(certfile=certfile, keyfile=keyfile)
            server.socket = context.wrap_socket(server.socket, server_side=True)
        except Exception:
            server.server_close()
            raise
    return server


def run_mobile_server(
    config: RuntimeConfig,
    host: str = "0.0.0.0",
    port: int = 8443,
    *,
    certfile: str | PathLike[str] | None = None,
    keyfile: str | PathLike[str] | None = None,
    detector: Any | None = None,
    cv2_module: Any | None = None,
    distance_fusion: DistanceFusion | None = None,
    static_package: str = DEFAULT_STATIC_PACKAGE,
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> None:
    """Run the mobile service until interrupted, then close its listening socket."""

    server = create_mobile_server(
        config,
        host,
        port,
        certfile=certfile,
        keyfile=keyfile,
        detector=detector,
        cv2_module=cv2_module,
        distance_fusion=distance_fusion,
        static_package=static_package,
        max_request_bytes=max_request_bytes,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _build_detector(config: RuntimeConfig) -> YoloDetector:
    return YoloDetector(
        config.model,
        confidence=config.confidence,
        device=config.device,
        image_size=config.image_size,
    )


def _build_temporal_filter(config: RuntimeConfig) -> TemporalFilter:
    return TemporalFilter(
        TemporalFilterConfig(
            confirmation_window=config.confirmation_window,
            minimum_confirmations=config.minimum_confirmations,
            high_risk_area_ratio=config.high_risk_area_ratio,
            high_risk_consecutive_frames=config.high_risk_consecutive_frames,
            max_missing_frames=config.max_missing_frames,
            direction_hysteresis=config.direction_hysteresis,
        )
    )


def _build_risk_engine(config: RuntimeConfig) -> RiskEngine:
    return RiskEngine(
        RiskRules(
            minimum_confidence=config.confidence,
            default_area_thresholds=AreaThresholds(high=config.high_risk_area_ratio),
        ),
        distance_rules=DistanceRiskRules(
            high_distance_m=config.distance_high_meters,
            medium_distance_m=config.distance_medium_meters,
            hysteresis_m=config.distance_hysteresis_meters,
            enter_frames=config.distance_enter_confirmations,
            exit_frames=config.distance_exit_confirmations,
        ),
    )


def _build_distance_fusion(config: RuntimeConfig) -> DistanceFusion:
    return DistanceFusion(
        maximum_distance_m=config.distance_max_meters,
        minimum_samples=config.distance_min_samples,
        minimum_coverage=config.distance_min_valid_ratio,
        history_size=config.distance_history_size,
    )


def _load_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError("OpenCV is not installed; run `uv sync` first") from error
    return cv2


def _parse_json_object(body: bytes) -> JsonObject:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RGBDProtocolError("invalid_json", "request body must be UTF-8 JSON") from error

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> JsonObject:
        result: JsonObject = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise RGBDProtocolError("invalid_json", "request body must be a JSON object") from error
    if not isinstance(payload, dict):
        raise RGBDProtocolError("invalid_json", "request body must be a JSON object")
    return payload


def _urgency_order(assessments: tuple[Any, ...]) -> tuple[int, ...]:
    return tuple(
        sorted(
            range(len(assessments)),
            key=lambda index: (
                assessments[index].rank,
                assessments[index].direction == "center",
                getattr(assessments[index], "distance_valid", False),
                -getattr(assessments[index], "distance_m", float("inf"))
                if getattr(assessments[index], "distance_m", None) is not None
                else float("-inf"),
                assessments[index].area_ratio,
                assessments[index].confidence,
            ),
            reverse=True,
        )
    )


def _detection_to_json(detection: Detection) -> JsonObject:
    return {
        "label": detection.label,
        "confidence": detection.confidence,
        "bbox": list(detection.bbox),
        "frame_width": detection.frame_width,
        "frame_height": detection.frame_height,
    }


def _assessment_to_json(stable: Any, assessment: Any) -> JsonObject:
    detection = stable.detection
    return {
        "track_key": stable.track_key,
        "label": assessment.label,
        "confidence": assessment.confidence,
        "bbox": list(detection.bbox),
        "frame_width": detection.frame_width,
        "frame_height": detection.frame_height,
        "direction": assessment.direction,
        "level": assessment.level,
        "should_alert": assessment.should_alert,
        "message": assessment.message,
        "area_ratio": assessment.area_ratio,
        "confirmed_frames": stable.confirmed_frames,
        "missed_frames": stable.missed_frames,
    }


def _fused_assessment_to_json(
    fused: FusedDetection,
    assessment: Any,
) -> JsonObject:
    payload = _assessment_to_json(fused.stable, assessment)
    estimate = fused.distance
    payload["distance"] = {
        "meters": estimate.distance_m,
        "usable": estimate.usable,
        "status": estimate.status,
        "depth_source": estimate.depth_source,
        "quality": estimate.quality,
        "freshness_frames": estimate.freshness_frames,
        "candidate_status": estimate.candidate_status,
        "is_stale": estimate.is_stale,
        "sample_count": estimate.sample_count,
        "coverage_ratio": estimate.coverage_ratio,
        "percentile_20_m": estimate.percentile_20_m,
        "candidate_median_m": estimate.candidate_median_m,
        "candidate_center_median_m": estimate.candidate_center_median_m,
        "candidate_p20_meters": estimate.percentile_20_m,
        "candidate_median_meters": estimate.candidate_median_m,
        "candidate_center_median_meters": estimate.candidate_center_median_m,
        "missed_frames": estimate.missed_frames,
    }
    risk_source = getattr(assessment, "distance_source", None)
    if risk_source is not None:
        payload["risk_source"] = risk_source
    distance_status = getattr(assessment, "distance_status", None)
    if distance_status is not None:
        payload["distance_status"] = distance_status
    return payload


def _alert_to_json(alert: AlertDecision, track_key: str | None) -> JsonObject:
    assessment = alert.assessment
    return {
        "should_emit": alert.should_emit,
        "reason": alert.reason,
        "message": alert.message,
        "track_key": track_key,
        "label": assessment.label,
        "direction": assessment.direction,
        "level": assessment.level,
        "distance_meters": getattr(assessment, "distance_m", None),
        "distance_status": getattr(assessment, "distance_status", "unavailable"),
        "risk_source": getattr(assessment, "distance_source", "area_fallback"),
        "cooldown_remaining": alert.cooldown_remaining,
    }
