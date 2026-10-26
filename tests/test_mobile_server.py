from __future__ import annotations

import json
import struct
import tempfile
import threading
import time
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest import mock

import numpy as np

from smart_cane.config import RuntimeConfig
from smart_cane.feedback.alert_policy import AlertPolicy
from smart_cane.mobile_server import MobilePipeline, create_mobile_server
from smart_cane.models import Detection
from smart_cane.rgbd_protocol import parse_rgbd_request


JPEG_BYTES = b"\xff\xd8fake-jpeg\xff\xd9"
RGBD_CONTENT_TYPE = "application/vnd.smart-cane.rgbd-frame;version=1"


def rgbd_body(
    session_id: str,
    frame_id: int,
    *,
    rgb_width: int = 100,
    rgb_height: int = 100,
    depth_width: int = 20,
    depth_height: int = 20,
    distance_m: float = 0.8,
) -> bytes:
    metadata = json.dumps(
        {
            "schema_version": 1,
            "session_id": session_id,
            "frame_id": frame_id,
            "timestamp_seconds": 12.5,
            "rgb_width": rgb_width,
            "rgb_height": rgb_height,
            "depth_width": depth_width,
            "depth_height": depth_height,
            "source": "arkit_scene_depth",
            "depth_source": "raw_scene_depth",
            "orientation": "portrait_up",
            "scenario_id": None,
        },
        separators=(",", ":"),
    ).encode()
    depth = np.full((depth_height, depth_width), distance_m, dtype="<f2").tobytes()
    confidence = np.full((depth_height, depth_width), 2, dtype=np.uint8).tobytes()
    header = struct.pack(
        ">4sIIII",
        b"SCF1",
        len(metadata),
        len(JPEG_BYTES),
        len(depth),
        len(confidence),
    )
    return header + metadata + JPEG_BYTES + depth + confidence


class FakeCv2:
    IMREAD_COLOR = 1

    def __init__(self, frame: object | None = None) -> None:
        self.frame = object() if frame is None else frame
        self.decoded: list[bytes] = []

    def imdecode(self, encoded: object, mode: int) -> object | None:
        self.decoded.append(bytes(encoded))
        return self.frame


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, frame: object) -> list[Detection]:
        self.calls += 1
        return [Detection("chair", 0.9, (10, 10, 90, 90), 100, 100)]


class ConcurrencyDetector(FakeDetector):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()

    def __call__(self, frame: object) -> list[Detection]:
        with self.lock:
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        time.sleep(0.03)
        try:
            return super().__call__(frame)
        finally:
            with self.lock:
                self.active -= 1


class RunningServer:
    def __init__(self, **options: object) -> None:
        config = RuntimeConfig(
            display=False,
            confirmation_window=1,
            minimum_confirmations=1,
        )
        self.server = create_mobile_server(
            config,
            "127.0.0.1",
            0,
            detector=options.pop("detector", FakeDetector()),
            cv2_module=options.pop("cv2_module", FakeCv2()),
            **options,
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def __enter__(self) -> RunningServer:
        self.thread.start()
        return self

    def __exit__(self, *args: object) -> bool:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        return False

    def request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        connection = HTTPConnection("127.0.0.1", self.port, timeout=2.0)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            response_body = response.read()
            return response.status, dict(response.getheaders()), response_body
        finally:
            connection.close()


class MobileServerTests(unittest.TestCase):
    def test_mobile_pipeline_uses_runtime_distance_configuration(self) -> None:
        config = RuntimeConfig(
            display=False,
            distance_high_meters=0.7,
            distance_medium_meters=1.7,
            distance_hysteresis_meters=0.1,
            distance_max_meters=4.0,
            distance_min_samples=8,
            distance_min_valid_ratio=0.3,
            distance_history_size=4,
            distance_enter_confirmations=3,
            distance_exit_confirmations=4,
        )
        pipeline = MobilePipeline(
            config,
            detector=FakeDetector(),
            cv2_module=FakeCv2(),
        )

        self.assertEqual(pipeline.distance_fusion.maximum_distance_m, 4.0)
        self.assertEqual(pipeline.distance_fusion.minimum_samples, 8)
        self.assertEqual(pipeline.distance_fusion.minimum_coverage, 0.3)
        self.assertEqual(pipeline.distance_fusion.history_size, 4)
        rules = pipeline.risk_engine.distance_rules
        self.assertEqual(rules.high_distance_m, 0.7)
        self.assertEqual(rules.medium_distance_m, 1.7)
        self.assertEqual(rules.hysteresis_m, 0.1)
        self.assertEqual(rules.enter_frames, 3)
        self.assertEqual(rules.exit_frames, 4)

    def test_health_and_static_files_are_same_origin_and_never_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "index.html").write_text("<main>phone</main>", encoding="utf-8")
            (root / "app.js").write_text("const ready = true;", encoding="utf-8")
            (root / "styles.css").write_text("main { color: white; }", encoding="utf-8")

            with (
                mock.patch("smart_cane.mobile_server.resources.files", return_value=root),
                RunningServer(static_package="test.mobile") as running,
            ):
                status, headers, body = running.request("GET", "/healthz")
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body), {"status": "ok"})
                self.assertEqual(headers["Cache-Control"], "no-store")
                self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
                self.assertEqual(headers["Referrer-Policy"], "no-referrer")
                self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
                self.assertEqual(
                    headers["Permissions-Policy"],
                    "camera=(self), microphone=(), geolocation=()",
                )

                status, headers, body = running.request("GET", "/app.js?v=1")
                self.assertEqual(status, 200)
                self.assertEqual(body, b"const ready = true;")
                self.assertEqual(headers["Content-Type"], "application/javascript; charset=utf-8")

                status, _, _ = running.request("GET", "/../pyproject.toml")
                self.assertEqual(status, 404)

    def test_real_packaged_mobile_page_is_served(self) -> None:
        with RunningServer() as running:
            page_status, _, page = running.request("GET", "/")
            script_status, _, script = running.request("GET", "/app.js")

        self.assertEqual((page_status, script_status), (200, 200))
        self.assertIn("障碍识别提醒".encode(), page)
        self.assertIn(b"getUserMedia", script)

    def test_frame_response_contains_detection_decision_alert_and_timing(self) -> None:
        detector = FakeDetector()
        cv2 = FakeCv2()
        with RunningServer(detector=detector, cv2_module=cv2) as running:
            status, _, body = running.request(
                "POST",
                "/api/frame",
                body=JPEG_BYTES,
                headers={"Content-Type": "image/jpeg"},
            )

        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["detections"][0]["label"], "chair")
        self.assertEqual(payload["detections"][0]["bbox"], [10.0, 10.0, 90.0, 90.0])
        self.assertEqual(payload["assessments"][0]["track_key"], "chair:1")
        self.assertEqual(payload["assessments"][0]["level"], "high")
        self.assertTrue(payload["alert"]["should_emit"])
        self.assertEqual(payload["alert"]["reason"], "first_alert")
        self.assertEqual(
            set(payload["timing"]),
            {"decode_ms", "inference_ms", "decision_ms", "total_ms"},
        )
        self.assertEqual(cv2.decoded, [JPEG_BYTES])
        self.assertEqual(detector.calls, 1)

    def test_alert_commits_only_after_delivery_succeeds(self) -> None:
        policy = AlertPolicy(cooldown_seconds=30.0)
        pipeline = MobilePipeline(
            RuntimeConfig(
                display=False,
                confirmation_window=1,
                minimum_confirmations=1,
            ),
            detector=FakeDetector(),
            cv2_module=FakeCv2(),
            alert_policy=policy,
        )

        def fail_delivery(payload: dict[str, object]) -> None:
            raise BrokenPipeError("phone disconnected")

        with self.assertRaises(BrokenPipeError):
            pipeline.deliver_frame(JPEG_BYTES, fail_delivery)

        retry_payload = pipeline.deliver_frame(JPEG_BYTES, lambda payload: None)
        cooling_payload = pipeline.deliver_frame(JPEG_BYTES, lambda payload: None)

        self.assertTrue(retry_payload["alert"]["should_emit"])
        self.assertEqual(retry_payload["alert"]["reason"], "first_alert")
        self.assertFalse(cooling_payload["alert"]["should_emit"])
        self.assertEqual(cooling_payload["alert"]["reason"], "cooldown_active")

    def test_session_reset_clears_temporal_and_alert_history(self) -> None:
        with RunningServer() as running:
            first_status, _, first_body = running.request(
                "POST",
                "/api/frame",
                body=JPEG_BYTES,
                headers={"Content-Type": "image/jpeg"},
            )
            second_status, _, second_body = running.request(
                "POST",
                "/api/frame",
                body=JPEG_BYTES,
                headers={"Content-Type": "image/jpeg"},
            )
            reset_status, _, reset_body = running.request("POST", "/api/session/reset")
            third_status, _, third_body = running.request(
                "POST",
                "/api/frame",
                body=JPEG_BYTES,
                headers={"Content-Type": "image/jpeg"},
            )

        self.assertEqual((first_status, second_status, reset_status, third_status), (200,) * 4)
        self.assertTrue(json.loads(first_body)["alert"]["should_emit"])
        self.assertFalse(json.loads(second_body)["alert"]["should_emit"])
        self.assertTrue(json.loads(reset_body)["reset"])
        self.assertTrue(json.loads(third_body)["alert"]["should_emit"])
        self.assertEqual(json.loads(third_body)["alert"]["reason"], "first_alert")

    def test_frame_endpoint_enforces_content_type_size_and_decodability(self) -> None:
        detector = FakeDetector()
        with RunningServer(detector=detector, max_request_bytes=8) as running:
            wrong_type, _, _ = running.request(
                "POST",
                "/api/frame",
                body=b"jpeg",
                headers={"Content-Type": "application/octet-stream"},
            )
            too_large, _, body = running.request(
                "POST",
                "/api/frame",
                body=b"123456789",
                headers={"Content-Type": "image/jpeg"},
            )

        undecodable_cv2 = FakeCv2()
        undecodable_cv2.frame = None
        with RunningServer(cv2_module=undecodable_cv2) as running:
            invalid_image, _, _ = running.request(
                "POST",
                "/api/frame",
                body=b"not-jpeg",
                headers={"Content-Type": "image/jpeg"},
            )

        self.assertEqual(wrong_type, 415)
        self.assertEqual(too_large, 413)
        self.assertEqual(json.loads(body)["error"]["code"], "frame_too_large")
        self.assertEqual(invalid_image, 422)
        self.assertEqual(detector.calls, 0)

    def test_cross_origin_browser_write_is_rejected(self) -> None:
        detector = FakeDetector()
        with RunningServer(detector=detector) as running:
            status, _, body = running.request(
                "POST",
                "/api/frame",
                body=JPEG_BYTES,
                headers={
                    "Content-Type": "image/jpeg",
                    "Origin": "https://example.invalid",
                },
            )

        self.assertEqual(status, 403)
        self.assertEqual(json.loads(body)["error"]["code"], "origin_not_allowed")
        self.assertEqual(detector.calls, 0)

    def test_pipeline_serializes_concurrent_detector_calls(self) -> None:
        detector = ConcurrencyDetector()
        pipeline = MobilePipeline(
            RuntimeConfig(
                display=False,
                confirmation_window=1,
                minimum_confirmations=1,
            ),
            detector=detector,
            cv2_module=FakeCv2(),
        )
        barrier = threading.Barrier(3)

        def process() -> None:
            barrier.wait()
            pipeline.deliver_frame(JPEG_BYTES, lambda payload: None)

        threads = [threading.Thread(target=process) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2.0)

        self.assertEqual(detector.calls, 2)
        self.assertEqual(detector.maximum_active, 1)

    def test_rgbd_pipeline_ack_is_delayed_stable_and_idempotent(self) -> None:
        pipeline = MobilePipeline(
            RuntimeConfig(
                display=False,
                confirmation_window=1,
                minimum_confirmations=1,
            ),
            detector=FakeDetector(),
            cv2_module=FakeCv2(np.zeros((100, 100, 3), dtype=np.uint8)),
        )
        session_id = pipeline.reset_session()["session_id"]
        payloads = [
            pipeline.deliver_rgbd_frame(
                parse_rgbd_request(rgbd_body(session_id, frame_id)),
                lambda payload: None,
            )
            for frame_id in range(3)
        ]

        first_alert = payloads[1]["alert"]
        self.assertEqual(payloads[1]["model_name"], "yolo26n.pt")
        retry_alert = payloads[2]["alert"]
        self.assertEqual(first_alert["alert_id"], retry_alert["alert_id"])
        self.assertEqual(first_alert["alert_frame_id"], 1)
        self.assertEqual(retry_alert["alert_frame_id"], 1)
        self.assertAlmostEqual(first_alert["distance_meters"], 0.8, places=2)
        self.assertEqual(first_alert["distance_status"], "valid")
        self.assertEqual(first_alert["risk_source"], "lidar")

        first_ack = pipeline.acknowledge_alert(
            session_id,
            first_alert["alert_frame_id"],
            first_alert["alert_id"],
        )
        repeated_ack = pipeline.acknowledge_alert(
            session_id,
            first_alert["alert_frame_id"],
            first_alert["alert_id"],
        )
        self.assertFalse(first_ack["already_acknowledged"])
        self.assertTrue(repeated_ack["already_acknowledged"])

        cooling = pipeline.deliver_rgbd_frame(
            parse_rgbd_request(rgbd_body(session_id, 3)),
            lambda payload: None,
        )
        self.assertFalse(cooling["alert"]["should_emit"])

        with self.assertRaisesRegex(ValueError, "greater than"):
            pipeline.deliver_rgbd_frame(
                parse_rgbd_request(rgbd_body(session_id, 3)),
                lambda payload: None,
            )

    def test_rgbd_pipeline_keeps_multiple_unacknowledged_alerts(self) -> None:
        pipeline = MobilePipeline(
            RuntimeConfig(
                display=False,
                confirmation_window=1,
                minimum_confirmations=1,
            ),
            detector=FakeDetector(),
            cv2_module=FakeCv2(np.zeros((100, 100, 3), dtype=np.uint8)),
        )
        session_id = pipeline.reset_session()["session_id"]

        frames = []
        for frame_id, distance_m in enumerate((0.8, 0.8, 1.5, 1.5, 1.5, 1.5)):
            frames.append(
                pipeline.deliver_rgbd_frame(
                    parse_rgbd_request(rgbd_body(session_id, frame_id, distance_m=distance_m)),
                    lambda payload: None,
                )
            )

        high_alert = frames[1]["alert"]
        self.assertEqual(frames[2]["alert"]["alert_id"], high_alert["alert_id"])
        medium_alert = frames[5]["alert"]
        self.assertNotEqual(high_alert["alert_id"], medium_alert["alert_id"])

        high_ack = pipeline.acknowledge_alert(
            session_id,
            high_alert["alert_frame_id"],
            high_alert["alert_id"],
        )
        medium_ack = pipeline.acknowledge_alert(
            session_id,
            medium_alert["alert_frame_id"],
            medium_alert["alert_id"],
        )
        self.assertTrue(high_ack["acknowledged"])
        self.assertTrue(medium_ack["acknowledged"])

        cooling = pipeline.deliver_rgbd_frame(
            parse_rgbd_request(rgbd_body(session_id, 6, distance_m=1.5)),
            lambda payload: None,
        )
        self.assertFalse(cooling["alert"]["should_emit"])

    def test_rgbd_alert_waits_for_ack_and_retries_with_stable_id(self) -> None:
        cv2 = FakeCv2(np.zeros((100, 100, 3), dtype=np.uint8))
        with RunningServer(cv2_module=cv2) as running:
            reset_status, _, reset_body = running.request("POST", "/api/session/reset")
            session_id = json.loads(reset_body)["session_id"]

            responses = []
            for frame_id in range(3):
                status, _, body = running.request(
                    "POST",
                    "/api/rgbd-frame",
                    body=rgbd_body(session_id, frame_id),
                    headers={"Content-Type": RGBD_CONTENT_TYPE},
                )
                responses.append((status, json.loads(body)))

            first_alert = responses[1][1]["alert"]
            retry_alert = responses[2][1]["alert"]
            ack_body = json.dumps(
                {
                    "session_id": session_id,
                    "frame_id": first_alert["alert_frame_id"],
                    "alert_id": first_alert["alert_id"],
                }
            ).encode()
            ack_status, _, ack_response = running.request(
                "POST",
                "/api/alert/ack",
                body=ack_body,
                headers={"Content-Type": "application/json"},
            )
            repeated_status, _, repeated_response = running.request(
                "POST",
                "/api/alert/ack",
                body=ack_body,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            cooling_status, _, cooling_body = running.request(
                "POST",
                "/api/rgbd-frame",
                body=rgbd_body(session_id, 3),
                headers={"Content-Type": RGBD_CONTENT_TYPE},
            )

        self.assertEqual(reset_status, 200)
        self.assertEqual([status for status, _ in responses], [200, 200, 200])
        self.assertEqual(responses[2][1]["session_id"], session_id)
        self.assertEqual(responses[2][1]["frame_id"], 2)
        distance = responses[1][1]["assessments"][0]["distance"]
        self.assertAlmostEqual(distance["meters"], 0.8, places=2)
        self.assertTrue(distance["usable"])
        self.assertEqual(distance["status"], "valid")
        self.assertEqual(first_alert["alert_id"], retry_alert["alert_id"])
        self.assertEqual(first_alert["alert_frame_id"], 1)
        self.assertEqual(retry_alert["alert_frame_id"], 1)
        self.assertEqual(ack_status, 200)
        self.assertFalse(json.loads(ack_response)["already_acknowledged"])
        self.assertEqual(repeated_status, 200)
        self.assertTrue(json.loads(repeated_response)["already_acknowledged"])
        self.assertEqual(cooling_status, 200)
        self.assertFalse(json.loads(cooling_body)["alert"]["should_emit"])
        self.assertNotIn("alert_id", json.loads(cooling_body)["alert"])

    def test_rgbd_rejects_stale_sessions_and_non_increasing_frames(self) -> None:
        cv2 = FakeCv2(np.zeros((100, 100, 3), dtype=np.uint8))
        with RunningServer(cv2_module=cv2) as running:
            _, _, first_reset = running.request("POST", "/api/session/reset")
            first_session = json.loads(first_reset)["session_id"]
            first_status, _, _ = running.request(
                "POST",
                "/api/rgbd-frame",
                body=rgbd_body(first_session, 5),
                headers={"Content-Type": RGBD_CONTENT_TYPE},
            )
            stale_frame_status, _, stale_frame_body = running.request(
                "POST",
                "/api/rgbd-frame",
                body=rgbd_body(first_session, 5),
                headers={"Content-Type": RGBD_CONTENT_TYPE},
            )
            _, _, second_reset = running.request("POST", "/api/session/reset")
            second_session = json.loads(second_reset)["session_id"]
            stale_session_status, _, stale_session_body = running.request(
                "POST",
                "/api/rgbd-frame",
                body=rgbd_body(first_session, 6),
                headers={"Content-Type": RGBD_CONTENT_TYPE},
            )

        self.assertEqual(first_status, 200)
        self.assertNotEqual(first_session, second_session)
        self.assertEqual(stale_frame_status, 409)
        self.assertEqual(json.loads(stale_frame_body)["error"]["code"], "stale_frame")
        self.assertEqual(stale_session_status, 409)
        self.assertEqual(json.loads(stale_session_body)["error"]["code"], "stale_session")

    def test_rgbd_endpoint_enforces_media_type_protocol_and_jpeg_dimensions(self) -> None:
        matching_cv2 = FakeCv2(np.zeros((100, 100, 3), dtype=np.uint8))
        with RunningServer(cv2_module=matching_cv2) as running:
            _, _, reset_body = running.request("POST", "/api/session/reset")
            session_id = json.loads(reset_body)["session_id"]
            missing_version, _, _ = running.request(
                "POST",
                "/api/rgbd-frame",
                body=rgbd_body(session_id, 0),
                headers={"Content-Type": "application/vnd.smart-cane.rgbd-frame"},
            )
            malformed, _, malformed_body = running.request(
                "POST",
                "/api/rgbd-frame",
                body=b"not-scf1",
                headers={"Content-Type": RGBD_CONTENT_TYPE},
            )

        mismatching_cv2 = FakeCv2(np.zeros((99, 100, 3), dtype=np.uint8))
        with RunningServer(cv2_module=mismatching_cv2) as running:
            _, _, reset_body = running.request("POST", "/api/session/reset")
            session_id = json.loads(reset_body)["session_id"]
            dimensions, _, dimensions_body = running.request(
                "POST",
                "/api/rgbd-frame",
                body=rgbd_body(session_id, 0),
                headers={"Content-Type": RGBD_CONTENT_TYPE},
            )

        self.assertEqual(missing_version, 415)
        self.assertEqual(malformed, 400)
        self.assertEqual(json.loads(malformed_body)["error"]["code"], "invalid_header")
        self.assertEqual(dimensions, 422)
        self.assertEqual(json.loads(dimensions_body)["error"]["code"], "invalid_image")

    def test_tls_certificate_and_key_must_be_supplied_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "provided together"):
            create_mobile_server(
                RuntimeConfig(display=False),
                "127.0.0.1",
                0,
                certfile="certificate.pem",
            )


if __name__ == "__main__":
    unittest.main()
