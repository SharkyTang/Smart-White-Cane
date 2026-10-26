from __future__ import annotations

import unittest
from types import SimpleNamespace

from smart_cane.perception.camera import CameraReadError, CameraSource
from smart_cane.perception.detector import YoloDetector, resolve_device


class FakeCapture:
    def __init__(self, frames: list[object]) -> None:
        self.frames = list(frames)
        self.opened = True
        self.released = False

    def isOpened(self) -> bool:
        return self.opened and not self.released

    def read(self) -> tuple[bool, object | None]:
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class FakeCv2:
    def __init__(self, capture: FakeCapture) -> None:
        self.capture = capture
        self.arguments: tuple[object, ...] | None = None

    def VideoCapture(self, *arguments: object) -> FakeCapture:
        self.arguments = arguments
        return self.capture


class TensorValue:
    def __init__(self, value: object) -> None:
        self.value = value

    def detach(self) -> TensorValue:
        return self

    def cpu(self) -> TensorValue:
        return self

    def tolist(self) -> object:
        return self.value


class FakeBox:
    xyxy = TensorValue([[10, 20, 110, 220]])
    conf = TensorValue([0.91])
    cls = TensorValue([0])


class FakeResult:
    orig_shape = (480, 640)
    names = {0: "chair"}
    boxes = [FakeBox()]


class FakeModel:
    names = {0: "chair"}

    def __init__(self) -> None:
        self.options: dict[str, object] | None = None
        self.calls = 0

    def predict(self, **options: object) -> list[FakeResult]:
        self.calls += 1
        self.options = options
        return [FakeResult()]


class CameraSourceTests(unittest.TestCase):
    def test_reads_and_releases_in_context_manager(self) -> None:
        frame = object()
        capture = FakeCapture([frame])
        cv2 = FakeCv2(capture)

        with CameraSource(0, backend=1200, cv2_module=cv2) as source:
            self.assertIs(source.read(), frame)

        self.assertTrue(capture.released)
        self.assertEqual(cv2.arguments, (0, 1200))

    def test_reports_end_of_stream(self) -> None:
        source = CameraSource("clip.mp4", cv2_module=FakeCv2(FakeCapture([])))
        with self.assertRaises(CameraReadError):
            source.read()


class YoloDetectorDefaultTests(unittest.TestCase):
    def test_uses_current_yolo26_nano_model(self) -> None:
        detector = YoloDetector(model=FakeModel())

        self.assertEqual(detector.model_name, "yolo26n.pt")


class YoloDetectorTests(unittest.TestCase):
    def test_translates_model_result_into_domain_detection(self) -> None:
        model = FakeModel()
        detector = YoloDetector("fake.pt", confidence=0.5, device="cpu", model=model)

        detections = detector.detect(object())

        self.assertEqual(len(detections), 1)
        detection = detections[0]
        self.assertEqual(detection.label, "chair")
        self.assertEqual(detection.bbox, (10.0, 20.0, 110.0, 220.0))
        self.assertEqual((detection.frame_width, detection.frame_height), (640, 480))
        assert model.options is not None
        self.assertEqual(model.options["conf"], 0.5)
        self.assertEqual(model.options["device"], "cpu")
        self.assertEqual(model.options["imgsz"], 640)

    def test_warmup_runs_one_prediction_and_returns_detections(self) -> None:
        model = FakeModel()
        detector = YoloDetector("fake.pt", model=model, image_size=320)

        detections = detector.warmup(object())

        self.assertEqual(model.calls, 1)
        self.assertEqual(detections[0].label, "chair")
        assert model.options is not None
        self.assertEqual(model.options["imgsz"], 320)

    def test_mps_request_falls_back_to_cpu_when_unavailable(self) -> None:
        unavailable = SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: False),
            )
        )
        available = SimpleNamespace(
            backends=SimpleNamespace(
                mps=SimpleNamespace(is_available=lambda: True),
            )
        )

        self.assertEqual(resolve_device(None), "cpu")
        self.assertEqual(resolve_device("auto"), "cpu")
        self.assertEqual(resolve_device(0), 0)
        self.assertEqual(resolve_device("0"), "0")
        self.assertEqual(resolve_device("mps", torch_module=unavailable), "cpu")
        self.assertEqual(resolve_device("mps", torch_module=available), "mps")

    def test_detector_rejects_boolean_confidence_and_invalid_device_types(self) -> None:
        with self.assertRaises(ValueError):
            YoloDetector("fake.pt", confidence=True)
        with self.assertRaises(TypeError):
            resolve_device(True)


if __name__ == "__main__":
    unittest.main()
