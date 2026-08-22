# Architecture

The prototype separates domain logic from hardware and machine-learning libraries.

```text
CameraSource ──frame──> YoloDetector ──Detection[]──> RiskEngine
                                                        │
                                      RiskAssessment[]──┤
                                                        ↓
                                              AudioFeedback
```

## Boundary rules

- `models.py` and `decision/` use only the Python standard library.
- OpenCV is imported only by camera/display code.
- Ultralytics is imported only when a YOLO detector is instantiated.
- Audio feedback accepts an injected runner and clock so cooldown behavior is testable.
- Future LiDAR or ultrasonic adapters should produce distance observations instead of leaking GPIO or serial details into the decision layer.

## Provisional risk heuristic

The first prototype has no reliable physical distance measurement. It therefore uses bounding-box area and horizontal position as a transparent proxy. That heuristic is deliberately labelled provisional and must not be presented as real-world distance or safety.
