import unittest

from smart_cane.config import DEFAULT_MODEL, RuntimeConfig, parse_source


class RuntimeConfigTests(unittest.TestCase):
    def test_numeric_source_becomes_camera_index(self) -> None:
        self.assertEqual(parse_source("0"), 0)

    def test_file_source_stays_a_string(self) -> None:
        self.assertIsInstance(parse_source("sample.mp4"), str)

    def test_invalid_confidence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeConfig(confidence=1.1)

    def test_default_model_is_yolo26_nano(self) -> None:
        self.assertEqual(DEFAULT_MODEL, "yolo26n.pt")
        self.assertEqual(RuntimeConfig().model, DEFAULT_MODEL)

    def test_milestone_02_defaults_match_the_stability_plan(self) -> None:
        config = RuntimeConfig()

        self.assertEqual(config.image_size, 640)
        self.assertEqual(config.warmup_frames, 5)
        self.assertEqual((config.minimum_confirmations, config.confirmation_window), (3, 5))
        self.assertEqual(config.high_risk_area_ratio, 0.18)
        self.assertEqual(config.high_risk_consecutive_frames, 2)
        self.assertEqual(config.max_missing_frames, 3)
        self.assertEqual(config.direction_hysteresis, 0.05)
        self.assertEqual(config.alert_cooldown_seconds, 3.0)

    def test_lidar_defaults_and_validation(self) -> None:
        config = RuntimeConfig()

        self.assertEqual(config.distance_high_meters, 1.0)
        self.assertEqual(config.distance_medium_meters, 2.0)
        self.assertEqual(config.distance_hysteresis_meters, 0.15)
        self.assertEqual(config.distance_max_meters, 5.0)
        self.assertEqual(config.distance_min_samples, 12)
        self.assertEqual(config.distance_min_valid_ratio, 0.20)

        invalid_configs = (
            {"distance_high_meters": 0.0},
            {"distance_high_meters": 2.0, "distance_medium_meters": 1.0},
            {"distance_medium_meters": 5.0},
            {"distance_hysteresis_meters": 4.0},
            {"distance_min_samples": 0},
            {"distance_min_valid_ratio": 0.0},
            {"distance_min_valid_ratio": 1.1},
            {"distance_enter_confirmations": True},
        )
        for values in invalid_configs:
            with self.subTest(values=values), self.assertRaises(ValueError):
                RuntimeConfig(**values)

    def test_invalid_temporal_and_performance_settings_are_rejected(self) -> None:
        invalid_settings = (
            {"image_size": 0},
            {"warmup_frames": -1},
            {"minimum_confirmations": 4, "confirmation_window": 3},
            {"high_risk_area_ratio": 0.06},
            {"high_risk_area_ratio": 1.1},
            {"high_risk_consecutive_frames": 0},
            {"max_missing_frames": 0},
            {"direction_hysteresis": 0.15},
            {"alert_cooldown_seconds": -0.1},
        )

        for settings in invalid_settings:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                RuntimeConfig(**settings)

    def test_legacy_audio_cooldown_keyword_maps_to_alert_policy(self) -> None:
        config = RuntimeConfig(audio_cooldown_seconds=2.5)

        self.assertEqual(config.alert_cooldown_seconds, 2.5)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            RuntimeConfig(
                alert_cooldown_seconds=1.0,
                audio_cooldown_seconds=2.5,
            )

    def test_numeric_detector_device_is_supported(self) -> None:
        self.assertEqual(RuntimeConfig(device=0).device, 0)
        with self.assertRaises(ValueError):
            RuntimeConfig(device=-1)


if __name__ == "__main__":
    unittest.main()
