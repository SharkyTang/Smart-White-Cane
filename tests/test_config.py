import unittest

from smart_cane.config import RuntimeConfig, parse_source


class RuntimeConfigTests(unittest.TestCase):
    def test_numeric_source_becomes_camera_index(self) -> None:
        self.assertEqual(parse_source("0"), 0)

    def test_file_source_stays_a_string(self) -> None:
        self.assertIsInstance(parse_source("sample.mp4"), str)

    def test_invalid_confidence_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RuntimeConfig(confidence=1.1)


if __name__ == "__main__":
    unittest.main()
