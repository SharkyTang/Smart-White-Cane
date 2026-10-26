from __future__ import annotations

import unittest
from importlib import resources


class MobileAssetTests(unittest.TestCase):
    def test_packaged_page_uses_iphone_camera_and_external_local_assets(self) -> None:
        root = resources.files("smart_cane.mobile")
        html = root.joinpath("index.html").read_text(encoding="utf-8")

        self.assertIn("playsinline", html)
        self.assertIn('src="app.js"', html)
        self.assertIn('href="styles.css"', html)
        self.assertNotIn("user-scalable=no", html)
        self.assertNotIn("https://", html)

    def test_mobile_loop_is_serial_audio_first_and_does_not_request_vibration(self) -> None:
        script = resources.files("smart_cane.mobile").joinpath("app.js").read_text(encoding="utf-8")

        self.assertIn("getUserMedia", script)
        self.assertIn('facing:"environment"', script)
        self.assertIn("await postFrame(blob)", script)
        self.assertIn("SpeechSynthesisUtterance", script)
        self.assertIn("audiocontext", script.casefold())
        self.assertNotIn("navigator.vibrate", script)


if __name__ == "__main__":
    unittest.main()
