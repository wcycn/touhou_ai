"""统一GUI的无显示导入测试。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT
sys.path.insert(0, str(PROJECT_DIR))

from desktop_gui import (  # noqa: E402
    ROOT_LAUNCHER,
    ProcessManager,
    build_mode_arguments,
)


class DesktopGUIImportTests(unittest.TestCase):
    def test_gui_uses_unified_launcher(self) -> None:
        self.assertEqual(ROOT_LAUNCHER, ROOT / "touhou_ai.py")
        self.assertTrue(ROOT_LAUNCHER.is_file())

    def test_process_manager_can_be_constructed_without_display(self) -> None:
        import queue

        manager = ProcessManager(queue.Queue())
        self.assertFalse(manager.is_running("ai"))

    def test_observe_and_ai_share_parameters_without_duplicate_options(self) -> None:
        common = {
            "mode": "balanced",
            "confidence": 0.35,
            "sensitivity": 0.05,
            "safe_margin": 36,
            "player_lost_timeout": 0.7,
            "allow_vertical": True,
            "recording_arguments": ["--record"],
        }
        observe = build_mode_arguments("observe", **common)
        ai = build_mode_arguments("ai", **common)
        self.assertEqual(observe[1:], ai[1:])
        self.assertEqual(ai.count("--confidence"), 1)
        self.assertEqual(observe.count("--confidence"), 1)


if __name__ == "__main__":
    unittest.main()
