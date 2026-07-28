"""观察模式的输入隔离和标注辅助测试。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT
sys.path.insert(0, str(PROJECT_DIR))

from observe_game import (  # noqa: E402
    DisabledPyAutoGUI,
    annotate,
    detection_group,
)


class InputIsolationTests(unittest.TestCase):
    def test_keyboard_calls_are_blocked(self) -> None:
        automation = DisabledPyAutoGUI("pyautogui")
        with self.assertRaisesRegex(RuntimeError, "观察模式禁止"):
            automation.keyDown("left")

    def test_observer_does_not_use_opencv_highgui(self) -> None:
        source = (PROJECT_DIR / "observe_game.py").read_text(encoding="utf-8")
        self.assertNotIn("cv2.imshow", source)
        self.assertNotIn("cv2.destroyAllWindows", source)


class AnnotationTests(unittest.TestCase):
    def test_model_classes_receive_expected_groups(self) -> None:
        self.assertEqual(detection_group("character"), "player")
        self.assertEqual(detection_group("bullet_player"), "player_bullet")
        self.assertEqual(
            detection_group("bullet_enemy_small_red"),
            "enemy_bullet",
        )
        self.assertEqual(detection_group("boss"), "enemy")

    def test_annotation_preserves_frame_shape(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {
                "bbox": [100, 100, 120, 130],
                "class_name": "character",
                "confidence": 0.9,
            }
        ]
        output = annotate(
            frame,
            detections,
            "planned=move:left",
            {"left": 20, "top": 30, "width": 640, "height": 480},
        )
        self.assertEqual(output.shape, frame.shape)
        self.assertGreater(int(output.sum()), 0)


if __name__ == "__main__":
    unittest.main()
