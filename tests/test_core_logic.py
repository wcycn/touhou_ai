"""不连接游戏、不发送按键的核心回归测试。"""

from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT
sys.path.insert(0, str(PROJECT_DIR))

# 核心模块导入 PyAutoGUI 时会立即连接 X11；逻辑测试不需要真实键盘。
sys.modules["pyautogui"] = types.ModuleType("pyautogui")

from autopilot import (  # noqa: E402
    TouhouAIController,
    WINDOW_CONTROLLER_AVAILABLE,
)


class AnalyzeGameStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ai = TouhouAIController.__new__(
            TouhouAIController
        )
        self.ai.screen_region = {"width": 640, "height": 480}

    def test_model_classes_are_mapped_to_correct_groups(self) -> None:
        detections = [
            {"class_name": "character", "center_x": 300, "center_y": 400},
            {
                "class_name": "bullet_enemy_small_red",
                "center_x": 310,
                "center_y": 405,
            },
            {"class_name": "bullet_player", "center_x": 300, "center_y": 300},
            {"class_name": "boss", "center_x": 300, "center_y": 80},
            {"class_name": "powerup_red", "center_x": 250, "center_y": 200},
        ]

        state = self.ai.analyze_game_state(detections)

        self.assertEqual(state["player"]["class_name"], "character")
        self.assertEqual(len(state["bullets"]), 1)
        self.assertEqual(len(state["player_bullets"]), 1)
        self.assertEqual(len(state["enemies"]), 1)
        self.assertEqual(len(state["powerups"]), 1)

    def test_detection_passes_selected_device_and_user_confidence(self) -> None:
        class FakeModel:
            names = {}

            def __init__(self):
                self.calls = []

            def predict(self, image, **kwargs):
                self.calls.append((image, kwargs))
                return []

        ai = TouhouAIController.__new__(TouhouAIController)
        ai.model = FakeModel()
        ai.inference_device = "cpu"
        ai.confidence_threshold = 0.12
        ai.stats = {"detections": 0}
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        self.assertEqual(ai.detect_objects(frame), [])
        _, kwargs = ai.model.calls[0]
        self.assertEqual(kwargs["device"], "cpu")
        self.assertEqual(kwargs["conf"], 0.03)

    def test_bottom_red_sprite_recovers_misclassified_reimu(self) -> None:
        detections = [
            {
                "class_name": "character",
                "center_x": 205,
                "center_y": 280,
                "width": 20,
                "height": 24,
                "confidence": 0.92,
            },
            {
                "class_name": "enemy_small_red",
                "center_x": 225,
                "center_y": 405,
                "width": 19,
                "height": 22,
                "confidence": 0.04,
            },
            {
                "class_name": "enemy_small_red",
                "center_x": 90,
                "center_y": 100,
                "width": 20,
                "height": 22,
                "confidence": 0.85,
            },
        ]

        state = self.ai.analyze_game_state(detections)

        self.assertEqual(state["player_x"], 225)
        self.assertEqual(state["player_y"], 405)
        self.assertEqual(
            state["player"]["player_detection_source"],
            "red_sprite_fallback",
        )
        self.assertEqual(len(state["enemies"]), 1)

    def test_danger_is_measured_from_detected_player(self) -> None:
        detections = [
            {"class_name": "character", "center_x": 100, "center_y": 400},
            {
                "class_name": "bullet_enemy_small_blue",
                "center_x": 110,
                "center_y": 405,
            },
        ]

        state = self.ai.analyze_game_state(detections)

        self.assertEqual(state["player_x"], 100)
        self.assertEqual(state["player_y"], 400)
        self.assertGreaterEqual(state["danger_level"], 3)
        self.assertGreater(state["max_collision_risk"], 0)
        self.assertEqual(len(state["immediate_threats"]), 1)

    def test_window_locator_is_independent_from_advanced_keyboard_module(self) -> None:
        self.assertTrue(WINDOW_CONTROLLER_AVAILABLE)

    def test_focus_guard_activates_game_before_allowing_input(self) -> None:
        class FakeWindowController:
            active = False

            def is_game_window_active(self):
                return self.active

            def activate_window(self):
                self.active = True
                return True

        ai = TouhouAIController.__new__(
            TouhouAIController
        )
        ai.window_controller = FakeWindowController()
        ai.game_found = True
        ai.allow_no_game = False
        ai.auto_focus_game = True
        ai.last_focus_check = 0.0
        ai.focus_check_interval = 0.0
        ai.game_focus_ok = False

        self.assertTrue(ai.ensure_game_focus(force=True))
        self.assertTrue(ai.window_controller.active)

    def test_scene_and_player_timeout_gate_actions(self) -> None:
        ai = TouhouAIController.__new__(TouhouAIController)
        ai.screen_region = {"width": 640, "height": 480}
        ai.safe_margin = 36
        ai.player_lost_timeout = 0.7
        ai.allow_vertical = False
        ai.ai_mode = "balanced"
        ai.bullet_threshold = 0.05
        ai.auto_bomb = False
        ai.stats = {
            "safe_stop_frames": 0,
            "direction_switches": 0,
            "blocked_switches": 0,
            "player_predicted_frames": 0,
            "collision_warning_frames": 0,
            "dodge_count": 0,
        }
        detections = [
            {
                "class_name": "character",
                "center_x": 300,
                "center_y": 400,
                "confidence": 0.9,
            },
            {
                "class_name": "bullet_enemy_small_red",
                "center_x": 300,
                "center_y": 250,
            },
        ]
        first = ai.analyze_game_state(detections, timestamp=1.0)
        self.assertFalse(first["action_allowed"])
        second = ai.analyze_game_state(detections, timestamp=1.1)
        self.assertTrue(second["action_allowed"])
        _movement, shooting, _focused = ai.make_decision(second)
        self.assertTrue(shooting)

        missing = ai.analyze_game_state([], timestamp=1.9)
        self.assertFalse(missing["safe_to_control"])
        movement, shooting, focused = ai.make_decision(missing)
        self.assertEqual((movement, shooting, focused), ("stay", False, False))

if __name__ == "__main__":
    unittest.main()
