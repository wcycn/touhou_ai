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
    detect_color_player_candidates,
    detect_spell_overlay,
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
        self.assertEqual(state["capture_width"], 640)
        self.assertEqual(state["screen_width"], 410)

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

    def test_red_color_fallback_finds_player_inside_playfield(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[380:398, 222:232] = (0, 0, 255)
        frame[400:408, 225:232] = (0, 0, 230)
        # HUD中的红块必须被游戏区域边界排除。
        frame[380:420, 520:560] = (0, 0, 255)

        candidates = detect_color_player_candidates(frame)

        self.assertEqual(len(candidates), 1)
        self.assertLess(candidates[0]["center_x"], 300)
        self.assertGreater(candidates[0]["center_y"], 380)
        self.assertEqual(
            candidates[0]["player_detection_source"],
            "red_color_fallback",
        )

    def test_red_color_fallback_rejects_square_power_items(self) -> None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[396:409, 58:71] = (0, 0, 255)
        frame[430:446, 330:347] = (0, 0, 230)

        self.assertEqual(detect_color_player_candidates(frame), [])

    def test_spell_overlay_requires_large_connected_red_area(self) -> None:
        ordinary = np.zeros((480, 640, 3), dtype=np.uint8)
        ordinary[380:420, 210:235] = (0, 0, 255)
        overlay = np.zeros((480, 640, 3), dtype=np.uint8)
        overlay[120:260, 35:180] = (0, 0, 230)

        self.assertFalse(detect_spell_overlay(ordinary))
        self.assertTrue(detect_spell_overlay(overlay))

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

    def test_player_continuity_rejects_far_hud_candidate(self) -> None:
        self.ai.player_lost_timeout = 0.7

        def red_player(x, y, confidence):
            return {
                "class_name": "enemy_small_red",
                "center_x": x,
                "center_y": y,
                "width": 19,
                "height": 22,
                "confidence": confidence,
            }

        first = self.ai.analyze_game_state(
            [red_player(225, 405, 0.4)],
            timestamp=1.0,
        )
        middle = self.ai.analyze_game_state(
            [
                red_player(228, 350, 0.25),
                red_player(404, 385, 0.9),
            ],
            timestamp=1.1,
        )
        upper = self.ai.analyze_game_state(
            [
                red_player(232, 285, 0.2),
                red_player(405, 384, 0.95),
            ],
            timestamp=1.2,
        )

        self.assertEqual(first["player_x"], 225)
        self.assertLess(middle["player_x"], 260)
        self.assertLess(upper["player_x"], 270)
        self.assertLess(upper["player_y"], 350)

    def test_bottom_color_candidate_reacquires_after_respawn(self) -> None:
        self.ai.player_lost_timeout = 0.7
        self.ai._ensure_control_components()
        tracker = self.ai.player_tracker
        tracker.x = 256
        tracker.y = 226
        tracker.started_at = 0.9
        tracker.last_seen = 1.0
        tracker.last_update = 1.0
        detection = {
            "class_name": "character_color_fallback",
            "center_x": 224,
            "center_y": 408,
            "width": 20,
            "height": 38,
            "confidence": 0.35,
            "player_fallback_candidate": True,
            "player_detection_source": "red_color_fallback",
        }

        first = self.ai.analyze_game_state([detection], timestamp=1.1)
        second = self.ai.analyze_game_state([detection], timestamp=1.2)

        self.assertIsNone(first["player"])
        self.assertIsNotNone(second["player"])
        self.assertEqual(
            second["player"]["player_detection_source"],
            "red_color_fallback",
        )
        self.assertGreater(second["player_y"], 350)

    def test_spell_overlay_holds_last_player_and_keeps_shooting(self) -> None:
        detections = [
            {
                "class_name": "character",
                "center_x": 220,
                "center_y": 405,
                "confidence": 0.9,
            },
            {
                "class_name": "bullet_enemy_small_blue",
                "center_x": 200,
                "center_y": 200,
            },
        ]
        self.ai.analyze_game_state(detections, timestamp=1.0)
        self.ai.analyze_game_state(detections, timestamp=1.1)
        overlay_state = self.ai.analyze_game_state(
            [{"class_name": "scene_spell_overlay", "confidence": 1.0}],
            timestamp=1.2,
        )

        movement, shooting, focused = self.ai.make_decision(overlay_state)

        self.assertTrue(overlay_state["spell_overlay"])
        self.assertAlmostEqual(overlay_state["player_x"], 220)
        self.assertAlmostEqual(overlay_state["player_y"], 405)
        self.assertEqual((movement, shooting, focused), ("stay", True, False))
        self.assertEqual(
            overlay_state["decision_reason"],
            "spell_overlay_hold",
        )

    def test_spell_overlay_timeout_releases_persistent_red_background(self) -> None:
        detections = [
            {
                "class_name": "character",
                "center_x": 220,
                "center_y": 405,
                "confidence": 0.9,
            },
            {
                "class_name": "bullet_enemy_small_blue",
                "center_x": 200,
                "center_y": 200,
            },
        ]
        self.ai.analyze_game_state(detections, timestamp=1.0)
        self.ai.analyze_game_state(detections, timestamp=1.1)
        marker = [{"class_name": "scene_spell_overlay", "confidence": 1.0}]

        first = self.ai.analyze_game_state(marker, timestamp=1.2)
        expired = self.ai.analyze_game_state(marker, timestamp=2.9)

        self.assertTrue(first["spell_overlay"])
        self.assertFalse(expired["spell_overlay"])

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

    def test_imminent_collision_uses_auto_bomb(self) -> None:
        ai = TouhouAIController.__new__(TouhouAIController)
        ai.safe_margin = 36
        ai.player_lost_timeout = 0.7
        ai.allow_vertical = True
        ai.ai_mode = "balanced"
        ai.bullet_threshold = 0.05
        ai.auto_bomb = True
        ai.last_bomb_time = float("-inf")
        ai.bomb_cooldown = 4.0
        ai.estimated_bombs = 3
        ai.stats = {
            "safe_stop_frames": 0,
            "direction_switches": 0,
            "blocked_switches": 0,
            "player_predicted_frames": 0,
            "collision_warning_frames": 0,
            "dodge_count": 0,
        }
        threat = {
            "center_x": 300,
            "center_y": 390,
            "velocity_x": 0,
            "velocity_y": 200,
        }
        state = {
            "action_allowed": True,
            "player_x": 300,
            "player_y": 400,
            "screen_width": 640,
            "screen_height": 480,
            "bullets": [threat],
            "enemies": [],
            "max_collision_risk": 0.97,
            "predicted_threats": [threat],
            "immediate_threats": [threat],
            "player_source": "detected",
        }

        movement, shooting, focused = ai.make_decision(state)

        self.assertEqual((movement, shooting, focused), ("bomb", False, False))
        self.assertEqual(state["decision_reason"], "imminent_collision_bomb")
        self.assertEqual(ai.estimated_bombs, 2)

    def test_safe_powerup_is_collected_before_recentering(self) -> None:
        ai = TouhouAIController.__new__(TouhouAIController)
        ai.safe_margin = 36
        ai.player_lost_timeout = 0.7
        ai.allow_vertical = True
        ai.ai_mode = "defensive"
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
        state = {
            "action_allowed": True,
            "player_x": 300,
            "player_y": 400,
            "screen_width": 410,
            "screen_height": 480,
            "bullets": [],
            "enemies": [],
            "powerups": [
                {
                    "class_name": "powerup_red",
                    "center_x": 120,
                    "center_y": 350,
                }
            ],
            "danger_level": 0,
            "max_collision_risk": 0.0,
            "predicted_threats": [],
            "immediate_threats": [],
            "player_source": "detected",
        }

        movement, shooting, _focused = ai.make_decision(state)

        self.assertIn("left", movement)
        self.assertTrue(shooting)
        self.assertEqual(state["decision_reason"], "item_collection")

if __name__ == "__main__":
    unittest.main()
