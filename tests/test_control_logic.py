"""跟踪、规划、场景门控和输入状态机测试。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control_logic import (  # noqa: E402
    ActionStabilizer,
    BulletTracker,
    GameSceneStateMachine,
    InputStateMachine,
    PlayerTracker,
    RiskPlanner,
    collision_metrics,
)


class PlayerTrackerTests(unittest.TestCase):
    def test_short_missing_run_is_predicted_then_stops(self) -> None:
        tracker = PlayerTracker(
            prediction_timeout=0.3,
            stop_timeout=0.6,
            smoothing=1.0,
        )
        tracker.update(
            {"center_x": 100, "center_y": 400, "confidence": 0.9},
            640,
            480,
            now=1.0,
        )
        tracker.update(
            {"center_x": 110, "center_y": 400, "confidence": 0.9},
            640,
            480,
            now=1.1,
        )
        predicted = tracker.update(None, 640, 480, now=1.2)
        self.assertEqual(predicted.source, "predicted")
        self.assertTrue(predicted.safe_to_control)
        self.assertGreater(predicted.x, 110)

        stale = tracker.update(None, 640, 480, now=1.5)
        self.assertEqual(stale.source, "stale")
        self.assertFalse(stale.safe_to_control)
        expired = tracker.update(None, 640, 480, now=1.8)
        self.assertFalse(expired.valid)


class BulletTrackingTests(unittest.TestCase):
    def test_velocity_and_collision_are_estimated(self) -> None:
        tracker = BulletTracker(velocity_smoothing=1.0)
        first = tracker.update(
            [{"center_x": 100, "center_y": 100, "class_name": "bullet"}],
            now=1.0,
        )[0]
        second = tracker.update(
            [{"center_x": 110, "center_y": 120, "class_name": "bullet"}],
            now=1.1,
        )[0]
        self.assertEqual(first["track_id"], second["track_id"])
        self.assertAlmostEqual(second["velocity_x"], 100, delta=0.01)
        self.assertAlmostEqual(second["velocity_y"], 200, delta=0.01)

        metrics = collision_metrics(
            {
                "center_x": 100,
                "center_y": 100,
                "velocity_x": 0,
                "velocity_y": 200,
            },
            player_x=100,
            player_y=200,
            horizon=1.0,
        )
        self.assertGreaterEqual(metrics["collision_risk"], 0.85)
        self.assertAlmostEqual(metrics["ttc_seconds"], 0.5)


class PlanningTests(unittest.TestCase):
    def test_boundary_forces_inward_and_reversal_is_cooled_down(self) -> None:
        stabilizer = ActionStabilizer(
            safe_margin=30,
            min_hold_seconds=0.1,
            reversal_cooldown=0.4,
        )
        movement, reason = stabilizer.stabilize(
            "left", 10, 300, 640, 480, True, now=1.0
        )
        self.assertEqual((movement, reason), ("right", "left_boundary"))
        movement, reason = stabilizer.stabilize(
            "left", 300, 300, 640, 480, True, now=1.1
        )
        self.assertEqual((movement, reason), ("right", "switch_cooldown"))
        movement, _ = stabilizer.stabilize(
            "left", 300, 300, 640, 480, True, now=1.5
        )
        self.assertEqual(movement, "left")

    def test_planner_avoids_future_bullet_position(self) -> None:
        planner = RiskPlanner(safe_margin=30, allow_vertical=False)
        movement, costs = planner.choose(
            [
                {
                    "center_x": 250,
                    "center_y": 300,
                    "velocity_x": 100,
                    "velocity_y": 0,
                }
            ],
            player_x=320,
            player_y=300,
            width=640,
            height=480,
        )
        self.assertIn(movement, {"left", "right", "stay"})
        self.assertNotEqual(costs["left"], costs["right"])


class SceneAndInputTests(unittest.TestCase):
    def test_scene_requires_repeated_battle_evidence(self) -> None:
        machine = GameSceneStateMachine(
            battle_confirm_frames=2,
            lost_confirm_frames=2,
        )
        first = machine.update(
            player_detected=True,
            player_safe=True,
            bullet_count=1,
            enemy_count=0,
            player_bullet_count=0,
        )
        self.assertEqual(first, "unknown")
        second = machine.update(
            player_detected=True,
            player_safe=True,
            bullet_count=1,
            enemy_count=0,
            player_bullet_count=0,
        )
        self.assertEqual(second, "battle")
        self.assertTrue(machine.action_allowed)
        machine.update(
            player_detected=False,
            player_safe=False,
            bullet_count=0,
            enemy_count=0,
            player_bullet_count=0,
        )
        final = machine.update(
            player_detected=False,
            player_safe=False,
            bullet_count=0,
            enemy_count=0,
            player_bullet_count=0,
        )
        self.assertEqual(final, "transition")
        self.assertFalse(machine.action_allowed)

    def test_input_backend_receives_only_state_differences(self) -> None:
        class Backend:
            def __init__(self):
                self.events = []

            def keyDown(self, key):
                self.events.append(("down", key))

            def keyUp(self, key):
                self.events.append(("up", key))

            def press(self, key):
                self.events.append(("press", key))

        backend = Backend()
        inputs = InputStateMachine(backend)
        inputs.apply("left", True, False)
        first_count = len(backend.events)
        inputs.apply("left", True, False)
        self.assertEqual(len(backend.events), first_count)
        inputs.apply("right", True, False)
        self.assertIn(("up", "left"), backend.events)
        self.assertIn(("down", "right"), backend.events)
        inputs.release_all()
        self.assertEqual(inputs.held, set())


if __name__ == "__main__":
    unittest.main()

