"""会话分析指标和审核候选测试。"""

from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from session_analysis import (  # noqa: E402
    analyze_events,
    build_review_manifest,
    export_review_dataset,
)


class SessionAnalysisTests(unittest.TestCase):
    def test_metrics_find_reversals_focus_loss_and_unsafe_input(self) -> None:
        events = [
            {
                "elapsed_seconds": 0.0,
                "screen_region": {"width": 640, "height": 480},
                "state": {
                    "player": {"center_x": 300},
                    "tracked_player": {"valid": True},
                    "player_x": 300,
                    "player_y": 400,
                    "scene_state": "battle",
                    "action_allowed": True,
                    "max_collision_risk": 0.2,
                },
                "action": {
                    "movement": "left",
                    "executed": True,
                    "focus_ok": True,
                },
            },
            {
                "elapsed_seconds": 1.0,
                "screen_region": {"width": 640, "height": 480},
                "state": {
                    "player": None,
                    "tracked_player": {"valid": False},
                    "player_source": "stale",
                    "player_x": 20,
                    "player_y": 400,
                    "scene_state": "transition",
                    "action_allowed": False,
                    "max_collision_risk": 0.9,
                },
                "action": {
                    "movement": "right",
                    "executed": True,
                    "focus_ok": False,
                },
            },
        ]
        metrics = analyze_events(events)
        self.assertEqual(metrics["direct_reversals"], 1)
        self.assertEqual(metrics["focus_loss_frames"], 1)
        self.assertEqual(metrics["unsafe_execution_frames"], 1)
        self.assertEqual(metrics["boundary_frame_rate"], 0.5)

    def test_review_manifest_only_references_saved_frames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory)
            event = {
                "index": 2,
                "elapsed_seconds": 1.5,
                "frame_file": "frames/00000002.jpg",
                "detections": [],
                "state": {
                    "player": None,
                    "tracked_player": {"valid": True},
                    "max_collision_risk": 0.8,
                    "action_allowed": True,
                },
                "action": {"movement": "left", "focus_ok": True},
            }
            manifest = build_review_manifest(session_dir, [event])
            self.assertEqual(len(manifest), 1)
            self.assertTrue(manifest[0]["needs_human_review"])
            self.assertIn("high_collision_risk", manifest[0]["reasons"])

    def test_review_export_keeps_predictions_separate_from_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session_dir = Path(directory) / "session"
            frames_dir = session_dir / "frames"
            frames_dir.mkdir(parents=True)
            (frames_dir / "00000000.jpg").write_bytes(b"jpeg-placeholder")
            event = {
                "index": 0,
                "elapsed_seconds": 0.0,
                "frame_file": "frames/00000000.jpg",
                "screen_region": {"width": 100, "height": 100},
                "detections": [
                    {
                        "class_id": 1,
                        "class_name": "bullet",
                        "bbox": [10, 20, 30, 40],
                    }
                ],
                "state": {
                    "player": None,
                    "tracked_player": {"valid": True},
                    "max_collision_risk": 0.9,
                    "action_allowed": True,
                },
                "action": {"movement": "stay", "focus_ok": True},
            }
            (session_dir / "events.jsonl").write_text(
                json.dumps(event) + "\n",
                encoding="utf-8",
            )
            output = export_review_dataset(session_dir)
            self.assertTrue((output / "prelabels").is_dir())
            self.assertFalse((output / "labels").exists())
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["must_be_human_reviewed"])


if __name__ == "__main__":
    unittest.main()
