"""会话记录、容错读取和报告生成测试。"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT
sys.path.insert(0, str(PROJECT_DIR))

from session_recorder import (  # noqa: E402
    SessionRecorder,
    build_report,
    iter_events,
    load_metadata,
    write_report,
)


class SessionRecorderTests(unittest.TestCase):
    def test_record_finalize_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorder = SessionRecorder(
                base_dir=temporary,
                config={"mode": "balanced"},
                frame_sample_fps=10.0,
                source="test",
            )
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            detections = [
                {
                    "class_name": "character",
                    "confidence": np.float32(0.9),
                    "bbox": [1, 2, 3, 4],
                    "center_x": 2,
                    "center_y": 3,
                }
            ]
            state = {
                "player": detections[0],
                "player_x": 2,
                "player_y": 3,
                "bullets": [],
                "player_bullets": [],
                "enemies": [],
                "powerups": [],
                "immediate_threats": [],
                "danger_level": 0,
            }
            action = {
                "movement": "left",
                "shooting": True,
                "executed": True,
            }

            recorder.record(frame, detections, state, action)
            recorder.close("test_complete", {"movements": 1})

            metadata = load_metadata(recorder.session_dir)
            events = list(iter_events(recorder.session_dir))
            report = build_report(recorder.session_dir)
            json_path, markdown_path = write_report(recorder.session_dir)

            self.assertEqual(metadata["status"], "completed")
            self.assertEqual(metadata["summary"]["event_count"], 1)
            self.assertEqual(
                metadata["summary"]["player_detection_rate"],
                1.0,
            )
            self.assertEqual(len(events), 1)
            self.assertAlmostEqual(
                events[0]["detections"][0]["confidence"],
                0.9,
                places=5,
            )
            self.assertEqual(report["event_count_read"], 1)
            self.assertTrue(json_path.is_file())
            self.assertTrue(markdown_path.is_file())

    def test_incomplete_final_jsonl_line_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            session_dir = Path(temporary)
            (session_dir / "events.jsonl").write_text(
                json.dumps({"index": 0}) + "\n{\"index\":",
                encoding="utf-8",
            )
            self.assertEqual(list(iter_events(session_dir)), [{"index": 0}])


if __name__ == "__main__":
    unittest.main()
