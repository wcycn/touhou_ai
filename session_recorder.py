#!/usr/bin/env python3
"""运行会话记录、汇总和离线报告。"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from typing import Any, Iterable, Optional

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_SESSIONS_DIR = PROJECT_DIR / "sessions"
FORMAT_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


class SessionRecorder:
    """以JSONL记录每次决策，并按固定频率抽样保存画面。"""

    def __init__(
        self,
        base_dir: Path | str = DEFAULT_SESSIONS_DIR,
        config: Optional[dict] = None,
        frame_sample_fps: float = 2.0,
        source: str = "ai",
    ):
        if frame_sample_fps < 0:
            raise ValueError("frame_sample_fps不能为负数")
        self.base_dir = Path(base_dir).expanduser().resolve()
        self.config = json_safe(config or {})
        self.frame_sample_fps = frame_sample_fps
        self.source = source

        local_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.session_id = f"{local_stamp}_{source}"
        self.session_dir = self.base_dir / self.session_id
        self.frames_dir = self.session_dir / "frames"
        self.events_path = self.session_dir / "events.jsonl"
        self.metadata_path = self.session_dir / "metadata.json"

        self.started_at = utc_now()
        self.started_monotonic = time.monotonic()
        self.last_saved_at = float("-inf")
        self.event_count = 0
        self.saved_frame_count = 0
        self.total_detections = 0
        self.player_detected_frames = 0
        self.max_danger = 0
        self.max_collision_risk = 0.0
        self.action_counts: Counter[str] = Counter()
        self.scene_counts: Counter[str] = Counter()
        self.tracked_player_frames = 0
        self.predicted_player_frames = 0
        self.safe_stop_frames = 0
        self.focus_loss_frames = 0
        self.direction_switches = 0
        self.last_movement: Optional[str] = None
        self._event_file = None
        self._closed = False
        self._lock = threading.Lock()

        self.frames_dir.mkdir(parents=True, exist_ok=False)
        self._event_file = self.events_path.open(
            "a",
            encoding="utf-8",
            buffering=1,
        )
        self._write_metadata(status="running")

    def _summary(self) -> dict:
        player_rate = (
            self.player_detected_frames / self.event_count
            if self.event_count
            else 0.0
        )
        return {
            "event_count": self.event_count,
            "saved_frame_count": self.saved_frame_count,
            "total_detections": self.total_detections,
            "player_detected_frames": self.player_detected_frames,
            "player_detection_rate": round(player_rate, 4),
            "tracked_player_frames": self.tracked_player_frames,
            "tracked_player_rate": round(
                self.tracked_player_frames / self.event_count
                if self.event_count
                else 0.0,
                4,
            ),
            "predicted_player_frames": self.predicted_player_frames,
            "max_danger": self.max_danger,
            "max_collision_risk": round(self.max_collision_risk, 4),
            "action_counts": dict(self.action_counts),
            "scene_counts": dict(self.scene_counts),
            "safe_stop_frames": self.safe_stop_frames,
            "focus_loss_frames": self.focus_loss_frames,
            "direction_switches": self.direction_switches,
        }

    def _write_metadata(
        self,
        status: str,
        end_reason: Optional[str] = None,
        final_stats: Optional[dict] = None,
    ) -> None:
        metadata = {
            "format_version": FORMAT_VERSION,
            "session_id": self.session_id,
            "source": self.source,
            "status": status,
            "started_at": self.started_at,
            "ended_at": utc_now() if status != "running" else None,
            "duration_seconds": round(
                time.monotonic() - self.started_monotonic,
                3,
            ),
            "end_reason": end_reason,
            "frame_sample_fps": self.frame_sample_fps,
            "config": self.config,
            "summary": self._summary(),
            "final_stats": json_safe(final_stats or {}),
        }
        atomic_write_json(self.metadata_path, metadata)

    def record(
        self,
        frame: np.ndarray,
        detections: list[dict],
        game_state: dict,
        action: dict,
        screen_region: Optional[dict] = None,
    ) -> None:
        with self._lock:
            if self._closed:
                return

            now_monotonic = time.monotonic()
            frame_file = None
            save_interval = (
                1.0 / self.frame_sample_fps
                if self.frame_sample_fps > 0
                else None
            )
            if (
                save_interval is not None
                and now_monotonic - self.last_saved_at >= save_interval
            ):
                frame_name = f"{self.event_count:08d}.jpg"
                frame_path = self.frames_dir / frame_name
                if cv2.imwrite(str(frame_path), frame):
                    frame_file = f"frames/{frame_name}"
                    self.saved_frame_count += 1
                    self.last_saved_at = now_monotonic

            movement = str(action.get("movement", "unknown"))
            self.action_counts[movement] += 1
            if (
                self.last_movement is not None
                and movement != self.last_movement
            ):
                self.direction_switches += 1
            self.last_movement = movement
            self.total_detections += len(detections)
            player = game_state.get("player")
            if player:
                self.player_detected_frames += 1
            tracked_player = game_state.get("tracked_player")
            if tracked_player and tracked_player.get("valid"):
                self.tracked_player_frames += 1
            if game_state.get("player_source") == "predicted":
                self.predicted_player_frames += 1
            danger = int(game_state.get("danger_level", 0))
            self.max_danger = max(self.max_danger, danger)
            collision_risk = float(game_state.get("max_collision_risk", 0.0))
            self.max_collision_risk = max(
                self.max_collision_risk,
                collision_risk,
            )
            scene_state = str(game_state.get("scene_state", "unknown"))
            self.scene_counts[scene_state] += 1
            if not game_state.get("action_allowed", False):
                self.safe_stop_frames += 1
            if action.get("focus_ok") is False:
                self.focus_loss_frames += 1

            event = {
                "index": self.event_count,
                "timestamp": utc_now(),
                "elapsed_seconds": round(
                    now_monotonic - self.started_monotonic,
                    4,
                ),
                "frame_file": frame_file,
                "screen_region": screen_region,
                "detections": detections,
                "state": {
                    "danger_level": danger,
                    "player": player,
                    "player_x": game_state.get("player_x"),
                    "player_y": game_state.get("player_y"),
                    "tracked_player": tracked_player,
                    "player_source": game_state.get("player_source"),
                    "player_valid": game_state.get("player_valid"),
                    "safe_to_control": game_state.get("safe_to_control"),
                    "scene_state": scene_state,
                    "action_allowed": game_state.get("action_allowed"),
                    "bullet_count": len(game_state.get("bullets", [])),
                    "player_bullet_count": len(
                        game_state.get("player_bullets", [])
                    ),
                    "enemy_count": len(game_state.get("enemies", [])),
                    "powerup_count": len(game_state.get("powerups", [])),
                    "immediate_threat_count": len(
                        game_state.get("immediate_threats", [])
                    ),
                    "predicted_threat_count": len(
                        game_state.get("predicted_threats", [])
                    ),
                    "max_collision_risk": collision_risk,
                    "risk_trigger": game_state.get("risk_trigger"),
                    "candidate_costs": game_state.get(
                        "candidate_costs",
                        {},
                    ),
                    "bullet_tracks": [
                        {
                            key: bullet.get(key)
                            for key in (
                                "track_id",
                                "class_name",
                                "center_x",
                                "center_y",
                                "velocity_x",
                                "velocity_y",
                                "ttc_seconds",
                                "closest_distance",
                                "collision_risk",
                                "predicted_x",
                                "predicted_y",
                            )
                        }
                        for bullet in game_state.get("bullets", [])
                    ],
                },
                "action": action,
            }
            self._event_file.write(
                json.dumps(json_safe(event), ensure_ascii=False) + "\n"
            )
            self.event_count += 1

            # 运行中也周期性更新摘要，异常退出后仍能看到接近最终的数据。
            if self.event_count % 50 == 0:
                self._write_metadata(status="running")

    def close(
        self,
        end_reason: str = "completed",
        final_stats: Optional[dict] = None,
    ) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._event_file:
                self._event_file.flush()
                self._event_file.close()
            self._write_metadata(
                status="completed",
                end_reason=end_reason,
                final_stats=final_stats,
            )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _traceback):
        reason = "exception" if exc_type else "completed"
        self.close(end_reason=reason)


def load_metadata(session_dir: Path | str) -> dict:
    path = Path(session_dir) / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def iter_events(session_dir: Path | str) -> Iterable[dict]:
    path = Path(session_dir) / "events.jsonl"
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as event_file:
        for line in event_file:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # 最后一行可能因断电/强杀而不完整，已完成的行仍然可用。
                continue


def list_sessions(base_dir: Path | str = DEFAULT_SESSIONS_DIR) -> list[dict]:
    root = Path(base_dir)
    if not root.is_dir():
        return []
    sessions = []
    for session_dir in sorted(root.iterdir(), reverse=True):
        if not session_dir.is_dir():
            continue
        try:
            metadata = load_metadata(session_dir)
        except (OSError, json.JSONDecodeError):
            continue
        metadata["session_dir"] = str(session_dir)
        sessions.append(metadata)
    return sessions


def build_report(session_dir: Path | str) -> dict:
    session_dir = Path(session_dir)
    try:
        metadata = load_metadata(session_dir)
    except (OSError, json.JSONDecodeError):
        metadata = {
            "session_id": session_dir.name,
            "status": "incomplete",
            "summary": {},
        }

    events = list(iter_events(session_dir))
    danger_events = sorted(
        events,
        key=lambda event: event.get("state", {}).get("danger_level", 0),
        reverse=True,
    )[:10]
    missing_player_runs = []
    current_run = None
    for event in events:
        player_present = bool(event.get("state", {}).get("player"))
        if not player_present and current_run is None:
            current_run = {
                "start_index": event.get("index"),
                "start_seconds": event.get("elapsed_seconds"),
                "frames": 0,
            }
        if not player_present and current_run is not None:
            current_run["frames"] += 1
        if player_present and current_run is not None:
            current_run["end_index"] = event.get("index", 0) - 1
            missing_player_runs.append(current_run)
            current_run = None
    if current_run is not None:
        current_run["end_index"] = events[-1].get("index") if events else None
        missing_player_runs.append(current_run)

    return {
        "metadata": metadata,
        "event_count_read": len(events),
        "highest_danger_events": [
            {
                "index": event.get("index"),
                "elapsed_seconds": event.get("elapsed_seconds"),
                "danger_level": event.get("state", {}).get("danger_level"),
                "frame_file": event.get("frame_file"),
                "action": event.get("action"),
            }
            for event in danger_events
        ],
        "longest_missing_player_runs": sorted(
            missing_player_runs,
            key=lambda item: item["frames"],
            reverse=True,
        )[:10],
    }


def write_report(session_dir: Path | str) -> tuple[Path, Path]:
    session_dir = Path(session_dir)
    report = build_report(session_dir)
    json_path = session_dir / "report.json"
    markdown_path = session_dir / "report.md"
    atomic_write_json(json_path, report)

    metadata = report["metadata"]
    summary = metadata.get("summary", {})
    lines = [
        f"# 会话报告：{metadata.get('session_id', session_dir.name)}",
        "",
        f"- 来源：{metadata.get('source', 'unknown')}",
        f"- 状态：{metadata.get('status', 'unknown')}",
        f"- 开始：{metadata.get('started_at', 'unknown')}",
        f"- 时长：{metadata.get('duration_seconds', 0)} 秒",
        f"- 事件帧：{summary.get('event_count', report['event_count_read'])}",
        f"- 保存画面：{summary.get('saved_frame_count', 0)}",
        f"- 自机检出率：{summary.get('player_detection_rate', 0):.1%}",
        f"- 跟踪后自机可用率：{summary.get('tracked_player_rate', 0):.1%}",
        f"- 最高危险度：{summary.get('max_danger', 0)}",
        f"- 最高碰撞风险：{summary.get('max_collision_risk', 0)}",
        f"- 方向切换：{summary.get('direction_switches', 0)}",
        f"- 安全停控帧：{summary.get('safe_stop_frames', 0)}",
        f"- 焦点丢失帧：{summary.get('focus_loss_frames', 0)}",
        "",
        "## 动作分布",
        "",
    ]
    for action, count in summary.get("action_counts", {}).items():
        lines.append(f"- {action}: {count}")
    lines.extend(["", "## 最高危险事件", ""])
    for event in report["highest_danger_events"]:
        lines.append(
            f"- 帧 {event['index']} / {event['elapsed_seconds']}秒："
            f"危险度 {event['danger_level']}，动作 {event['action']}"
        )
    lines.extend(["", "## 最长自机漏检区间", ""])
    for run in report["longest_missing_player_runs"]:
        lines.append(
            f"- 帧 {run['start_index']}–{run['end_index']}："
            f"连续 {run['frames']} 帧"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
