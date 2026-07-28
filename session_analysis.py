#!/usr/bin/env python3
"""会话质量评估与人工审核候选导出。"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from statistics import mean
from typing import Iterable

from session_recorder import (
    DEFAULT_SESSIONS_DIR,
    atomic_write_json,
    iter_events,
    list_sessions,
    load_metadata,
)


OPPOSITES = {
    ("left", "right"),
    ("right", "left"),
    ("up", "down"),
    ("down", "up"),
}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def event_elapsed(event: dict) -> float:
    return float(event.get("elapsed_seconds", 0.0) or 0.0)


def analyze_events(events: list[dict], safe_margin: int = 36) -> dict:
    event_count = len(events)
    duration = event_elapsed(events[-1]) if events else 0.0
    raw_player = 0
    tracked_player = 0
    predicted_player = 0
    safe_stops = 0
    focus_losses = 0
    unsafe_executions = 0
    boundary_frames = 0
    direction_switches = 0
    reversals = 0
    last_movement = None
    risks = []
    scene_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for event in events:
        state = event.get("state", {})
        action = event.get("action", {})
        if state.get("player"):
            raw_player += 1
        tracked = state.get("tracked_player") or {}
        if tracked.get("valid"):
            tracked_player += 1
        if state.get("player_source") == "predicted":
            predicted_player += 1
        scene_counts[str(state.get("scene_state", "unknown"))] += 1
        if not state.get("action_allowed", False):
            safe_stops += 1
            if action.get("executed") and action.get("movement") != "stay":
                unsafe_executions += 1
        if action.get("focus_ok") is False:
            focus_losses += 1

        risk = float(state.get("max_collision_risk", 0.0) or 0.0)
        risks.append(risk)
        movement = str(action.get("movement", "unknown"))
        action_counts[movement] += 1
        reason_counts[str(action.get("decision_reason") or "unknown")] += 1
        if last_movement is not None and movement != last_movement:
            direction_switches += 1
            if (last_movement, movement) in OPPOSITES:
                reversals += 1
        last_movement = movement

        region = event.get("screen_region") or {}
        width = int(region.get("width", 0) or 0)
        height = int(region.get("height", 0) or 0)
        player_x = state.get("player_x")
        player_y = state.get("player_y")
        if (
            width > 0
            and height > 0
            and player_x is not None
            and player_y is not None
            and (
                float(player_x) <= safe_margin
                or float(player_x) >= width - safe_margin
                or float(player_y) <= safe_margin
                or float(player_y) >= height - safe_margin
            )
        ):
            boundary_frames += 1

    minutes = duration / 60.0
    return {
        "event_count": event_count,
        "duration_seconds": round(duration, 3),
        "estimated_event_fps": round(event_count / duration, 3) if duration else 0.0,
        "raw_player_detection_rate": round(
            raw_player / event_count if event_count else 0.0,
            4,
        ),
        "tracked_player_rate": round(
            tracked_player / event_count if event_count else 0.0,
            4,
        ),
        "predicted_player_frames": predicted_player,
        "safe_stop_frames": safe_stops,
        "focus_loss_frames": focus_losses,
        "unsafe_execution_frames": unsafe_executions,
        "boundary_frame_rate": round(
            boundary_frames / event_count if event_count else 0.0,
            4,
        ),
        "direction_switches": direction_switches,
        "direction_switches_per_minute": round(
            direction_switches / minutes if minutes else 0.0,
            3,
        ),
        "direct_reversals": reversals,
        "collision_risk_mean": round(mean(risks), 4) if risks else 0.0,
        "collision_risk_p95": round(percentile(risks, 0.95), 4),
        "collision_risk_max": round(max(risks), 4) if risks else 0.0,
        "scene_counts": dict(scene_counts),
        "action_counts": dict(action_counts),
        "decision_reason_counts": dict(reason_counts),
    }


def review_score(event: dict, safe_margin: int = 36) -> tuple[float, list[str]]:
    state = event.get("state", {})
    action = event.get("action", {})
    score = 0.0
    reasons = []
    risk = float(state.get("max_collision_risk", 0.0) or 0.0)
    if risk >= 0.65:
        score += risk * 5
        reasons.append("high_collision_risk")
    if not state.get("player") and state.get("tracked_player"):
        score += 3.0
        reasons.append("raw_player_missing")
    if action.get("focus_ok") is False:
        score += 4.0
        reasons.append("focus_loss")
    if (
        action.get("executed")
        and not state.get("action_allowed", False)
        and action.get("movement") != "stay"
    ):
        score += 10.0
        reasons.append("unsafe_execution")

    region = event.get("screen_region") or {}
    width = int(region.get("width", 0) or 0)
    height = int(region.get("height", 0) or 0)
    x = state.get("player_x")
    y = state.get("player_y")
    if (
        width
        and height
        and x is not None
        and y is not None
        and (
            float(x) <= safe_margin
            or float(x) >= width - safe_margin
            or float(y) <= safe_margin
            or float(y) >= height - safe_margin
        )
    ):
        score += 2.0
        reasons.append("near_boundary")
    return score, reasons


def build_review_manifest(
    session_dir: Path | str,
    events: Iterable[dict],
    limit: int = 100,
) -> list[dict]:
    session_dir = Path(session_dir)
    candidates = []
    for event in events:
        frame_file = event.get("frame_file")
        if not frame_file:
            continue
        score, reasons = review_score(event)
        if score <= 0:
            continue
        candidates.append(
            {
                "session_id": session_dir.name,
                "event_index": event.get("index"),
                "elapsed_seconds": event.get("elapsed_seconds"),
                "image": str((session_dir / frame_file).resolve()),
                "score": round(score, 4),
                "reasons": reasons,
                "detections_are_pseudo_labels": True,
                "needs_human_review": True,
                "detections": event.get("detections", []),
                "state": event.get("state", {}),
                "action": event.get("action", {}),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[: max(0, limit)]


def write_analysis(
    session_dir: Path | str,
    review_limit: int = 100,
) -> tuple[Path, Path, Path]:
    session_dir = Path(session_dir).resolve()
    events = list(iter_events(session_dir))
    try:
        metadata = load_metadata(session_dir)
    except (OSError, json.JSONDecodeError):
        metadata = {"session_id": session_dir.name}
    safe_margin = int(metadata.get("config", {}).get("safe_margin", 36))
    metrics = analyze_events(events, safe_margin=safe_margin)
    review = build_review_manifest(session_dir, events, limit=review_limit)
    result = {
        "session_id": metadata.get("session_id", session_dir.name),
        "metrics": metrics,
        "review_candidate_count": len(review),
    }

    json_path = session_dir / "analysis.json"
    markdown_path = session_dir / "analysis.md"
    review_path = session_dir / "review_manifest.jsonl"
    atomic_write_json(json_path, result)

    lines = [
        f"# 会话分析：{result['session_id']}",
        "",
        f"- 事件帧：{metrics['event_count']}",
        f"- 时长：{metrics['duration_seconds']} 秒",
        f"- 原始自机检出率：{metrics['raw_player_detection_rate']:.1%}",
        f"- 跟踪后自机可用率：{metrics['tracked_player_rate']:.1%}",
        f"- 自机预测帧：{metrics['predicted_player_frames']}",
        f"- 安全停控帧：{metrics['safe_stop_frames']}",
        f"- 焦点丢失帧：{metrics['focus_loss_frames']}",
        f"- 非法执行帧：{metrics['unsafe_execution_frames']}",
        f"- 边界帧占比：{metrics['boundary_frame_rate']:.1%}",
        f"- 方向切换/分钟：{metrics['direction_switches_per_minute']}",
        f"- 直接反向次数：{metrics['direct_reversals']}",
        f"- 碰撞风险P95：{metrics['collision_risk_p95']}",
        f"- 人工审核候选：{len(review)}",
        "",
        "## 场景分布",
        "",
    ]
    for name, count in metrics["scene_counts"].items():
        lines.append(f"- {name}: {count}")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with review_path.open("w", encoding="utf-8") as manifest:
        for item in review:
            manifest.write(json.dumps(item, ensure_ascii=False) + "\n")
    return json_path, markdown_path, review_path


def export_review_dataset(
    session_dir: Path | str,
    limit: int = 100,
    output_dir: Path | str | None = None,
) -> Path:
    """复制高价值抽样帧并生成明确标为伪标签的YOLO预标注。"""
    session_dir = Path(session_dir).resolve()
    events = list(iter_events(session_dir))
    candidates = build_review_manifest(session_dir, events, limit=limit)
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else session_dir / "labeling_export"
    )
    images_dir = output_dir / "images"
    prelabels_dir = output_dir / "prelabels"
    images_dir.mkdir(parents=True, exist_ok=True)
    prelabels_dir.mkdir(parents=True, exist_ok=True)
    events_by_index = {
        event.get("index"): event
        for event in events
    }

    exported = []
    for item in candidates:
        source = Path(item["image"])
        if not source.is_file():
            continue
        stem = f"{session_dir.name}_{int(item['event_index']):08d}"
        target = images_dir / f"{stem}{source.suffix.lower()}"
        shutil.copy2(source, target)
        event = events_by_index.get(item.get("event_index"), {})
        region = event.get("screen_region") or {}
        width = float(region.get("width", 0) or 0)
        height = float(region.get("height", 0) or 0)
        label_lines = []
        if width > 0 and height > 0:
            for detection in item.get("detections", []):
                bbox = detection.get("bbox", [])
                class_id = detection.get("class_id")
                if len(bbox) != 4 or class_id is None:
                    continue
                x1, y1, x2, y2 = (float(value) for value in bbox)
                center_x = ((x1 + x2) / 2) / width
                center_y = ((y1 + y2) / 2) / height
                box_width = (x2 - x1) / width
                box_height = (y2 - y1) / height
                values = [
                    max(0.0, min(1.0, value))
                    for value in (center_x, center_y, box_width, box_height)
                ]
                label_lines.append(
                    f"{int(class_id)} "
                    + " ".join(f"{value:.6f}" for value in values)
                )
        (prelabels_dir / f"{stem}.txt").write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )
        exported.append(
            {
                **item,
                "exported_image": str(target.relative_to(output_dir)),
                "prelabel": str(
                    (prelabels_dir / f"{stem}.txt").relative_to(output_dir)
                ),
            }
        )

    atomic_write_json(
        output_dir / "manifest.json",
        {
            "source_session": str(session_dir),
            "image_count": len(exported),
            "labels_are_model_predictions": True,
            "must_be_human_reviewed": True,
            "items": exported,
        },
    )
    (output_dir / "README.md").write_text(
        "# 标注候选\n\n"
        "`prelabels/` 中的框来自当前模型预测，不是真实标注。"
        "必须逐张人工检查、修正或删除后，才能移动到正式 `labels/` "
        "并用于模型评估或训练。\n",
        encoding="utf-8",
    )
    return output_dir


def resolve_session(path: Path | None) -> Path:
    if path is not None:
        resolved = path.expanduser().resolve()
        if not (resolved / "events.jsonl").is_file():
            raise FileNotFoundError(f"不是有效会话目录: {resolved}")
        return resolved
    sessions = list_sessions()
    if not sessions:
        raise FileNotFoundError("没有可分析的会话")
    return Path(sessions[0]["session_dir"])


def main() -> int:
    parser = argparse.ArgumentParser(description="分析Touhou AI运行会话")
    parser.add_argument("session", nargs="?", type=Path, help="会话目录；默认最新")
    parser.add_argument("--all", action="store_true", help="分析sessions下全部会话")
    parser.add_argument("--review-limit", type=int, default=100)
    parser.add_argument(
        "--export-review",
        action="store_true",
        help="复制候选图片并生成待人工修正的YOLO预标注",
    )
    args = parser.parse_args()
    try:
        if args.all:
            sessions = [
                Path(item["session_dir"])
                for item in list_sessions(DEFAULT_SESSIONS_DIR)
            ]
            if not sessions:
                raise FileNotFoundError("没有可分析的会话")
        else:
            sessions = [resolve_session(args.session)]
        for session_dir in sessions:
            paths = write_analysis(session_dir, args.review_limit)
            print(f"✅ 会话分析完成: {session_dir}")
            for path in paths:
                print(f"  {path}")
            if args.export_review:
                export_dir = export_review_dataset(
                    session_dir,
                    args.review_limit,
                )
                print(f"  标注候选: {export_dir}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"❌ 会话分析失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
