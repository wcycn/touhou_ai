"""可独立测试的跟踪、风险规划、场景判断和键盘状态逻辑。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import time
from typing import Any, Optional


MOVEMENT_KEYS = {
    "left": {"left"},
    "right": {"right"},
    "up": {"up"},
    "down": {"down"},
    "stay": set(),
}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass
class PlayerTrack:
    x: float
    y: float
    vx: float
    vy: float
    confidence: float
    source: str
    age_seconds: float
    missing_seconds: float
    valid: bool
    safe_to_control: bool

    def as_dict(self) -> dict:
        missing_seconds = (
            None
            if math.isinf(self.missing_seconds)
            else round(self.missing_seconds, 4)
        )
        return {
            "center_x": round(self.x, 3),
            "center_y": round(self.y, 3),
            "velocity_x": round(self.vx, 3),
            "velocity_y": round(self.vy, 3),
            "confidence": round(self.confidence, 4),
            "source": self.source,
            "age_seconds": round(self.age_seconds, 4),
            "missing_seconds": missing_seconds,
            "valid": self.valid,
            "safe_to_control": self.safe_to_control,
        }


class PlayerTracker:
    """指数平滑自机位置，并在很短的漏检窗口内做匀速预测。"""

    def __init__(
        self,
        smoothing: float = 0.65,
        prediction_timeout: float = 0.35,
        stop_timeout: float = 0.70,
        max_speed: float = 900.0,
    ):
        self.smoothing = clamp(smoothing, 0.0, 1.0)
        self.prediction_timeout = max(0.0, prediction_timeout)
        self.stop_timeout = max(self.prediction_timeout, stop_timeout)
        self.max_speed = max(1.0, max_speed)
        self.reset()

    def reset(self) -> None:
        self.x: Optional[float] = None
        self.y: Optional[float] = None
        self.vx = 0.0
        self.vy = 0.0
        self.started_at: Optional[float] = None
        self.last_update: Optional[float] = None
        self.last_seen: Optional[float] = None
        self.confidence = 0.0

    def update(
        self,
        detection: Optional[dict],
        width: int,
        height: int,
        now: Optional[float] = None,
    ) -> PlayerTrack:
        now = time.monotonic() if now is None else now
        if self.started_at is None:
            self.started_at = now
        previous_update = self.last_update
        self.last_update = now

        if detection is not None:
            measured_x = float(detection["center_x"])
            measured_y = float(detection["center_y"])
            if self.x is None or self.y is None or previous_update is None:
                self.x = measured_x
                self.y = measured_y
                self.vx = self.vy = 0.0
            else:
                dt = max(1e-3, now - previous_update)
                raw_vx = clamp(
                    (measured_x - self.x) / dt,
                    -self.max_speed,
                    self.max_speed,
                )
                raw_vy = clamp(
                    (measured_y - self.y) / dt,
                    -self.max_speed,
                    self.max_speed,
                )
                self.vx = (
                    self.smoothing * raw_vx
                    + (1.0 - self.smoothing) * self.vx
                )
                self.vy = (
                    self.smoothing * raw_vy
                    + (1.0 - self.smoothing) * self.vy
                )
                self.x = (
                    self.smoothing * measured_x
                    + (1.0 - self.smoothing) * self.x
                )
                self.y = (
                    self.smoothing * measured_y
                    + (1.0 - self.smoothing) * self.y
                )
            self.x = clamp(self.x, 0, max(0, width - 1))
            self.y = clamp(self.y, 0, max(0, height - 1))
            self.last_seen = now
            self.confidence = float(detection.get("confidence", 1.0))
            source = "detected"
            missing = 0.0
        elif self.x is not None and self.y is not None and self.last_seen is not None:
            missing = max(0.0, now - self.last_seen)
            dt = max(0.0, now - (previous_update or now))
            if missing <= self.prediction_timeout:
                self.x = clamp(self.x + self.vx * dt, 0, max(0, width - 1))
                self.y = clamp(self.y + self.vy * dt, 0, max(0, height - 1))
                source = "predicted"
            else:
                source = "stale"
                self.vx *= 0.5
                self.vy *= 0.5
        else:
            missing = float("inf")
            source = "missing"

        valid = (
            self.x is not None
            and self.y is not None
            and missing <= self.stop_timeout
        )
        safe_to_control = valid and missing <= self.prediction_timeout
        return PlayerTrack(
            x=float(self.x if self.x is not None else width / 2),
            y=float(self.y if self.y is not None else height * 0.8),
            vx=self.vx,
            vy=self.vy,
            confidence=self.confidence,
            source=source,
            age_seconds=max(0.0, now - (self.started_at or now)),
            missing_seconds=missing,
            valid=valid,
            safe_to_control=safe_to_control,
        )


@dataclass
class _BulletTrack:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    last_seen: float
    age_frames: int
    class_name: str


class BulletTracker:
    """使用最近邻匹配估算敌弹速度；所有输出都保持JSON可序列化。"""

    def __init__(
        self,
        max_match_distance: float = 55.0,
        stale_timeout: float = 0.35,
        velocity_smoothing: float = 0.55,
        max_speed: float = 1600.0,
    ):
        self.max_match_distance = max_match_distance
        self.stale_timeout = stale_timeout
        self.velocity_smoothing = clamp(velocity_smoothing, 0.0, 1.0)
        self.max_speed = max_speed
        self._tracks: dict[int, _BulletTrack] = {}
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(
        self,
        bullets: list[dict],
        now: Optional[float] = None,
    ) -> list[dict]:
        now = time.monotonic() if now is None else now
        available = set(self._tracks)
        results = []
        cell_size = max(16.0, self.max_match_distance)
        track_grid: dict[tuple[int, int], list[int]] = {}
        for track_id, track in self._tracks.items():
            dt = max(0.0, now - track.last_seen)
            predicted_x = track.x + track.vx * dt
            predicted_y = track.y + track.vy * dt
            cell = (
                math.floor(predicted_x / cell_size),
                math.floor(predicted_y / cell_size),
            )
            track_grid.setdefault(cell, []).append(track_id)

        for bullet in bullets:
            x = float(bullet["center_x"])
            y = float(bullet["center_y"])
            class_name = str(bullet.get("class_name", "bullet"))
            best_id = None
            best_distance = float("inf")
            cell_x = math.floor(x / cell_size)
            cell_y = math.floor(y / cell_size)
            nearby_ids = []
            for offset_x in range(-2, 3):
                for offset_y in range(-2, 3):
                    nearby_ids.extend(
                        track_grid.get(
                            (cell_x + offset_x, cell_y + offset_y),
                            (),
                        )
                    )
            for track_id in nearby_ids:
                if track_id not in available:
                    continue
                track = self._tracks[track_id]
                if track.class_name != class_name:
                    continue
                dt = max(0.0, now - track.last_seen)
                predicted_x = track.x + track.vx * dt
                predicted_y = track.y + track.vy * dt
                distance = math.hypot(x - predicted_x, y - predicted_y)
                dynamic_limit = self.max_match_distance + min(40.0, dt * 250)
                if distance <= dynamic_limit and distance < best_distance:
                    best_id = track_id
                    best_distance = distance

            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
                track = _BulletTrack(
                    track_id=best_id,
                    x=x,
                    y=y,
                    vx=0.0,
                    vy=0.0,
                    last_seen=now,
                    age_frames=1,
                    class_name=class_name,
                )
                self._tracks[best_id] = track
            else:
                available.remove(best_id)
                track = self._tracks[best_id]
                dt = max(1e-3, now - track.last_seen)
                raw_vx = clamp((x - track.x) / dt, -self.max_speed, self.max_speed)
                raw_vy = clamp((y - track.y) / dt, -self.max_speed, self.max_speed)
                alpha = self.velocity_smoothing
                track.vx = alpha * raw_vx + (1.0 - alpha) * track.vx
                track.vy = alpha * raw_vy + (1.0 - alpha) * track.vy
                track.x = x
                track.y = y
                track.last_seen = now
                track.age_frames += 1

            enriched = dict(bullet)
            enriched.update(
                {
                    "track_id": best_id,
                    "velocity_x": round(track.vx, 3),
                    "velocity_y": round(track.vy, 3),
                    "speed": round(math.hypot(track.vx, track.vy), 3),
                    "track_age_frames": track.age_frames,
                }
            )
            results.append(enriched)

        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if now - track.last_seen > self.stale_timeout
        ]
        for track_id in stale_ids:
            del self._tracks[track_id]
        return results


def collision_metrics(
    bullet: dict,
    player_x: float,
    player_y: float,
    horizon: float = 0.8,
    collision_radius: float = 22.0,
) -> dict:
    """计算匀速模型下最近接近时间、距离与0..1碰撞风险。"""
    rx = float(bullet["center_x"]) - player_x
    ry = float(bullet["center_y"]) - player_y
    vx = float(bullet.get("velocity_x", 0.0))
    vy = float(bullet.get("velocity_y", 0.0))
    speed_squared = vx * vx + vy * vy
    if speed_squared < 1.0:
        ttc = 0.0
        closest = math.hypot(rx, ry)
    else:
        ttc = clamp(-(rx * vx + ry * vy) / speed_squared, 0.0, horizon)
        closest = math.hypot(rx + vx * ttc, ry + vy * ttc)

    distance_factor = clamp(
        (collision_radius * 3.0 - closest) / (collision_radius * 3.0),
        0.0,
        1.0,
    )
    time_factor = 1.0 - (ttc / horizon if horizon > 0 else 1.0)
    risk = distance_factor * (0.35 + 0.65 * time_factor)
    if closest <= collision_radius:
        risk = max(risk, 0.85)
    return {
        "ttc_seconds": round(ttc, 4),
        "closest_distance": round(closest, 3),
        "collision_risk": round(clamp(risk, 0.0, 1.0), 4),
        "predicted_x": round(float(bullet["center_x"]) + vx * horizon, 3),
        "predicted_y": round(float(bullet["center_y"]) + vy * horizon, 3),
    }


class RiskPlanner:
    """比较候选方向未来短时间内的弹幕和边界代价。"""

    VECTORS = {
        "stay": (0.0, 0.0),
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
        "up": (0.0, -1.0),
        "down": (0.0, 1.0),
    }

    def __init__(
        self,
        safe_margin: int = 36,
        horizon: float = 0.55,
        player_speed: float = 170.0,
        allow_vertical: bool = True,
    ):
        self.safe_margin = max(8, safe_margin)
        self.horizon = max(0.1, horizon)
        self.player_speed = max(10.0, player_speed)
        self.allow_vertical = allow_vertical

    def candidate_costs(
        self,
        bullets: list[dict],
        player_x: float,
        player_y: float,
        width: int,
        height: int,
        preferred_x: Optional[float] = None,
    ) -> dict[str, float]:
        movements = ["stay", "left", "right"]
        if self.allow_vertical:
            movements.extend(["up", "down"])
        costs = {}
        for movement in movements:
            dx, dy = self.VECTORS[movement]
            target_x = player_x + dx * self.player_speed * self.horizon
            target_y = player_y + dy * self.player_speed * self.horizon
            boundary_cost = 0.0
            if target_x < self.safe_margin:
                boundary_cost += (self.safe_margin - target_x) * 8.0
            if target_x > width - self.safe_margin:
                boundary_cost += (target_x - (width - self.safe_margin)) * 8.0
            if target_y < self.safe_margin:
                boundary_cost += (self.safe_margin - target_y) * 8.0
            if target_y > height - self.safe_margin:
                boundary_cost += (target_y - (height - self.safe_margin)) * 8.0

            bullet_cost = 0.0
            for bullet in bullets:
                future_x = float(bullet["center_x"]) + float(
                    bullet.get("velocity_x", 0.0)
                ) * self.horizon
                future_y = float(bullet["center_y"]) + float(
                    bullet.get("velocity_y", 0.0)
                ) * self.horizon
                distance = math.hypot(future_x - target_x, future_y - target_y)
                if distance < 150:
                    bullet_cost += ((150.0 - distance) / 150.0) ** 2 * 100.0

            inertia_cost = 3.0 if movement != "stay" else 0.0
            aim_cost = (
                abs(target_x - preferred_x) * 0.015
                if preferred_x is not None
                else 0.0
            )
            costs[movement] = round(
                boundary_cost + bullet_cost + inertia_cost + aim_cost,
                4,
            )
        return costs

    def choose(
        self,
        bullets: list[dict],
        player_x: float,
        player_y: float,
        width: int,
        height: int,
        preferred_x: Optional[float] = None,
    ) -> tuple[str, dict[str, float]]:
        costs = self.candidate_costs(
            bullets,
            player_x,
            player_y,
            width,
            height,
            preferred_x,
        )
        order = ("stay", "left", "right", "up", "down")
        movement = min(costs, key=lambda item: (costs[item], order.index(item)))
        return movement, costs


class ActionStabilizer:
    """边界门控和确定性的方向切换冷却。"""

    OPPOSITE = {
        ("left", "right"),
        ("right", "left"),
        ("up", "down"),
        ("down", "up"),
    }

    def __init__(
        self,
        safe_margin: int = 36,
        min_hold_seconds: float = 0.16,
        reversal_cooldown: float = 0.28,
    ):
        self.safe_margin = safe_margin
        self.min_hold_seconds = min_hold_seconds
        self.reversal_cooldown = reversal_cooldown
        self.current = "stay"
        self.last_change: Optional[float] = None
        self.switch_count = 0
        self.blocked_count = 0

    def stabilize(
        self,
        proposed: str,
        player_x: float,
        player_y: float,
        width: int,
        height: int,
        safe_to_control: bool,
        now: Optional[float] = None,
    ) -> tuple[str, str]:
        now = time.monotonic() if now is None else now
        if proposed not in MOVEMENT_KEYS:
            proposed = "stay"
        reason = "accepted"
        if not safe_to_control:
            proposed = "stay"
            reason = "player_unavailable"
        elif proposed == "left" and player_x <= self.safe_margin:
            proposed, reason = "right", "left_boundary"
        elif proposed == "right" and player_x >= width - self.safe_margin:
            proposed, reason = "left", "right_boundary"
        elif proposed == "up" and player_y <= self.safe_margin:
            proposed, reason = "down", "top_boundary"
        elif proposed == "down" and player_y >= height - self.safe_margin:
            proposed, reason = "up", "bottom_boundary"

        if self.last_change is None:
            self.current = proposed
            self.last_change = now
            return proposed, reason
        if proposed == self.current:
            return proposed, reason

        elapsed = now - self.last_change
        required = (
            self.reversal_cooldown
            if (self.current, proposed) in self.OPPOSITE
            else self.min_hold_seconds
        )
        if elapsed < required and safe_to_control:
            self.blocked_count += 1
            return self.current, "switch_cooldown"

        self.current = proposed
        self.last_change = now
        self.switch_count += 1
        return proposed, reason


class GameSceneStateMachine:
    """保守场景状态机：只在连续战斗证据充分时允许输入。"""

    def __init__(
        self,
        battle_confirm_frames: int = 2,
        lost_confirm_frames: int = 4,
    ):
        self.battle_confirm_frames = battle_confirm_frames
        self.lost_confirm_frames = lost_confirm_frames
        self.state = "unknown"
        self._battle_evidence = 0
        self._lost_evidence = 0
        self.transitions = 0

    def update(
        self,
        *,
        player_detected: bool,
        player_safe: bool,
        bullet_count: int,
        enemy_count: int,
        player_bullet_count: int,
    ) -> str:
        battle_objects = bullet_count + enemy_count + player_bullet_count
        if player_detected and battle_objects > 0:
            self._battle_evidence += 1
            self._lost_evidence = 0
        elif self.state == "battle" and player_safe:
            # 已确认战斗后，短暂清屏或短时漏检不应误判为菜单。
            self._lost_evidence = 0
        else:
            self._battle_evidence = 0
            self._lost_evidence += 1

        new_state = self.state
        if self._battle_evidence >= self.battle_confirm_frames:
            new_state = "battle"
        elif self.state == "battle" and self._lost_evidence >= self.lost_confirm_frames:
            new_state = "transition"
        elif self.state == "unknown" and self._lost_evidence >= self.lost_confirm_frames:
            new_state = "non_battle"

        if new_state != self.state:
            self.state = new_state
            self.transitions += 1
        return self.state

    @property
    def action_allowed(self) -> bool:
        return self.state == "battle"


class InputStateMachine:
    """只发送按键状态差异，避免每帧释放并重按。"""

    def __init__(self, backend: Any):
        self.backend = backend
        self.held: set[str] = set()
        self.key_down_count = 0
        self.key_up_count = 0

    def apply(self, movement: str, shooting: bool, focused: bool) -> dict:
        desired = set(MOVEMENT_KEYS.get(movement, set()))
        if shooting:
            desired.add("z")
        if focused:
            desired.add("shift")

        released = sorted(self.held - desired)
        pressed = sorted(desired - self.held)
        for key in released:
            self.backend.keyUp(key)
            self.key_up_count += 1
        for key in pressed:
            self.backend.keyDown(key)
            self.key_down_count += 1
        self.held = desired
        return {"pressed": pressed, "released": released, "held": sorted(self.held)}

    def press_once(self, key: str) -> None:
        if key in self.held:
            self.backend.keyUp(key)
            self.held.remove(key)
            self.key_up_count += 1
        self.backend.press(key)

    def release_all(self) -> list[str]:
        released = sorted(self.held)
        for key in released:
            self.backend.keyUp(key)
            self.key_up_count += 1
        self.held.clear()
        return released


def summarize_scene_counts(events: list[dict]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for event in events:
        counts[str(event.get("state", {}).get("scene_state", "unknown"))] += 1
    return dict(counts)
