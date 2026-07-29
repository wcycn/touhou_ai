#!/usr/bin/env python3
"""Build an annotated README demo from a recorded Touhou AI session."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "touhou-ai-demo.gif"
PLAYFIELD_RIGHT = 410

COLORS = {
    "background": (23, 24, 36, 238),
    "panel": (28, 30, 45, 246),
    "text": (240, 241, 247),
    "muted": (172, 176, 197),
    "accent": (235, 71, 111),
    "safe": (105, 127, 244),
    "success": (55, 211, 153),
    "warning": (245, 177, 66),
    "danger": (255, 82, 106),
    "power": (73, 205, 255),
}

DIRECTION_VECTORS = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, -1),
    "down": (0, 1),
    "left_up": (-1, -1),
    "left_down": (-1, 1),
    "right_up": (1, -1),
    "right_down": (1, 1),
}


def load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / name,
        Path("/usr/share/fonts/dejavu") / name,
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_SMALL = load_font(11)
FONT_BODY = load_font(13)
FONT_BODY_BOLD = load_font(13, bold=True)
FONT_TITLE = load_font(20, bold=True)


def load_saved_events(
    session_dir: Path,
    *,
    start: float,
    duration: float,
) -> list[dict]:
    events_path = session_dir / "events.jsonl"
    if not events_path.is_file():
        raise FileNotFoundError(f"Missing session events: {events_path}")
    selected = []
    with events_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            elapsed = float(event.get("elapsed_seconds", 0.0))
            if (
                event.get("frame_file")
                and start <= elapsed <= start + duration
            ):
                frame_path = session_dir / event["frame_file"]
                if frame_path.is_file():
                    event["_frame_path"] = frame_path
                    selected.append(event)
    if not selected:
        raise ValueError("No saved frames matched the requested time range")
    return selected


def detection_color(class_name: str) -> tuple[int, int, int] | None:
    name = class_name.casefold()
    if "bullet_enemy" in name:
        return COLORS["danger"]
    if "power" in name or "item" in name:
        return COLORS["power"]
    if "boss" in name or ("enemy" in name and "bullet" not in name):
        return COLORS["warning"]
    return None


def draw_box(
    draw: ImageDraw.ImageDraw,
    bbox: list[float],
    color: tuple[int, int, int],
    *,
    width: int = 2,
) -> None:
    if len(bbox) != 4:
        return
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=width)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    movement: str,
) -> None:
    vector = DIRECTION_VECTORS.get(movement)
    if not vector:
        return
    vx, vy = vector
    length = math.hypot(vx, vy)
    vx, vy = vx / length, vy / length
    x1, y1 = start
    x2, y2 = x1 + vx * 42, y1 + vy * 42
    color = COLORS["success"]
    draw.line((x1, y1, x2, y2), fill=color, width=4)
    angle = math.atan2(y2 - y1, x2 - x1)
    head = []
    for offset in (2.55, -2.55):
        head.append(
            (
                x2 + math.cos(angle + offset) * 11,
                y2 + math.sin(angle + offset) * 11,
            )
        )
    draw.polygon(((x2, y2), head[0], head[1]), fill=color)


def risk_color(risk: float) -> tuple[int, int, int]:
    if risk >= 0.65:
        return COLORS["danger"]
    if risk >= 0.25:
        return COLORS["warning"]
    return COLORS["success"]


def annotate_frame(event: dict, relative_start: float) -> Image.Image:
    frame = Image.open(event["_frame_path"]).convert("RGB")
    overlay = Image.new("RGBA", frame.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(frame)

    state = event.get("state", {})
    action = event.get("action", {})
    risk = float(state.get("max_collision_risk", 0.0) or 0.0)
    movement = str(action.get("movement", "stay"))

    for detection in event.get("detections", []):
        color = detection_color(str(detection.get("class_name", "")))
        if color:
            draw_box(draw, detection.get("bbox", []), color)

    player = state.get("player")
    if isinstance(player, dict):
        draw_box(
            draw,
            player.get("bbox", []),
            COLORS["success"],
            width=3,
        )

    for bullet in state.get("bullet_tracks", []):
        start = (
            float(bullet.get("center_x", 0.0)),
            float(bullet.get("center_y", 0.0)),
        )
        end = (
            max(0.0, min(PLAYFIELD_RIGHT, float(bullet.get("predicted_x", 0.0)))),
            max(0.0, min(frame.height - 1, float(bullet.get("predicted_y", 0.0)))),
        )
        track_risk = float(bullet.get("collision_risk", 0.0) or 0.0)
        draw.line((*start, *end), fill=risk_color(track_risk), width=2)
        radius = 3
        draw.ellipse(
            (
                start[0] - radius,
                start[1] - radius,
                start[0] + radius,
                start[1] + radius,
            ),
            fill=risk_color(track_risk),
        )

    tracked = state.get("tracked_player") or {}
    if tracked.get("valid"):
        draw_arrow(
            draw,
            (
                float(tracked.get("center_x", 0.0)),
                float(tracked.get("center_y", 0.0)),
            ),
            movement,
        )

    overlay_draw.rectangle(
        (0, 0, PLAYFIELD_RIGHT, 43),
        fill=COLORS["background"],
    )
    overlay_draw.text(
        (12, 7),
        "TOUHOU AI",
        font=FONT_BODY_BOLD,
        fill=COLORS["text"],
    )
    mode_color = COLORS["danger"] if action.get("executed") else COLORS["safe"]
    mode_text = "CONTROL MODE" if action.get("executed") else "SAFE OBSERVATION"
    overlay_draw.text(
        (111, 7),
        f"● {mode_text}",
        font=FONT_BODY_BOLD,
        fill=mode_color,
    )
    elapsed = float(event.get("elapsed_seconds", 0.0)) - relative_start
    overlay_draw.text(
        (12, 25),
        f"Recorded inference · demo {elapsed:04.1f}s",
        font=FONT_SMALL,
        fill=COLORS["muted"],
    )

    overlay_draw.rectangle(
        (PLAYFIELD_RIGHT, 0, frame.width, frame.height),
        fill=COLORS["panel"],
    )
    panel_x = PLAYFIELD_RIGHT + 16
    overlay_draw.text(
        (panel_x, 18),
        "LIVE DECISION",
        font=FONT_TITLE,
        fill=COLORS["text"],
    )
    overlay_draw.text(
        (panel_x, 46),
        "vision → tracking → planning",
        font=FONT_SMALL,
        fill=COLORS["muted"],
    )

    y = 85
    rows = (
        ("SCENE", str(state.get("scene_state", "unknown")).upper()),
        ("ACTION", movement.replace("_", " ").upper()),
        ("PLAYER", str(state.get("player_source", "missing")).upper()),
        ("BULLETS", str(state.get("bullet_count", 0))),
        ("POWERUPS", str(state.get("powerup_count", 0))),
    )
    for label, value in rows:
        overlay_draw.text(
            (panel_x, y),
            label,
            font=FONT_SMALL,
            fill=COLORS["muted"],
        )
        overlay_draw.text(
            (panel_x + 78, y - 1),
            value,
            font=FONT_BODY_BOLD,
            fill=COLORS["text"],
        )
        y += 30

    overlay_draw.text(
        (panel_x, y + 5),
        "COLLISION RISK",
        font=FONT_SMALL,
        fill=COLORS["muted"],
    )
    overlay_draw.text(
        (frame.width - 56, y + 3),
        f"{risk:.2f}",
        font=FONT_BODY_BOLD,
        fill=risk_color(risk),
    )
    bar_y = y + 27
    bar_width = frame.width - panel_x - 18
    overlay_draw.rounded_rectangle(
        (panel_x, bar_y, panel_x + bar_width, bar_y + 10),
        radius=5,
        fill=(67, 70, 91, 255),
    )
    overlay_draw.rounded_rectangle(
        (
            panel_x,
            bar_y,
            panel_x + max(4, int(bar_width * min(1.0, risk))),
            bar_y + 10,
        ),
        radius=5,
        fill=(*risk_color(risk), 255),
    )

    y = bar_y + 36
    held = action.get("input_transition", {}).get("held", [])
    held_text = " + ".join(str(key).upper() for key in held) or "NONE"
    overlay_draw.text(
        (panel_x, y),
        "KEYS HELD",
        font=FONT_SMALL,
        fill=COLORS["muted"],
    )
    overlay_draw.text(
        (panel_x, y + 18),
        held_text,
        font=FONT_BODY_BOLD,
        fill=COLORS["success"],
    )

    y += 68
    legend = (
        (COLORS["success"], "Player / planned move"),
        (COLORS["danger"], "Enemy bullet / risk"),
        (COLORS["warning"], "Enemy / boss"),
        (COLORS["power"], "Power item"),
    )
    for color, label in legend:
        overlay_draw.rectangle(
            (panel_x, y + 3, panel_x + 10, y + 13),
            fill=(*color, 255),
        )
        overlay_draw.text(
            (panel_x + 17, y),
            label,
            font=FONT_SMALL,
            fill=COLORS["muted"],
        )
        y += 22

    overlay_draw.text(
        (panel_x, frame.height - 36),
        "v1.1.0 · EXPERIMENTAL",
        font=FONT_SMALL,
        fill=COLORS["accent"],
    )
    overlay_draw.text(
        (panel_x, frame.height - 20),
        "unofficial fan-made project",
        font=FONT_SMALL,
        fill=COLORS["muted"],
    )

    return Image.alpha_composite(frame.convert("RGBA"), overlay).convert("RGB")


def encode_gif(frame_dir: Path, output: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to encode the demo GIF")
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        "split[s0][s1];"
        "[s0]palettegen=max_colors=128:stats_mode=diff[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
    )
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "%04d.png"),
            "-filter_complex",
            filter_graph,
            "-loop",
            "0",
            str(output),
        ],
        check=True,
    )


def build_demo(
    session_dir: Path,
    output: Path,
    *,
    start: float,
    duration: float,
    fps: float,
) -> Path:
    session_dir = session_dir.expanduser().resolve()
    events = load_saved_events(session_dir, start=start, duration=duration)
    with tempfile.TemporaryDirectory(prefix="touhou-ai-demo-") as temporary:
        frame_dir = Path(temporary)
        for index, event in enumerate(events):
            frame = annotate_frame(event, start)
            frame.save(frame_dir / f"{index:04d}.png")
        encode_gif(frame_dir, output, fps)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build an annotated GIF from a recorded Touhou AI session"
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start", type=float, default=54.5)
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--fps", type=float, default=3.0)
    args = parser.parse_args()
    if args.duration <= 0 or args.fps <= 0:
        parser.error("--duration and --fps must be positive")
    try:
        output = build_demo(
            args.session,
            args.output.expanduser().resolve(),
            start=args.start,
            duration=args.duration,
            fps=args.fps,
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Demo build failed: {exc}")
        return 1
    print(f"Demo created: {output}")
    print(f"Size: {output.stat().st_size / 1_000_000:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
