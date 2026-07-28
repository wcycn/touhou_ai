#!/usr/bin/env python3
"""安全观察模式：检测、标框和预览决策，但绝不发送键盘事件。"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time
import types

import cv2
import mss
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent


class TkPreview:
    """使用Tk/Pillow显示画面，不依赖OpenCV HighGUI编译选项。"""

    def __init__(self):
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except ImportError as exc:
            raise RuntimeError(f"Tkinter/Pillow不可用: {exc}") from exc

        self._tk = tk
        self._Image = Image
        self._ImageTk = ImageTk
        self.closed = False
        self._photo = None

        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            raise RuntimeError(f"无法连接图形桌面: {exc}") from exc
        self.root.title("Touhou AI - SAFE Observe")
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Key-q>", lambda _event: self.close())
        self.root.bind("<Key-Q>", lambda _event: self.close())
        self.root.bind("<Escape>", lambda _event: self.close())
        self.label = tk.Label(self.root, bg="black")
        self.label.pack(fill="both", expand=True)
        self.root.update_idletasks()

    def show(self, frame: np.ndarray) -> bool:
        if self.closed:
            return False
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self._Image.fromarray(rgb)
            image.thumbnail((1100, 800), self._Image.Resampling.LANCZOS)
            self._photo = self._ImageTk.PhotoImage(image=image)
            self.label.configure(image=self._photo)
            self.root.update_idletasks()
            self.root.update()
            return not self.closed
        except self._tk.TclError:
            self.closed = True
            return False

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.root.destroy()
        except self._tk.TclError:
            pass


class DisabledPyAutoGUI(types.ModuleType):
    """阻止观察模式意外调用任何自动化输入函数。"""

    def __getattr__(self, name):
        if name in {
            "keyDown",
            "keyUp",
            "press",
            "click",
            "moveTo",
            "write",
            "hotkey",
        }:
            def blocked(*_args, **_kwargs):
                raise RuntimeError("观察模式禁止发送键盘或鼠标事件")

            return blocked
        raise AttributeError(name)


# 必须在延迟导入旧核心之前替换；旧核心在模块顶层导入 PyAutoGUI。
sys.modules["pyautogui"] = DisabledPyAutoGUI("pyautogui")
os.environ["YOLO_CONFIG_DIR"] = str(PROJECT_DIR / "runs" / "ultralytics")

from window_controller import WindowController  # noqa: E402


COLORS = {
    "player": (60, 220, 60),
    "enemy_bullet": (40, 40, 240),
    "player_bullet": (255, 220, 40),
    "enemy": (20, 150, 255),
    "powerup": (255, 100, 40),
    "other": (200, 200, 200),
}


def detection_group(class_name: str) -> str:
    name = class_name.casefold()
    if name == "character" or ("player" in name and "bullet" not in name):
        return "player"
    if "bullet_player" in name:
        return "player_bullet"
    if "bullet_enemy" in name or name == "bullet":
        return "enemy_bullet"
    if "enemy" in name or "boss" in name:
        return "enemy"
    if "power" in name or "item" in name:
        return "powerup"
    return "other"


def annotate(
    frame: np.ndarray,
    detections: list[dict],
    action_text: str,
    region: dict,
    game_state: dict | None = None,
) -> np.ndarray:
    output = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(value) for value in detection["bbox"])
        group = detection_group(detection["class_name"])
        color = COLORS[group]
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{detection['class_name']} "
            f"{detection['confidence']:.2f}"
        )
        cv2.putText(
            output,
            label,
            (max(0, x1), max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    if game_state:
        for bullet in game_state.get("bullets", []):
            start = (
                int(bullet.get("center_x", 0)),
                int(bullet.get("center_y", 0)),
            )
            end = (
                int(bullet.get("predicted_x", start[0])),
                int(bullet.get("predicted_y", start[1])),
            )
            risk = float(bullet.get("collision_risk", 0.0))
            color = (0, 0, 255) if risk >= 0.65 else (80, 180, 255)
            cv2.arrowedLine(output, start, end, color, 1, tipLength=0.25)

        tracked = game_state.get("tracked_player") or {}
        if tracked.get("source") == "predicted":
            center = (
                int(tracked.get("center_x", 0)),
                int(tracked.get("center_y", 0)),
            )
            cv2.circle(output, center, 10, (255, 0, 255), 2)

    status = f"SAFE OBSERVE | {action_text} | objects={len(detections)}"
    region_text = (
        f"capture={region['left']},{region['top']} "
        f"{region['width']}x{region['height']}"
    )
    cv2.rectangle(output, (0, 0), (output.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(
        output,
        status,
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        region_text,
        (8, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (190, 190, 190),
        1,
        cv2.LINE_AA,
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只观察游戏画面和检测结果，不发送任何按键",
    )
    parser.add_argument(
        "--mode",
        choices=("balanced", "aggressive", "defensive"),
        default="balanced",
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--sensitivity", type=float, default=0.05)
    parser.add_argument("--interval", type=float, default=0.10)
    parser.add_argument(
        "--region",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        help="跳过自动定位，手动指定截图区域",
    )
    parser.add_argument(
        "--move-window",
        nargs=2,
        type=int,
        metavar=("X", "Y"),
        help="找到窗口后显式移动到指定坐标，不改变大小",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        help="保存标框画面；默认只显示，不写文件",
    )
    parser.add_argument(
        "--save-every",
        type=float,
        default=1.0,
        help="保存间隔秒数",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="不打开 OpenCV 预览窗口，需配合 --save-dir",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="处理指定帧数后退出；0 表示持续运行",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="记录检测、计划动作和抽样画面",
    )
    parser.add_argument("--record-dir", type=Path, help="会话记录根目录")
    parser.add_argument("--record-fps", type=float, default=2.0)
    parser.add_argument("--safe-margin", type=int, default=36)
    parser.add_argument("--player-lost-timeout", type=float, default=0.70)
    parser.add_argument("--no-vertical", action="store_true")
    return parser.parse_args()


def resolve_region(
    args: argparse.Namespace,
    controller: WindowController,
) -> dict:
    if args.region:
        left, top, width, height = args.region
        geometry = (left, top, width, height)
        if not controller.valid_geometry(geometry):
            raise ValueError(f"手动截图区域不合理: {geometry}")
        return {
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        }

    if not controller.find_game_window(timeout=2.0):
        raise RuntimeError(
            "没有找到游戏窗口。请先以窗口模式启动游戏，"
            "或使用 --region LEFT TOP WIDTH HEIGHT。"
        )
    if args.move_window and not controller.move_window(*args.move_window):
        raise RuntimeError("找到了游戏窗口，但移动窗口失败")
    region_tuple = controller.get_screenshot_region()
    if not region_tuple:
        raise RuntimeError("游戏窗口在读取坐标时消失")
    left, top, width, height = region_tuple
    return {
        "left": left,
        "top": top,
        "width": width,
        "height": height,
    }


def main() -> int:
    args = parse_args()
    if args.no_preview and not args.save_dir and not args.frames:
        print("❌ --no-preview 至少应配合 --save-dir 或 --frames")
        return 2

    controller = WindowController()
    try:
        region = resolve_region(args, controller)
    except (RuntimeError, ValueError) as exc:
        print(f"❌ {exc}")
        print("💡 可运行 python3 window_controller.py 查看候选窗口。")
        return 1

    # 定位成功后才加载较重的AI依赖，避免“未启动游戏”时输出无关环境日志。
    from autopilot import TouhouAIController

    ai = TouhouAIController(
        allow_no_game=True,
        safe_margin=args.safe_margin,
        player_lost_timeout=args.player_lost_timeout,
        allow_vertical=not args.no_vertical,
    )
    ai.ai_mode = args.mode
    ai.confidence_threshold = args.confidence
    ai.bullet_threshold = args.sensitivity
    ai.screen_region = region
    # 即使未来恢复了高级按键模块，观察模式也不保留控制器实例。
    if not ai.load_model():
        print("❌ 模型加载失败")
        return 1

    recorder = None
    if args.record:
        from session_recorder import SessionRecorder

        try:
            recorder = SessionRecorder(
                base_dir=args.record_dir or (PROJECT_DIR / "sessions"),
                frame_sample_fps=args.record_fps,
                source="observe",
                config={
                    "mode": args.mode,
                    "confidence": args.confidence,
                    "sensitivity": args.sensitivity,
                    "screen_region": region,
                },
            )
            print(f"📝 会话记录: {recorder.session_dir}")
        except Exception as exc:
            print(f"❌ 无法创建会话记录: {exc}")
            return 1

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    preview = None
    if not args.no_preview:
        try:
            preview = TkPreview()
        except RuntimeError as exc:
            print(f"⚠️ 无法打开实时预览: {exc}")
            if not args.save_dir:
                args.save_dir = PROJECT_DIR / "runs" / "observe"
                args.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"📸 将改为保存标框截图: {args.save_dir}")

    print("👁️ 安全观察模式已启动：不会发送任何键盘或鼠标事件")
    print(f"🎯 当前截图区域: {region}")
    if controller.window_info:
        print(
            f"🪟 游戏窗口: {controller.window_info.title!r} "
            f"id={controller.window_info.window_id}"
        )
    if preview:
        print("按预览窗口中的 Q 或 ESC 退出，也可在终端按 Ctrl+C。")
    else:
        print("当前为截图保存模式；在终端按 Ctrl+C 退出。")

    frame_count = 0
    last_refresh = 0.0
    last_save = 0.0
    try:
        with mss.mss() as capture:
            while True:
                loop_started = time.monotonic()
                if not args.region and loop_started - last_refresh >= 1.0:
                    if controller.refresh():
                        current = controller.get_screenshot_region()
                        if current:
                            left, top, width, height = current
                            region = {
                                "left": left,
                                "top": top,
                                "width": width,
                                "height": height,
                            }
                            ai.screen_region = region
                    last_refresh = loop_started

                raw = np.asarray(capture.grab(region))
                frame = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
                detections = ai.detect_objects(frame)
                state = ai.analyze_game_state(detections)
                movement, shooting, focused = ai.make_decision(state)
                if movement == "bomb":
                    action_text = "planned=BOMB"
                else:
                    action_text = (
                        f"planned=move:{movement} shoot:{bool(shooting)} "
                        f"focus:{bool(focused)} "
                        f"scene:{state['scene_state']} "
                        f"risk:{state['max_collision_risk']:.2f} "
                        f"player:{state['player_source']}"
                    )
                marked = annotate(
                    frame,
                    detections,
                    action_text,
                    region,
                    state,
                )
                frame_count += 1

                if recorder:
                    recorder.record(
                        frame=frame,
                        detections=detections,
                        game_state=state,
                        action={
                            "movement": movement,
                            "shooting": bool(shooting),
                            "focused": bool(focused),
                            "executed": False,
                            "focus_ok": None,
                            "reason": "observe_only",
                            "decision_reason": state.get("decision_reason"),
                            "stabilization_reason": state.get(
                                "stabilization_reason"
                            ),
                        },
                        screen_region=region,
                    )

                now = time.monotonic()
                if args.save_dir and now - last_save >= args.save_every:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    path = args.save_dir / f"observe_{timestamp}.jpg"
                    cv2.imwrite(str(path), marked)
                    cv2.imwrite(str(args.save_dir / "latest.jpg"), marked)
                    last_save = now

                if preview:
                    if not preview.show(marked):
                        break
                if args.frames and frame_count >= args.frames:
                    break

                remaining = args.interval - (time.monotonic() - loop_started)
                if remaining > 0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"❌ 观察模式运行失败: {exc}")
        return 1
    finally:
        if preview:
            preview.close()
        if recorder:
            recorder.close(
                end_reason="observe_ended",
                final_stats={"frame_count": frame_count},
            )
            print(f"📝 会话记录已完成: {recorder.session_dir}")

    print(f"✅ 观察结束，共处理 {frame_count} 帧；没有发送按键")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
