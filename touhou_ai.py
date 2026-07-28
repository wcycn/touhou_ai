#!/usr/bin/env python3
"""Touhou AI 发布版统一入口。"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
PROJECT_DIR = ROOT


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def run_check() -> int:
    required_files = {
        "自动控制核心": PROJECT_DIR / "autopilot.py",
        "输入诊断工具": PROJECT_DIR / "control_test.py",
        "独立游戏启动器": PROJECT_DIR / "launch_game.py",
        "统一桌面GUI": PROJECT_DIR / "desktop_gui.py",
        "会话记录模块": PROJECT_DIR / "session_recorder.py",
        "会话分析模块": PROJECT_DIR / "session_analysis.py",
        "跟踪与控制逻辑": PROJECT_DIR / "control_logic.py",
        "模型评估工具": PROJECT_DIR / "model_evaluation.py",
        "安全观察模式": PROJECT_DIR / "observe_game.py",
        "X11窗口定位器": PROJECT_DIR / "window_controller.py",
        "YOLO 模型": PROJECT_DIR / "models" / "best.pt",
    }
    required_modules = (
        "cv2",
        "numpy",
        "mss",
        "pyautogui",
        "PIL",
        "yaml",
        "tkinter",
        "torch",
        "ultralytics",
    )
    print("Touhou AI v1.1.0 发布版检查")
    print(f"Python: {sys.version.split()[0]}")
    print(f"项目目录: {PROJECT_DIR}")

    failed = False
    print("\n文件：")
    for label, path in required_files.items():
        ok = path.is_file()
        print(f"  {'OK' if ok else '缺失':<4} {label}: {path.relative_to(ROOT)}")
        failed = failed or not ok

    print("\nPython依赖：")
    for name in required_modules:
        ok = module_available(name)
        print(f"  {'OK' if ok else '缺失':<4} {name}")
        failed = failed or not ok

    if module_available("torch"):
        try:
            import torch

            from inference_device import select_inference_device

            inference_device = select_inference_device(torch)
            print(
                f"\nYOLO推理设备: {inference_device.label}"
            )
            if inference_device.reason:
                print(f"  说明: {inference_device.reason}")
        except Exception as exc:
            print(f"\nYOLO推理设备: 无法检查（{exc}）")
            failed = True

    print("\n外部环境：")
    wine_available = shutil.which("wine") is not None
    xdotool_available = shutil.which("xdotool") is not None
    xwininfo_available = shutil.which("xwininfo") is not None
    display_available = bool(os.environ.get("DISPLAY"))
    print(f"  {'OK' if wine_available else '缺失':<4} wine")
    print(f"  {'OK' if xdotool_available else '缺失':<4} xdotool")
    print(f"  {'OK' if xwininfo_available else '缺失':<4} xwininfo")
    print(f"  {'OK' if display_available else '未设置':<4} DISPLAY")
    failed = (
        failed
        or not wine_available
        or not xdotool_available
        or not xwininfo_available
        or not display_available
    )
    game_available = any(
        (PROJECT_DIR / "game" / name).is_file()
        for name in ("vpatch.exe", "th06c.exe", "th06.exe", "東方紅魔郷.exe")
    )
    print(
        f"  {'OK' if game_available else '未放置':<4} "
        "game/ 游戏文件（发布包不附带）"
    )

    if failed:
        print("\n结论：自动控制的必要文件或依赖不完整。")
        return 1

    conclusion = (
        "游戏资源已就绪。"
        if game_available
        else "请将合法取得的游戏文件放入 game/ 后再使用“启动游戏”。"
    )
    print(f"\n结论：源码与运行环境完整。{conclusion}")
    return 0


def call_project(command: list[str]) -> int:
    try:
        return subprocess.call(command, cwd=PROJECT_DIR)
    except KeyboardInterrupt:
        # GUI的安全停止会同时通知整个进程组；统一入口无需打印回溯。
        return 130


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="东方红魔乡自动操作项目的统一入口",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="检查文件、Python 依赖和外部环境")
    subparsers.add_parser("test", help="运行不发送按键的核心逻辑测试")
    subparsers.add_parser("gui", help="启动新的统一桌面控制中心")
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="分析最新、指定或全部运行会话",
    )
    analyze_parser.add_argument("session", nargs="?", type=Path)
    analyze_parser.add_argument("--all", action="store_true")
    analyze_parser.add_argument("--review-limit", type=int, default=100)
    analyze_parser.add_argument("--export-review", action="store_true")
    evaluate_parser = subparsers.add_parser(
        "model-eval",
        help="使用人工标注的YOLO数据集评估模型",
    )
    evaluate_parser.add_argument("data", type=Path)
    evaluate_parser.add_argument("--model", type=Path)
    evaluate_parser.add_argument("--confidence", type=float, default=0.25)
    evaluate_parser.add_argument("--iou", type=float, default=0.7)
    evaluate_parser.add_argument("--image-size", type=int, default=640)
    evaluate_parser.add_argument("--device", default="")
    game_parser = subparsers.add_parser("game", help="通过 Wine 启动游戏")
    game_parser.add_argument("--exe", help="显式指定游戏目录内的EXE")
    game_parser.add_argument(
        "--direct",
        action="store_true",
        help="跳过vpatch，直接启动游戏EXE",
    )
    game_parser.add_argument("--window-timeout", type=float, default=45.0)
    game_parser.add_argument("--wine-prefix", type=Path)
    subparsers.add_parser("locate", help="列出窗口候选并显示选中的游戏区域")
    control_parser = subparsers.add_parser(
        "control-test",
        help="聚焦游戏并短暂测试左右键，不启动AI",
    )
    control_parser.add_argument("--duration", type=float, default=0.35)
    control_parser.add_argument("--shoot", action="store_true")

    observe_parser = subparsers.add_parser(
        "observe",
        help="安全观察检测框和计划动作，绝不发送按键",
    )
    observe_parser.add_argument(
        "--mode",
        choices=("balanced", "aggressive", "defensive"),
        default="balanced",
    )
    observe_parser.add_argument("--confidence", type=float, default=0.35)
    observe_parser.add_argument("--sensitivity", type=float, default=0.05)
    observe_parser.add_argument("--interval", type=float, default=0.10)
    observe_parser.add_argument("--region", nargs=4, type=int)
    observe_parser.add_argument("--move-window", nargs=2, type=int)
    observe_parser.add_argument("--save-dir", type=Path)
    observe_parser.add_argument("--save-every", type=float, default=1.0)
    observe_parser.add_argument("--no-preview", action="store_true")
    observe_parser.add_argument("--frames", type=int, default=0)
    observe_parser.add_argument("--record", action="store_true")
    observe_parser.add_argument("--record-dir", type=Path)
    observe_parser.add_argument("--record-fps", type=float, default=2.0)
    observe_parser.add_argument("--safe-margin", type=int, default=36)
    observe_parser.add_argument("--player-lost-timeout", type=float, default=0.70)
    observe_parser.add_argument("--no-vertical", action="store_true")

    ai_parser = subparsers.add_parser("ai", help="游戏已运行后启动自动控制")
    ai_parser.add_argument(
        "--mode",
        choices=("balanced", "aggressive", "defensive"),
        default="balanced",
    )
    ai_parser.add_argument("--confidence", type=float, default=0.15)
    ai_parser.add_argument("--sensitivity", type=float, default=0.05)
    ai_parser.add_argument("--record", action="store_true")
    ai_parser.add_argument("--record-dir", type=Path)
    ai_parser.add_argument("--record-fps", type=float, default=2.0)
    ai_parser.add_argument("--safe-margin", type=int, default=36)
    ai_parser.add_argument("--player-lost-timeout", type=float, default=0.70)
    ai_parser.add_argument("--no-vertical", action="store_true")
    ai_parser.add_argument(
        "--allow-no-game",
        action="store_true",
        help="即使检测不到游戏进程也发送按键（有误操作风险）",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "check":
        return run_check()
    if args.command == "test":
        return subprocess.call(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
            cwd=ROOT,
        )
    if args.command == "gui":
        return call_project([sys.executable, "desktop_gui.py"])
    if args.command == "analyze":
        command = [
            sys.executable,
            "session_analysis.py",
            "--review-limit",
            str(args.review_limit),
        ]
        if args.all:
            command.append("--all")
        elif args.session:
            command.append(str(args.session.expanduser().resolve()))
        if args.export_review:
            command.append("--export-review")
        return call_project(command)
    if args.command == "model-eval":
        command = [
            sys.executable,
            "model_evaluation.py",
            str(args.data.expanduser().resolve()),
            "--confidence",
            str(args.confidence),
            "--iou",
            str(args.iou),
            "--image-size",
            str(args.image_size),
        ]
        if args.model:
            command.extend(["--model", str(args.model.expanduser().resolve())])
        if args.device:
            command.extend(["--device", args.device])
        return call_project(command)
    if args.command == "game":
        command = [
            sys.executable,
            "launch_game.py",
            "--window-timeout",
            str(args.window_timeout),
        ]
        if args.exe:
            command.extend(["--exe", args.exe])
        if args.direct:
            command.append("--direct")
        if args.wine_prefix:
            command.extend(["--wine-prefix", str(args.wine_prefix.expanduser())])
        return call_project(command)
    if args.command == "locate":
        return call_project([sys.executable, "window_controller.py"])
    if args.command == "control-test":
        command = [
            sys.executable,
            "control_test.py",
            "--duration",
            str(args.duration),
        ]
        if args.shoot:
            command.append("--shoot")
        return call_project(command)
    if args.command == "observe":
        command = [
            sys.executable,
            "observe_game.py",
            "--mode",
            args.mode,
            "--confidence",
            str(args.confidence),
            "--sensitivity",
            str(args.sensitivity),
            "--interval",
            str(args.interval),
            "--save-every",
            str(args.save_every),
            "--frames",
            str(args.frames),
            "--safe-margin",
            str(args.safe_margin),
            "--player-lost-timeout",
            str(args.player_lost_timeout),
        ]
        if args.region:
            command.extend(["--region", *(str(value) for value in args.region)])
        if args.move_window:
            command.extend(
                ["--move-window", *(str(value) for value in args.move_window)]
            )
        if args.save_dir:
            command.extend(["--save-dir", str(args.save_dir.resolve())])
        if args.no_preview:
            command.append("--no-preview")
        if args.record:
            command.append("--record")
        if args.no_vertical:
            command.append("--no-vertical")
        if args.record_dir:
            command.extend(["--record-dir", str(args.record_dir.resolve())])
        command.extend(["--record-fps", str(args.record_fps)])
        return call_project(command)
    if args.command == "ai":
        command = [
            sys.executable,
            "autopilot.py",
            "--mode",
            args.mode,
            "--confidence",
            str(args.confidence),
            "--sensitivity",
            str(args.sensitivity),
            "--safe-margin",
            str(args.safe_margin),
            "--player-lost-timeout",
            str(args.player_lost_timeout),
        ]
        if args.allow_no_game:
            command.append("--allow-no-game")
        if args.no_vertical:
            command.append("--no-vertical")
        if args.record:
            command.append("--record")
        if args.record_dir:
            command.extend(["--record-dir", str(args.record_dir.resolve())])
        command.extend(["--record-fps", str(args.record_fps)])
        return call_project(command)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
