#!/usr/bin/env python3
"""通过 Wine 独立启动东方红魔乡，并等待 X11 游戏窗口出现。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from window_controller import WindowController, format_candidates


PROJECT_DIR = Path(__file__).resolve().parent
GAME_DIR = PROJECT_DIR / "game"
DEFAULT_EXECUTABLES = ("vpatch.exe", "th06c.exe", "th06.exe", "東方紅魔郷.exe")


def select_executable(
    game_dir: Path,
    requested: str | None = None,
    direct: bool = False,
) -> Path:
    """选择启动文件；默认遵循游戏包内说明优先使用vpatch。"""
    if requested:
        if Path(requested).name != requested:
            raise ValueError("--exe 只能填写游戏目录内的文件名")
        path = game_dir / requested
        if not path.is_file():
            raise FileNotFoundError(f"启动文件不存在: {path}")
        return path

    order = DEFAULT_EXECUTABLES[1:] if direct else DEFAULT_EXECUTABLES
    for name in order:
        path = game_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"{game_dir} 中没有可用的游戏启动文件")


def tail_text(path: Path, line_count: int = 20) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-line_count:])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="通过Wine独立启动东方红魔乡")
    parser.add_argument("--exe", help="显式选择游戏目录内的EXE文件")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="跳过vpatch，直接选择th06c.exe/th06.exe",
    )
    parser.add_argument(
        "--window-timeout",
        type=float,
        default=45.0,
        help="等待游戏窗口出现的秒数",
    )
    parser.add_argument(
        "--wine-prefix",
        type=Path,
        default=Path(
            os.environ.get("TOUHOU_WINEPREFIX", str(Path.home() / ".wine32"))
        ),
        help="Wine前缀目录",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not shutil.which("wine"):
        print("❌ 找不到 wine，请先安装Wine。")
        return 1
    if not os.environ.get("DISPLAY"):
        print("❌ DISPLAY 未设置，Wine无法创建游戏窗口。")
        return 1

    try:
        executable = select_executable(
            GAME_DIR,
            requested=args.exe,
            direct=args.direct,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"❌ {exc}")
        return 1

    wine_prefix = args.wine_prefix.expanduser()
    log_dir = PROJECT_DIR / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "wine_game.log"

    environment = os.environ.copy()
    environment["WINEPREFIX"] = str(wine_prefix)
    if not wine_prefix.exists():
        environment["WINEARCH"] = "win32"

    command = ["wine", executable.name]
    print("🎮 正在启动东方红魔乡")
    print(f"📂 游戏目录: {GAME_DIR}")
    print(f"🚀 启动文件: {executable.name}")
    print(f"🍷 Wine前缀: {wine_prefix}")
    print(f"📝 Wine日志: {log_path}")

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=GAME_DIR,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            print(f"✅ Wine进程已创建，PID={process.pid}")
            print("⏳ 正在等待游戏窗口；首次创建Wine前缀可能需要较长时间...")

            controller = WindowController()
            deadline = time.monotonic() + max(1.0, args.window_timeout)
            while time.monotonic() < deadline:
                if controller.find_game_window():
                    info = controller.window_info
                    print("✅ 已找到游戏窗口")
                    print(f"🪟 标题: {info.title!r}")
                    print(f"🎯 截图区域: {info.region}")
                    print("\n下一步运行: python3 touhou_ai.py observe")
                    return 0

                return_code = process.poll()
                if return_code not in (None, 0):
                    print(f"❌ Wine进程提前退出，退出码={return_code}")
                    break
                time.sleep(0.5)
    except OSError as exc:
        print(f"❌ 无法启动游戏: {exc}")
        return 1

    print("⚠️ 在等待时间内没有识别到游戏窗口。")
    candidates = WindowController().list_candidates()
    if candidates:
        print("当前窗口候选：")
        print(format_candidates(candidates[:10]))
    log_tail = tail_text(log_path)
    if log_tail:
        print("\nWine日志末尾：")
        print(log_tail)
    print(f"\n完整日志: {log_path}")
    print("如果游戏窗口其实已经出现，请运行: python3 touhou_ai.py locate")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
