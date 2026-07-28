#!/usr/bin/env python3
"""低风险输入诊断：聚焦游戏后短暂按左、右方向键。"""

from __future__ import annotations

import argparse
import time

import pyautogui

from window_controller import WindowController


TEST_KEYS = ("left", "right")
ALL_CONTROL_KEYS = ("left", "right", "up", "down", "z", "x", "shift")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="聚焦游戏窗口并短暂测试左右移动，不启动AI",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.35,
        help="每个方向按住的秒数，范围0.05-1.0",
    )
    parser.add_argument(
        "--shoot",
        action="store_true",
        help="额外短按一次Z；可能会在菜单中执行确认",
    )
    return parser


def release_keys():
    for key in ALL_CONTROL_KEYS:
        try:
            pyautogui.keyUp(key)
        except Exception:
            pass


def main() -> int:
    args = build_parser().parse_args()
    if not 0.05 <= args.duration <= 1.0:
        print("❌ --duration 必须在0.05到1.0秒之间")
        return 2

    controller = WindowController()
    if not controller.find_game_window(timeout=2.0):
        print("❌ 没有找到游戏窗口，请先运行 python3 touhou_ai.py game")
        return 1
    if not controller.activate_window():
        print("❌ 找到了游戏窗口，但无法切换输入焦点")
        return 1

    info = controller.window_info
    print(f"🎯 已聚焦游戏窗口: {info.title!r} id={info.window_id}")
    print("🧪 即将短暂按下：左 → 右")

    pyautogui.PAUSE = 0.01
    pyautogui.FAILSAFE = True
    release_keys()
    try:
        for key in TEST_KEYS:
            if not controller.is_game_window_active():
                print("❌ 测试期间焦点离开游戏，已停止")
                return 1
            print(f"  按下 {key}")
            pyautogui.keyDown(key)
            time.sleep(args.duration)
            pyautogui.keyUp(key)
            time.sleep(0.20)

        if args.shoot:
            if not controller.is_game_window_active():
                print("❌ 射击测试前焦点离开游戏，已停止")
                return 1
            print("  短按 z")
            pyautogui.press("z")
    finally:
        release_keys()

    print("✅ 输入测试结束。若自机/菜单完全没有反应，请把此输出告诉我。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
