#!/usr/bin/env python3
"""通过 X11 工具定位 Wine 中的东方红魔乡窗口。

模块只读取窗口信息；除非显式调用 ``move_window``，不会移动、缩放或激活窗口。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import subprocess
import time
from typing import Callable, Iterable, Optional, Sequence


CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess]


@dataclass(frozen=True)
class WindowInfo:
    window_id: str
    title: str
    window_class: str
    x: int
    y: int
    width: int
    height: int
    pid: Optional[int] = None
    score: int = 0

    @property
    def region(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


class WindowController:
    """查找、验证并跟踪游戏客户端区域。"""

    TITLE_MARKERS = (
        "東方紅魔郷",
        "东方红魔乡",
        "touhou koumakyou",
        "embodiment of scarlet devil",
        "th06",
        "vpatch",
    )
    EXCLUDED_TITLE_MARKERS = (
        "touhou ai",
        "project_status",
        "window_controller.py",
    )

    def __init__(
        self,
        command_runner: Optional[CommandRunner] = None,
        minimum_score: int = 55,
    ):
        self._command_runner = command_runner or self._default_run
        self.minimum_score = minimum_score
        self.window_info: Optional[WindowInfo] = None

    @staticmethod
    def _default_run(command: Sequence[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )

    def _run(self, command: Sequence[str]) -> Optional[subprocess.CompletedProcess]:
        try:
            return self._command_runner(command)
        except (FileNotFoundError, subprocess.SubprocessError, OSError):
            return None

    @staticmethod
    def parse_xwininfo(output: str) -> Optional[tuple[int, int, int, int]]:
        """解析 ``xwininfo -id`` 的客户端绝对坐标。"""
        patterns = {
            "x": r"Absolute upper-left X:\s*(-?\d+)",
            "y": r"Absolute upper-left Y:\s*(-?\d+)",
            "width": r"Width:\s*(\d+)",
            "height": r"Height:\s*(\d+)",
        }
        values: dict[str, int] = {}
        for name, pattern in patterns.items():
            match = re.search(pattern, output)
            if not match:
                return None
            values[name] = int(match.group(1))
        return values["x"], values["y"], values["width"], values["height"]

    @staticmethod
    def parse_xdotool_geometry(output: str) -> Optional[tuple[int, int, int, int]]:
        """解析 ``xdotool getwindowgeometry --shell`` 输出。"""
        values: dict[str, int] = {}
        for name in ("X", "Y", "WIDTH", "HEIGHT"):
            match = re.search(rf"^{name}=(-?\d+)$", output, re.MULTILINE)
            if not match:
                return None
            values[name] = int(match.group(1))
        return values["X"], values["Y"], values["WIDTH"], values["HEIGHT"]

    @staticmethod
    def valid_geometry(geometry: tuple[int, int, int, int]) -> bool:
        _, _, width, height = geometry
        if width < 320 or height < 240:
            return False
        if width > 4096 or height > 4096:
            return False
        # 窗口模式通常是4:3；全屏模式可能由Wine/vpatch扩展到16:9。
        aspect_ratio = width / height
        return 1.0 <= aspect_ratio <= 2.0

    @classmethod
    def score_candidate(
        cls,
        title: str,
        window_class: str,
        geometry: tuple[int, int, int, int],
    ) -> int:
        title_lower = title.casefold()
        class_lower = window_class.casefold()
        if any(marker in title_lower for marker in cls.EXCLUDED_TITLE_MARKERS):
            return -100

        score = 0
        for marker in cls.TITLE_MARKERS:
            if marker.casefold() in title_lower:
                score += 80 if marker in cls.TITLE_MARKERS[:4] else 55
                break

        if any(marker in class_lower for marker in ("th06", "vpatch")):
            score += 45
        elif "wine" in class_lower:
            score += 10
        if any(
            marker in class_lower
            for marker in ("terminal", "konsole", "xterm", "code")
        ):
            score -= 60

        _, _, width, height = geometry
        ratio_error = abs((width / height) - (4 / 3))
        if ratio_error < 0.03:
            score += 20
        elif ratio_error < 0.12:
            score += 10

        # 640×480 及其等比缩放是这个游戏最常见的窗口尺寸。
        scale_x = width / 640
        scale_y = height / 480
        if abs(scale_x - scale_y) < 0.08:
            score += 10
        return score

    def _read_text_property(self, window_id: str, operation: str) -> str:
        result = self._run(["xdotool", operation, window_id])
        if not result or result.returncode != 0:
            return ""
        return result.stdout.strip()

    def _get_geometry(self, window_id: str) -> Optional[tuple[int, int, int, int]]:
        result = self._run(["xwininfo", "-id", window_id])
        if result and result.returncode == 0:
            geometry = self.parse_xwininfo(result.stdout)
            if geometry:
                return geometry

        result = self._run(
            ["xdotool", "getwindowgeometry", "--shell", window_id]
        )
        if result and result.returncode == 0:
            return self.parse_xdotool_geometry(result.stdout)
        return None

    def _build_candidate(self, window_id: str) -> Optional[WindowInfo]:
        geometry = self._get_geometry(window_id)
        if not geometry or not self.valid_geometry(geometry):
            return None

        title = self._read_text_property(window_id, "getwindowname")
        window_class = self._read_text_property(window_id, "getwindowclassname")
        pid_text = self._read_text_property(window_id, "getwindowpid")
        try:
            pid = int(pid_text)
        except ValueError:
            pid = None

        score = self.score_candidate(title, window_class, geometry)
        return WindowInfo(
            window_id=window_id,
            title=title,
            window_class=window_class,
            x=geometry[0],
            y=geometry[1],
            width=geometry[2],
            height=geometry[3],
            pid=pid,
            score=score,
        )

    def list_candidates(self) -> list[WindowInfo]:
        result = self._run(["xdotool", "search", "--onlyvisible", "--name", "."])
        if not result or result.returncode != 0:
            return []

        candidates: list[WindowInfo] = []
        seen: set[str] = set()
        for window_id in result.stdout.split():
            if window_id in seen or not window_id.isdigit():
                continue
            seen.add(window_id)
            candidate = self._build_candidate(window_id)
            if candidate:
                candidates.append(candidate)
        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def find_game_window(self, timeout: float = 0.0) -> bool:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            candidates = self.list_candidates()
            if candidates and candidates[0].score >= self.minimum_score:
                self.window_info = candidates[0]
                return True
            if time.monotonic() >= deadline:
                self.window_info = None
                return False
            time.sleep(0.25)

    def refresh(self) -> bool:
        """刷新当前窗口坐标；窗口消失时尝试重新查找。"""
        if self.window_info:
            candidate = self._build_candidate(self.window_info.window_id)
            if candidate and candidate.score >= self.minimum_score:
                self.window_info = candidate
                return True
        return self.find_game_window()

    def get_screenshot_region(self) -> Optional[tuple[int, int, int, int]]:
        if not self.window_info and not self.find_game_window():
            return None
        if not self.refresh() or not self.window_info:
            return None
        return self.window_info.region

    def get_active_window_id(self) -> Optional[str]:
        result = self._run(["xdotool", "getactivewindow"])
        if not result or result.returncode != 0:
            return None
        window_id = result.stdout.strip()
        return window_id if window_id.isdigit() else None

    def is_game_window_active(self) -> bool:
        if not self.window_info:
            return False
        active_window = self.get_active_window_id()
        return active_window == self.window_info.window_id

    def activate_window(self) -> bool:
        """激活游戏窗口并验证焦点确实已经切换。"""
        if not self.window_info and not self.find_game_window():
            return False

        window_id = self.window_info.window_id
        result = self._run(
            ["xdotool", "windowactivate", "--sync", window_id]
        )
        if result and result.returncode == 0:
            time.sleep(0.1)
            if self.is_game_window_active():
                return True

        # 某些轻量窗口管理器不支持_NET_ACTIVE_WINDOW，回退到直接聚焦。
        result = self._run(["xdotool", "windowfocus", "--sync", window_id])
        if not result or result.returncode != 0:
            return False
        time.sleep(0.1)
        return self.is_game_window_active()

    def move_window(self, x: int, y: int) -> bool:
        """显式移动游戏窗口；不会改变窗口大小。"""
        if not self.window_info and not self.find_game_window():
            return False
        result = self._run(
            ["xdotool", "windowmove", self.window_info.window_id, str(x), str(y)]
        )
        if not result or result.returncode != 0:
            return False
        time.sleep(0.1)
        return self.refresh()


def format_candidates(candidates: Iterable[WindowInfo]) -> str:
    lines = []
    for item in candidates:
        lines.append(
            f"id={item.window_id} score={item.score} "
            f"region={item.region} class={item.window_class!r} title={item.title!r}"
        )
    return "\n".join(lines)


def main() -> int:
    controller = WindowController()
    candidates = controller.list_candidates()
    if not candidates:
        print("没有读取到可见的 X11 窗口；请确认 DISPLAY 和 xdotool。")
        return 1
    print(format_candidates(candidates))
    if not controller.find_game_window() or not controller.window_info:
        print("\n没有找到置信度足够高的游戏窗口。")
        return 2
    print(f"\n已选择: {controller.window_info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
