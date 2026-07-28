"""X11窗口定位器的纯解析与候选选择测试。"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT
sys.path.insert(0, str(PROJECT_DIR))

from window_controller import WindowController, WindowInfo  # noqa: E402


def completed(command, stdout="", returncode=0):
    return subprocess.CompletedProcess(command, returncode, stdout, "")


class GeometryParserTests(unittest.TestCase):
    def test_parse_xwininfo_supports_negative_monitor_coordinates(self) -> None:
        output = """
          Absolute upper-left X:  -1280
          Absolute upper-left Y:  42
          Width: 640
          Height: 480
        """
        self.assertEqual(
            WindowController.parse_xwininfo(output),
            (-1280, 42, 640, 480),
        )

    def test_parse_xdotool_geometry(self) -> None:
        output = "WINDOW=12\nX=120\nY=80\nWIDTH=1280\nHEIGHT=960\nSCREEN=0\n"
        self.assertEqual(
            WindowController.parse_xdotool_geometry(output),
            (120, 80, 1280, 960),
        )


class CandidateSelectionTests(unittest.TestCase):
    def test_real_game_beats_terminal_with_th06_in_title(self) -> None:
        game_score = WindowController.score_candidate(
            "東方紅魔郷 ～ the Embodiment of Scarlet Devil",
            "th06.exe",
            (100, 100, 640, 480),
        )
        terminal_score = WindowController.score_candidate(
            "python th06 debug",
            "gnome-terminal-server",
            (0, 0, 1200, 900),
        )
        self.assertGreater(game_score, terminal_score)
        self.assertGreaterEqual(game_score, 55)

    def test_controller_reads_and_selects_game_window(self) -> None:
        def runner(command):
            command = tuple(command)
            if command == ("xdotool", "search", "--onlyvisible", "--name", "."):
                return completed(command, "10\n20\n")
            if command[:2] == ("xwininfo", "-id"):
                window_id = command[2]
                if window_id == "10":
                    return completed(
                        command,
                        "Absolute upper-left X: 0\n"
                        "Absolute upper-left Y: 0\n"
                        "Width: 1000\nHeight: 700\n",
                    )
                return completed(
                    command,
                    "Absolute upper-left X: 240\n"
                    "Absolute upper-left Y: 120\n"
                    "Width: 640\nHeight: 480\n",
                )
            operation, window_id = command[1], command[2]
            properties = {
                ("getwindowname", "10"): "Terminal",
                ("getwindowclassname", "10"): "gnome-terminal-server",
                ("getwindowpid", "10"): "1000",
                ("getwindowname", "20"): "東方紅魔郷",
                ("getwindowclassname", "20"): "th06.exe",
                ("getwindowpid", "20"): "2000",
            }
            return completed(command, properties[(operation, window_id)])

        controller = WindowController(command_runner=runner)

        self.assertTrue(controller.find_game_window())
        self.assertIsNotNone(controller.window_info)
        self.assertEqual(controller.window_info.window_id, "20")
        self.assertEqual(controller.window_info.region, (240, 120, 640, 480))

    def test_activate_window_verifies_active_window(self) -> None:
        active = {"id": "10"}

        def runner(command):
            command = tuple(command)
            if command == ("xdotool", "getactivewindow"):
                return completed(command, active["id"])
            if command == (
                "xdotool",
                "windowactivate",
                "--sync",
                "20",
            ):
                active["id"] = "20"
                return completed(command)
            raise AssertionError(f"unexpected command: {command}")

        controller = WindowController(command_runner=runner)
        controller.window_info = WindowInfo(
            window_id="20",
            title="東方紅魔郷",
            window_class="th06.exe",
            x=100,
            y=100,
            width=640,
            height=480,
            score=100,
        )

        self.assertTrue(controller.activate_window())
        self.assertTrue(controller.is_game_window_active())


if __name__ == "__main__":
    unittest.main()
