"""独立游戏启动器的文件选择测试，不实际运行Wine。"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT
sys.path.insert(0, str(PROJECT_DIR))

from launch_game import select_executable  # noqa: E402


class ExecutableSelectionTests(unittest.TestCase):
    def test_vpatch_is_default_because_package_requires_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            game_dir = Path(directory)
            (game_dir / "vpatch.exe").touch()
            (game_dir / "th06c.exe").touch()
            self.assertEqual(select_executable(game_dir).name, "vpatch.exe")

    def test_direct_mode_skips_vpatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            game_dir = Path(directory)
            (game_dir / "vpatch.exe").touch()
            (game_dir / "th06c.exe").touch()
            self.assertEqual(
                select_executable(game_dir, direct=True).name,
                "th06c.exe",
            )

    def test_requested_executable_cannot_escape_game_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "游戏目录"):
                select_executable(Path(directory), requested="../outside.exe")


if __name__ == "__main__":
    unittest.main()
