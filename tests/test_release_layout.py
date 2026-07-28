"""发布目录边界检查。"""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ReleaseLayoutTests(unittest.TestCase):
    def test_single_gui_and_no_legacy_gui(self) -> None:
        self.assertTrue((ROOT / "desktop_gui.py").is_file())
        self.assertFalse((ROOT / "mindspore_english_gui.py").exists())
        self.assertFalse((ROOT / "mindspore_english_gui_pc.py").exists())
        self.assertFalse((ROOT / "pc_orangepi_gui.py").exists())

    def test_local_game_files_are_ignored(self) -> None:
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("game/*", gitignore)
        self.assertIn("!game/README.md", gitignore)
        self.assertIn("!game/.gitkeep", gitignore)

    def test_release_sources_have_no_stale_external_project_paths(self) -> None:
        forbidden = (
            "/" + "media/",
            "touhou_ai_" + "release",
            "touhou_ai_" + "orangepi",
            "touhou_ai_" + "mindspore_complete",
        )
        checked_suffixes = {".py", ".json", ".md", ".sh"}
        for path in ROOT.rglob("*"):
            relative_path = path.relative_to(ROOT)
            if (
                not path.is_file()
                or path.resolve() == Path(__file__).resolve()
                or ".git" in path.parts
                or relative_path.parts[0] in {"runs", "sessions"}
                or path.suffix not in checked_suffixes
            ):
                continue
            content = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, content, f"{relative_path}: {marker}")

    def test_required_release_documents_exist(self) -> None:
        for relative_path in (
            "README.md",
            "VERSION",
            "requirements.txt",
            "control_logic.py",
            "session_analysis.py",
            "model_evaluation.py",
            "docs/ROADMAP.md",
            "docs/ARCHITECTURE.md",
            "docs/RELEASE_CHECKLIST.md",
        ):
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)


if __name__ == "__main__":
    unittest.main()
