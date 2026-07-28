"""模型评估配置检查，不运行真实推理。"""

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_evaluation import validate_dataset_yaml  # noqa: E402


class DatasetValidationTests(unittest.TestCase):
    def test_valid_dataset_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.yaml"
            path.write_text(
                "path: .\ntrain: images/train\nval: images/val\n"
                "names: [character, bullet]\n",
                encoding="utf-8",
            )
            data = validate_dataset_yaml(path)
            self.assertEqual(data["names"][0], "character")

    def test_missing_validation_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.yaml"
            path.write_text("names: [character]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "val"):
                validate_dataset_yaml(path)


if __name__ == "__main__":
    unittest.main()

