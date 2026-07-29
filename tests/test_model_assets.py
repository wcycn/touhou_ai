"""Tests for downloading and verifying the published model asset."""

from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from model_assets import (
    ModelAssetError,
    download_model,
    ensure_model,
    file_sha256,
    verify_model,
)


class ModelAssetsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.source = self.root / "source.pt"
        self.source.write_bytes(b"test-model-weights")
        self.expected_sha256 = hashlib.sha256(
            self.source.read_bytes()
        ).hexdigest()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_download_is_verified_and_written_to_destination(self) -> None:
        destination = self.root / "models" / "best.pt"

        result = download_model(
            destination,
            url=self.source.as_uri(),
            expected_sha256=self.expected_sha256,
            reporter=lambda _message: None,
        )

        self.assertEqual(result, destination)
        self.assertEqual(result.read_bytes(), self.source.read_bytes())
        self.assertEqual(file_sha256(result), self.expected_sha256)
        self.assertTrue(verify_model(result, self.expected_sha256))

    def test_bad_download_is_rejected_without_partial_destination(self) -> None:
        destination = self.root / "models" / "best.pt"

        with self.assertRaises(ModelAssetError):
            download_model(
                destination,
                url=self.source.as_uri(),
                expected_sha256="0" * 64,
                reporter=lambda _message: None,
            )

        self.assertFalse(destination.exists())
        self.assertEqual(list(destination.parent.glob("*.part")), [])

    def test_existing_verified_model_is_reused(self) -> None:
        destination = self.root / "best.pt"
        destination.write_bytes(self.source.read_bytes())
        messages: list[str] = []

        result = ensure_model(
            destination,
            url="https://invalid.example/unused",
            expected_sha256=self.expected_sha256,
            reporter=messages.append,
        )

        self.assertEqual(result, destination.resolve())
        self.assertIn("verified", messages[0].lower())

    def test_existing_modified_model_is_not_overwritten(self) -> None:
        destination = self.root / "best.pt"
        destination.write_bytes(b"custom-model")

        with self.assertRaises(ModelAssetError):
            ensure_model(
                destination,
                url=self.source.as_uri(),
                expected_sha256=self.expected_sha256,
                reporter=lambda _message: None,
            )

        self.assertEqual(destination.read_bytes(), b"custom-model")


if __name__ == "__main__":
    unittest.main()
