#!/usr/bin/env python3
"""Download and verify the published Touhou AI detector."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_REPO_ID = "wcycn/touhou-ai-yolo"
MODEL_FILENAME = "best.pt"
MODEL_URL = (
    f"https://huggingface.co/{MODEL_REPO_ID}/resolve/main/{MODEL_FILENAME}"
)
MODEL_PAGE_URL = f"https://huggingface.co/{MODEL_REPO_ID}"
MODEL_SHA256 = (
    "78eb395d277bb5f35f27025a7bada772"
    "5928d6e7f7b15681f659a43b5bf60ab2"
)
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / MODEL_FILENAME


class ModelAssetError(RuntimeError):
    """Raised when the published model cannot be obtained safely."""


def file_sha256(path: Path | str) -> str:
    """Return the SHA-256 digest of a local file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(
    path: Path | str = DEFAULT_MODEL_PATH,
    expected_sha256: str = MODEL_SHA256,
) -> bool:
    """Return whether a model exists and matches the published checksum."""
    path = Path(path)
    return path.is_file() and file_sha256(path) == expected_sha256


def download_model(
    destination: Path | str = DEFAULT_MODEL_PATH,
    *,
    url: str = MODEL_URL,
    expected_sha256: str = MODEL_SHA256,
    reporter: Callable[[str], None] = print,
) -> Path:
    """Download a model atomically and reject incomplete or modified files."""
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            request = Request(
                url,
                headers={"User-Agent": "touhou-ai/1.1.0"},
            )
            with urlopen(request, timeout=60) as response:
                total_header = response.headers.get("Content-Length")
                total = int(total_header) if total_header else 0
                downloaded = 0
                next_report = 25
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    temporary.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        percent = int(downloaded * 100 / total)
                        if percent >= next_report:
                            reporter(
                                f"Model download: {min(percent, 100)}% "
                                f"({downloaded / 1_000_000:.1f} MB)"
                            )
                            next_report += 25

        actual_sha256 = file_sha256(temporary_path)
        if actual_sha256 != expected_sha256:
            raise ModelAssetError(
                "Downloaded model checksum mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        return destination
    except ModelAssetError:
        raise
    except (HTTPError, URLError, OSError, ValueError) as exc:
        raise ModelAssetError(f"Could not download model from {url}: {exc}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def ensure_model(
    path: Path | str = DEFAULT_MODEL_PATH,
    *,
    url: str = MODEL_URL,
    expected_sha256: str = MODEL_SHA256,
    reporter: Callable[[str], None] = print,
) -> Path:
    """Return a verified local model, downloading it when it is absent."""
    path = Path(path).expanduser().resolve()
    if path.exists():
        if verify_model(path, expected_sha256):
            reporter(f"Model verified: {path}")
            return path
        raise ModelAssetError(
            f"Existing model failed checksum verification: {path}. "
            "Move or remove that file, then retry the download."
        )

    reporter(f"Downloading published model from {MODEL_PAGE_URL}")
    downloaded = download_model(
        path,
        url=url,
        expected_sha256=expected_sha256,
        reporter=reporter,
    )
    reporter(f"Model ready: {downloaded}")
    return downloaded


def main() -> int:
    try:
        path = ensure_model()
    except ModelAssetError as exc:
        print(f"Model setup failed: {exc}")
        return 1
    print(f"SHA-256: {file_sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
