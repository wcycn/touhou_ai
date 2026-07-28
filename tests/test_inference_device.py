"""推理设备选择回归测试。"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference_device import select_inference_device


class FakeCuda:
    def __init__(self, available: bool, arches: list[str]):
        self.available = available
        self.arches = arches
        self.synchronized = False

    def is_available(self) -> bool:
        return self.available

    def get_device_capability(self, _index: int) -> tuple[int, int]:
        return (12, 0)

    def get_arch_list(self) -> list[str]:
        return self.arches

    def get_device_name(self, _index: int) -> str:
        return "Test GPU"

    def synchronize(self) -> None:
        self.synchronized = True


class FakeTorch:
    def __init__(self, available: bool, arches: list[str]):
        self.cuda = FakeCuda(available, arches)
        self.empty_calls: list[tuple[int, str]] = []

    def empty(self, size: int, device: str):
        self.empty_calls.append((size, device))
        return object()


class DeviceSelectionTests(unittest.TestCase):
    def test_unavailable_cuda_uses_cpu(self) -> None:
        backend = FakeTorch(False, [])
        selection = select_inference_device(backend)
        self.assertEqual(selection.value, "cpu")
        self.assertEqual(backend.empty_calls, [])

    def test_newer_gpu_than_torch_build_uses_cpu(self) -> None:
        backend = FakeTorch(True, ["sm_80", "sm_86", "sm_90"])
        selection = select_inference_device(backend)
        self.assertEqual(selection.value, "cpu")
        self.assertIn("sm_120", selection.reason)
        self.assertEqual(backend.empty_calls, [])

    def test_compatible_gpu_is_probed_and_selected(self) -> None:
        backend = FakeTorch(True, ["sm_90", "sm_120"])
        selection = select_inference_device(backend)
        self.assertEqual(selection.value, "cuda:0")
        self.assertEqual(backend.empty_calls, [(1, "cuda:0")])
        self.assertTrue(backend.cuda.synchronized)


if __name__ == "__main__":
    unittest.main()
