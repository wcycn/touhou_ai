"""为 Ultralytics 选择可实际执行的 PyTorch 推理设备。"""

from __future__ import annotations

from dataclasses import dataclass
import warnings


@dataclass(frozen=True)
class InferenceDevice:
    value: str
    label: str
    reason: str | None = None


def select_inference_device(torch_module) -> InferenceDevice:
    """避免 ``cuda.is_available()`` 对不兼容新显卡产生假阳性。"""
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not cuda.is_available():
        return InferenceDevice("cpu", "CPU", "CUDA 不可用")

    try:
        # PyTorch 会在读取新显卡能力时先输出自身的兼容性警告；这里会给出
        # 更准确的中文回退说明，因此抑制重复警告。
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            major, minor = cuda.get_device_capability(0)
        target_arch = f"sm_{major}{minor}"
        target_ptx = f"compute_{major}{minor}"
        supported_arches = set(cuda.get_arch_list())
        device_name = cuda.get_device_name(0)
    except Exception as exc:
        return InferenceDevice(
            "cpu",
            "CPU",
            f"无法确认 CUDA 兼容性：{exc}",
        )

    if (
        supported_arches
        and target_arch not in supported_arches
        and target_ptx not in supported_arches
    ):
        available = ", ".join(sorted(supported_arches))
        return InferenceDevice(
            "cpu",
            "CPU",
            (
                f"{device_name} 需要 {target_arch}，当前 PyTorch 仅包含 "
                f"{available or '未知架构'}"
            ),
        )

    try:
        torch_module.empty(1, device="cuda:0")
        cuda.synchronize()
    except Exception as exc:
        return InferenceDevice(
            "cpu",
            "CPU",
            f"CUDA 试运行失败：{exc}",
        )

    return InferenceDevice(
        "cuda:0",
        f"CUDA ({device_name}, {target_arch})",
    )
