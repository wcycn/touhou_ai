#!/usr/bin/env python3
"""在人工标注的YOLO数据集上评估当前模型。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from numbers import Real
from pathlib import Path
from typing import Any

import yaml

from model_assets import DEFAULT_MODEL_PATH, ensure_model


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = DEFAULT_MODEL_PATH
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "evaluation"


def validate_dataset_yaml(path: Path | str) -> dict:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"数据集配置不存在: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("数据集YAML必须是映射")
    missing = [key for key in ("val", "names") if key not in data]
    if missing:
        raise ValueError(f"数据集YAML缺少字段: {', '.join(missing)}")
    names = data["names"]
    if not isinstance(names, (list, dict)) or not names:
        raise ValueError("names必须是非空列表或映射")
    data["_yaml_path"] = str(path)
    return data


def metrics_to_dict(metrics: Any) -> dict:
    box = getattr(metrics, "box", None)
    speed = getattr(metrics, "speed", {}) or {}
    per_class = getattr(box, "maps", None)
    results = {
        "map50_95": float(getattr(box, "map", 0.0) or 0.0),
        "map50": float(getattr(box, "map50", 0.0) or 0.0),
        "map75": float(getattr(box, "map75", 0.0) or 0.0),
        "per_class_map50_95": [
            float(value)
            for value in ([] if per_class is None else per_class)
        ],
        "speed_ms": {
            str(key): float(value)
            for key, value in speed.items()
        },
    }
    results_dict = getattr(metrics, "results_dict", None)
    if isinstance(results_dict, dict):
        results["results"] = {
            str(key): float(value)
            for key, value in results_dict.items()
            if isinstance(value, Real)
        }
    return results


def run_evaluation(
    data_yaml: Path | str,
    model_path: Path | str = DEFAULT_MODEL,
    confidence: float = 0.25,
    iou: float = 0.7,
    image_size: int = 640,
    device: str = "",
    output_dir: Path | str = DEFAULT_OUTPUT,
) -> Path:
    data = validate_dataset_yaml(data_yaml)
    data_yaml = Path(data["_yaml_path"])
    model_path = Path(model_path).expanduser().resolve()
    if model_path == DEFAULT_MODEL.resolve():
        model_path = ensure_model(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在: {model_path}")
    if not 0 <= confidence <= 1 or not 0 <= iou <= 1:
        raise ValueError("confidence和iou必须在0到1之间")
    if image_size < 64:
        raise ValueError("image_size过小")

    from ultralytics import YOLO

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = datetime.now().strftime("val_%Y%m%d_%H%M%S")
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        conf=confidence,
        iou=iou,
        imgsz=image_size,
        device=device or None,
        project=str(output_dir),
        name=run_name,
        exist_ok=False,
        plots=True,
    )
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path),
        "dataset": str(data_yaml),
        "parameters": {
            "confidence": confidence,
            "iou": iou,
            "image_size": image_size,
            "device": device or "auto",
        },
        "metrics": metrics_to_dict(metrics),
    }
    result_path = output_dir / run_name / "evaluation.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result_path


def main() -> int:
    parser = argparse.ArgumentParser(description="评估YOLO模型")
    parser.add_argument("data", type=Path, help="人工标注数据集的data.yaml")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = run_evaluation(
            args.data,
            args.model,
            args.confidence,
            args.iou,
            args.image_size,
            args.device,
            args.output_dir,
        )
        print(f"✅ 模型评估完成: {result}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"❌ 模型评估失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
