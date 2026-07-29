# 数据集目录

本机训练集、验证集和对应的 YOLO `data.yaml` 放在此目录内。

建议使用：

```text
datasets/
└── validation/
    ├── data.yaml
    ├── images/
    └── labels/
```

数据集默认被 Git 忽略，避免把体积较大或授权不明确的数据误放进发布包。

当前训练集另有一个仅仓库所有者可访问的 Hugging Face 私人备份：

<https://huggingface.co/datasets/wcycn/touhou-ai-dataset>

私人备份用于防止本机数据丢失，并不代表原始游戏截图可以公开再分发。仓库中的
`HUGGINGFACE_DATASET_CARD.md` 和 `HUGGINGFACE_DATASET_CONFIG.yaml` 分别是该
私人仓库的说明页与便携式 YOLO 配置；它们不包含任何游戏截图。
