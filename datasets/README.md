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
