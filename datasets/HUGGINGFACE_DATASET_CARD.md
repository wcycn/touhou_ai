---
pretty_name: Touhou AI YOLO Training Data
task_categories:
  - object-detection
size_categories:
  - 1K<n<10K
tags:
  - yolo
  - object-detection
  - bullet-hell
  - touhou-project
---

# Touhou AI YOLO Training Data

Private archival backup of the labeled screenshot dataset used to train the
[Touhou AI YOLO Detector](https://huggingface.co/wcycn/touhou-ai-yolo).

Related source project:
[wcycn/touhou_ai](https://github.com/wcycn/touhou_ai).

## Visibility and rights

This repository is intentionally private. The screenshots contain imagery from
*Touhou Koumakyou: the Embodiment of Scarlet Devil*. They are stored for
personal backup and reproducibility, not offered for public redistribution.

The annotations and project metadata were created by the repository owner.
Keeping this backup private does not grant or imply a license to the underlying
game imagery.

## Contents

```text
train/
├── images/  # 1,794 PNG files
└── labels/  # 1,794 YOLO TXT files
val/
├── images/  # 40 PNG files
└── labels/  # 40 YOLO TXT files
data.yaml
dataset_stats.json
metadata/
└── yolo_config.original.yaml
```

- Total images: 1,834
- Total label files: 1,834
- Classes: 21
- Format: YOLO normalized bounding boxes
- Approximate stored size: 3.9 GB

`data.yaml` uses relative paths and is the portable configuration.
`metadata/yolo_config.original.yaml` preserves the historical training
configuration, including its old machine-specific path and experiment notes.

## Relationship to the published checkpoint

The public `best.pt` checkpoint is hosted in the linked model repository. The
historical training result was compared with that checkpoint and both files had
the following SHA-256:

```text
78eb395d277bb5f35f27025a7bada7725928d6e7f7b15681f659a43b5bf60ab2
```

The later `filtered_training_dataset` and `final_train` folders are derivative
or automatically annotated working sets and are not part of this canonical
backup.
