---
license: agpl-3.0
library_name: ultralytics
pipeline_tag: object-detection
tags:
  - yolo
  - ultralytics
  - object-detection
  - bullet-hell
  - touhou-project
model_name: Touhou AI YOLO Detector
---

# Touhou AI YOLO Detector

An experimental 21-class object detector used by
[Touhou AI](https://github.com/wcycn/touhou_ai), an unofficial visual-control
project for *Touhou Koumakyou: the Embodiment of Scarlet Devil*.

This is a fan-made technical experiment. It is not affiliated with, endorsed
by, or sponsored by Team Shanghai Alice, ZUN, or the Touhou Project.

## Demo

[![Touhou AI detection, tracking and control demo](https://raw.githubusercontent.com/wcycn/touhou_ai/main/docs/assets/touhou-ai-demo.gif)](https://github.com/wcycn/touhou_ai)

The overlay is reconstructed from a recorded AI Control session and shows
detections, tracked bullet trajectories, collision risk, planned movement, and
the keys held by the controller.

## Model details

- Task: object detection
- Framework: Ultralytics YOLO / PyTorch
- Checkpoint format: PyTorch `.pt`
- Training image size: 640
- Classes: 21
- SHA-256:
  `78eb395d277bb5f35f27025a7bada7725928d6e7f7b15681f659a43b5bf60ab2`

## Classes

| ID | Class |
|---:|---|
| 0 | boss |
| 1 | boss_occluded |
| 2 | bullet_enemy |
| 3 | bullet_enemy_big_blue |
| 4 | bullet_enemy_big_green |
| 5 | bullet_enemy_big_red |
| 6 | bullet_enemy_big_yellow |
| 7 | bullet_enemy_small_blue |
| 8 | bullet_enemy_small_green |
| 9 | bullet_enemy_small_red |
| 10 | bullet_enemy_small_yellow |
| 11 | bullet_enemy_unique_blue |
| 12 | bullet_enemy_unique_green |
| 13 | bullet_enemy_unique_red |
| 14 | bullet_enemy_unique_yellow |
| 15 | bullet_player |
| 16 | character |
| 17 | enemy_small_blue |
| 18 | enemy_small_red |
| 19 | powerup_blue |
| 20 | powerup_red |

## Training data

The repository owner trained this checkpoint on a locally collected and
annotated screenshot dataset:

- Total images: 1,834
- Training split: 1,794 images and 1,794 YOLO label files
- Validation split: 40 images and 40 YOLO label files
- Annotation format: YOLO bounding boxes

The raw screenshots are not distributed with this model because they contain
game imagery. The repository owner keeps a
[private archival backup](https://huggingface.co/datasets/wcycn/touhou-ai-dataset)
for recovery and reproducibility; it is not a public dataset. A separate
`final_train` folder found in the historical project was produced later through
automatic annotation and was not used to train this checkpoint.

## Usage

```python
from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict(
    source="frame.png",
    imgsz=640,
    conf=0.15,
)
```

The checkpoint is designed for the original project's capture pipeline and
should not be assumed to generalize to other Touhou games, resolutions, visual
mods, scaling settings, or capture methods.

## Known limitations

- No independent, manually reviewed benchmark is currently published.
- Detection quality is uneven across the 21 classes.
- Lasers are not represented as a dedicated class.
- Dense boss patterns and partially occluded player sprites remain difficult.
- Some training samples were augmented; reported sample counts are not a
  substitute for real-world evaluation.

This model should not be presented as a reliable game-clear system or as a
general-purpose Touhou detector.

## License

The checkpoint is released under the
[GNU Affero General Public License v3.0](https://huggingface.co/wcycn/touhou-ai-yolo/blob/main/LICENSE),
consistent with the
Ultralytics YOLO open-source license.

The Touhou Project, game imagery, names, characters, and other third-party
materials remain the property of their respective rights holders and are not
licensed by this model repository.
