# 模型说明

当前自动控制使用的 21 类 YOLO 权重发布在：

<https://huggingface.co/wcycn/touhou-ai-yolo>

`best.pt` 不再跟随 GitHub 源码仓库分发。程序在第一次启动 AI 时自动下载并校验
权重，也可以通过 GUI 的 **Tools → Download / Verify YOLO Model** 提前获取。
下载后的本地缓存仍位于本目录，但已被 Git 忽略。

## 来源

该权重由项目作者使用自己整理和标注的《东方红魔乡》截图数据训练。

本机旧工程中仍保留了对应的训练结果和数据集，训练结果
`weights/best.pt` 与当前发布权重的 SHA-256 完全一致。原始数据集未包含在仓库中：

- 总图片：1,834 张
- 训练集：1,794 张图片及 1,794 份 YOLO 标签
- 验证集：40 张图片及 40 份 YOLO 标签
- 类别：21

旧工程中另有一份 `final_train` 自动预标注候选集。它由当前模型生成，不是训练
当前模型所用的原始数据集。

对应训练数据已备份到私人 Hugging Face Dataset 仓库：
<https://huggingface.co/datasets/wcycn/touhou-ai-dataset>。该地址仅仓库所有者
可访问，不作为公开数据集分发。

## 权重信息

- 任务：目标检测
- 训练参数记录的输入尺寸：640
- SHA-256：`78eb395d277bb5f35f27025a7bada7725928d6e7f7b15681f659a43b5bf60ab2`

## 类别

| ID | 名称 |
|---:|---|
| 0 | boss |
| 1 | boss_occluded |
| 2 | bullet_enemy |
| 3–6 | bullet_enemy_big_blue/green/red/yellow |
| 7–10 | bullet_enemy_small_blue/green/red/yellow |
| 11–14 | bullet_enemy_unique_blue/green/red/yellow |
| 15 | bullet_player |
| 16 | character |
| 17 | enemy_small_blue |
| 18 | enemy_small_red |
| 19 | powerup_blue |
| 20 | powerup_red |

训练数据包含游戏画面，因此默认不随代码仓库发布。当前仓库也没有提供可复现
训练过程的完整脚本、环境锁定文件和基准指标；权重主要用于运行本项目的实验版本。

## 许可证

Hugging Face 提供的 `best.pt` 按
[GNU Affero General Public License v3.0](../LICENSE) 发布，以保持与 Ultralytics
YOLO 开源许可一致。原始训练截图不包含在此授权中。
