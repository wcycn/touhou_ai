# 模型说明

`best.pt` 是当前自动控制使用的 21 类 YOLO 权重。

已从权重中确认：

- 任务：目标检测
- 训练参数记录的输入尺寸：640
- SHA-256：`78eb395d277bb5f35f27025a7bada7725928d6e7f7b15681f659a43b5bf60ab2`

类别：

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

权重内部只保留了原训练配置路径
`/home/ma-user/work/fixed_config.yaml`，项目中没有该配置或训练数据，因此不能据此
复现训练。

公开发布前仍需补齐：

- 模型训练者和版本来源
- 训练数据来源及许可
- 权重允许使用和再分发的依据
- 基础评估指标

在这些信息确认前，不应默认宣称模型具有可自由再分发的开源许可。
