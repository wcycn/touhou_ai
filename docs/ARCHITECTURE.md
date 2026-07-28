# 架构与安全边界

## 运行数据流

```text
window_controller.py
  └─ 发现、定位、激活X11游戏窗口
       ↓
autopilot.py / observe_game.py
  └─ MSS截图 → YOLO检测
       ↓
control_logic.py
  ├─ PlayerTracker：自机平滑、短时预测、超时停控
  ├─ BulletTracker：敌弹跨帧匹配与速度估计
  ├─ collision_metrics：TTC、最近距离、碰撞风险
  ├─ RiskPlanner：比较停留/左右/上下候选代价
  ├─ ActionStabilizer：边界、最短保持、反向冷却
  ├─ GameSceneStateMachine：保守战斗状态确认
  └─ InputStateMachine：只发送按键状态差异
       ↓
PyAutoGUI（仅正式AI；观察模式从模块层禁用）
```

每一帧的检测、跟踪状态、场景、风险、候选代价、计划动作、实际按键差异和焦点结果
由 `session_recorder.py` 写入会话。`session_analysis.py` 对会话计算质量指标并挑选
人工审核候选。`model_evaluation.py` 只接受人工确认后的YOLO数据集。

## 输入安全条件

只有同时满足以下条件时才允许移动或射击：

1. 游戏窗口已经找到并确认处于活动状态。
2. 场景状态机连续观察到战斗证据。
3. 自机是当前检测结果或仍处于短时预测窗口。
4. 动作通过边界保护和方向切换冷却。

焦点丢失、自机超时、场景离开战斗或进程退出都会释放当前持有按键。

## 场景状态机的限制

当前状态机能保守地区分“确认战斗”和“非战斗/过场”，不会在未知画面中乱按。
它还不能可靠命名具体菜单、暂停、Miss、Game Over或结局，因为项目没有这些画面的
模板或标注数据。因此目前不会自动执行菜单导航。收集并确认相应截图后，可以增加
模板分类或专用场景模型，而无需修改输入安全边界。

## 数据边界

- `detections` 是模型当前帧原始输出。
- `tracked_player` 和 `bullet_tracks` 是时序估计，不是真实标注。
- `prelabels/` 是模型伪标签，必须人工修正。
- 只有人工确认的 `labels/` 才能用于正式模型指标和训练。

