# Touhou AI v1.1.0 — First Public Experimental Release

这是 Touhou AI 的首个公开实验版本。项目已经串联游戏窗口截图、YOLO 目标检测、
规则规划和键盘输入，并提供无需命令行的桌面控制中心。

![Touhou AI detection, tracking and control demo](https://raw.githubusercontent.com/wcycn/touhou_ai/main/docs/assets/touhou-ai-demo.gif)

> Touhou AI 是非官方二次创作技术实验，与上海爱丽丝幻乐团、ZUN 或东方 Project
> 官方无关。发行版不包含游戏本体。

## 主要功能

- 通过 Wine 启动并自动定位《东方红魔乡》X11 窗口
- 使用 21 类 Ultralytics YOLO 模型识别自机、弹幕、敌人、Boss 和 Power 道具
- 跟踪自机短时漏检与敌弹运动，估计轨迹、碰撞风险和 TTC
- 比较八方向候选路径，执行移动、射击和有限的自动 Bomb
- 窗口失焦、自机丢失或场景不确定时立即释放全部按键
- Safe Observation 运行相同的检测、跟踪和规划流程，但禁止键盘和鼠标事件
- 统一桌面 GUI 集成启动、观察、控制、诊断、会话复盘和模型工具
- 保存结构化会话记录、抽样画面、质量报告和人工审核候选

## 模型与下载

GitHub Release 不附带额外二进制文件。GitHub 会自动提供本标签对应的
**Source code (zip)** 与 **Source code (tar.gz)**。

YOLO 权重由 Hugging Face 单独托管：

- 模型页面：<https://huggingface.co/wcycn/touhou-ai-yolo>
- 文件：`best.pt`
- SHA-256：`78eb395d277bb5f35f27025a7bada7725928d6e7f7b15681f659a43b5bf60ab2`

首次启动 Safe Observation 或 AI Control 时，程序会自动下载并校验模型。也可以
在 GUI 中选择 **Tools → Download / Verify YOLO Model** 提前获取。

原始训练截图只保存在私人备份中，不属于本公开发行版。

## 快速开始

当前版本仅支持 Linux X11。

```bash
git clone https://github.com/wcycn/touhou_ai.git
cd touhou_ai

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python3 touhou_ai.py gui
```

请将合法取得的《东方红魔乡》游戏文件放入 `game/`。完整安装和首次运行步骤见
[README](https://github.com/wcycn/touhou_ai#快速开始)。

## 当前能力

在当前测试环境中，控制器曾完成基本移动和射击、回到场地中部、收集部分 Power
道具，并运行到第一面 Boss。这只说明各模块可以连续工作；当前版本不能稳定完成
第一关。

## 已知限制

- 模型尚无标准化人工验证集指标，部分类别识别不稳定
- 激光没有独立标注，密集 Boss 弹幕仍难以规避
- 菜单、死亡、续关和关卡切换尚未形成完整自动状态机
- 运行效果受到分辨率、窗口缩放、游戏版本和推理速度影响
- 仅适配 Linux X11；Wayland、Windows 原生和 macOS 尚未支持

## 安全提示

Control Mode 会聚焦游戏窗口并发送真实键盘事件。切换到其他程序前，请先点击
**STOP AI AND RELEASE ALL KEYS**。第一次使用应先运行 Safe Observation 和
左右输入测试。

## 验证与许可

- 58 项不启动游戏、不发送输入的自动测试已通过
- 源代码与公开 YOLO 权重采用
  [GNU AGPL v3.0](https://github.com/wcycn/touhou_ai/blob/main/LICENSE)
- 游戏本体、原始截图及其他第三方素材不在该许可证授权范围内

完整变更记录见
[CHANGELOG.md](https://github.com/wcycn/touhou_ai/blob/main/CHANGELOG.md)。
