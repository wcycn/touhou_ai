# Touhou AI

![Version](https://img.shields.io/badge/version-1.1.0-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Platform](https://img.shields.io/badge/platform-Linux%20X11-lightgrey)
![Status](https://img.shields.io/badge/status-experimental-orange)

《东方红魔乡》视觉检测与规则控制实验项目。

```text
Wine 游戏窗口 → MSS 截图 → YOLO/PyTorch 检测
→ 轨迹与风险分析 → 规则决策 → 焦点守护 → PyAutoGUI 输入
```

> [!IMPORTANT]
> 本项目是非官方二次创作技术实验，与上海爱丽丝幻乐团、ZUN 及东方 Project
> 官方无关联，也未获得其认可或赞助。仓库不提供游戏本体。

## 当前状态

v1.1.0 已完成基本闭环：启动游戏、定位窗口、安全观察、输入诊断、AI 控制、
会话记录和离线分析。项目尚未证明能够稳定生存或通关，因此当前定位是实验性
自动控制原型，不是成熟的自动通关工具。

## 功能

- 统一桌面 GUI，无需记忆日常命令
- X11 游戏窗口发现、客户区定位、移动与焦点确认
- YOLO/PyTorch 识别自机、敌弹、敌人、Boss 和道具
- 自机短时跟踪、敌弹速度估计、TTC 与未来碰撞风险
- 四方向风险规划、边界保护、方向保持和反向冷却
- 场景状态确认：未知、菜单和过场阶段不发送动作
- 焦点丢失或自机超时未识别时立即释放按键
- AI 模拟观察：运行完整 AI，但强制禁止键鼠输入
- AI 自动控制：向确认获得焦点的游戏窗口发送动作
- 会话记录、抽样画面、JSON/Markdown 报告和预标注导出

## 系统要求

- Linux X11 桌面
- Python 3.10 或更高版本
- Wine、`xdotool`、`xwininfo`
- Tkinter
- 支持 PyTorch 的 CPU 或 NVIDIA GPU 环境

如果当前 PyTorch 不包含显卡所需的 CUDA 架构，程序会明确提示原因并自动使用
CPU，不再让 YOLO 静默产生空检测。

Ubuntu/Debian 系统组件：

```bash
sudo apt install python3-venv python3-tk wine xdotool x11-utils
```

## 安装

```bash
git clone git@github.com:wcycn/touhou_ai.git
cd touhou_ai

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

将你合法取得的《东方红魔乡》游戏文件放入 `game/`。启动器依次寻找：

1. `vpatch.exe`
2. `th06c.exe`
3. `th06.exe`
4. `東方紅魔郷.exe`

游戏、用户设置、运行日志、截图、会话和本机数据集均已被 Git 忽略。

## 快速启动

先检查环境并运行无输入测试：

```bash
python touhou_ai.py check
python touhou_ai.py test
```

启动桌面控制中心：

```bash
python touhou_ai.py gui
```

也可以双击 `启动控制中心.sh`。

## 第一次实际测试

请严格按照以下顺序进行：

1. 在 GUI 点击“启动游戏”。
2. 把游戏设置为窗口模式，手动进入一个关卡。
3. 点击“定位游戏窗口”，确认日志中的坐标和尺寸正确。
4. 点击“启动 AI 模拟观察（不按键）”，观察 30 至 60 秒。
5. 确认预览能跟随自机和弹幕，且游戏没有收到 AI 按键。
6. 点击红色“立即停止 AI / 模拟观察并释放按键”。
7. 点击“输入测试”，确认游戏角色能短暂左右移动。
8. 重新进入关卡，选择均衡模式，并先使用推荐置信度 `0.15`。
9. 点击“启动 AI 自动控制（会按键）”。
10. 随时使用红色停止按钮；紧急情况下把鼠标移到屏幕角落触发
   PyAutoGUI failsafe。

测试时重点观察：

- 自机框是否稳定跟随真实位置
- 敌弹框是否覆盖主要威胁
- “当前模式”是否准确显示观察或控制状态
- 切出游戏窗口后是否立即停止发送动作
- 自机漏检后是否释放所有按键
- `sessions/` 是否产生会话记录和抽样画面

安全观察和自动控制使用同一套检测、跟踪、规划与决策逻辑；区别仅在于观察模式
使用禁用输入后端，绝不会执行计划动作。

## 常用命令

```bash
# 列出并定位游戏窗口
python touhou_ai.py locate

# 安全观察，不发送任何按键
python touhou_ai.py observe

# 短暂测试左右键
python touhou_ai.py control-test

# 启动正式自动控制
python touhou_ai.py ai --mode balanced

# 分析最新会话
python touhou_ai.py analyze

# 导出待人工审核的截图和 YOLO 预标注
python touhou_ai.py analyze --export-review

# 使用内部验证集评估模型
python touhou_ai.py model-eval datasets/validation/data.yaml
```

## 项目路径

项目可以整体移动或改名，不依赖启动时的工作目录。默认运行数据均位于项目内部：

| 内容 | 路径 |
|---|---|
| YOLO 模型 | `models/best.pt` |
| 本机游戏 | `game/` |
| 训练或验证数据 | `datasets/` |
| GUI 设置 | `settings.json` |
| 会话和抽样画面 | `sessions/` |
| 日志、评估和 Ultralytics 配置 | `runs/` |

## 目录结构

```text
.
├── touhou_ai.py          # 统一命令入口
├── desktop_gui.py        # 桌面控制中心
├── autopilot.py          # YOLO 检测与自动控制
├── control_logic.py      # 跟踪、场景、规划和输入状态
├── observe_game.py       # 无输入的 AI 模拟观察
├── control_test.py       # 短按键诊断
├── launch_game.py        # Wine 游戏启动器
├── window_controller.py  # X11 窗口定位与聚焦
├── session_recorder.py   # 会话记录与报告
├── session_analysis.py   # 会话指标与审核候选
├── model_evaluation.py   # YOLO 数据集评估
├── inference_device.py   # CUDA 兼容性检测与 CPU 回退
├── models/               # 模型权重
├── game/                 # 本机游戏文件，不进入仓库
├── datasets/             # 本机数据集，不进入仓库
├── docs/                 # 架构、路线和发布检查
└── tests/                # 无游戏输入的回归测试
```

参见：

- [项目状态](docs/PROJECT_STATUS.md)
- [架构与安全边界](docs/ARCHITECTURE.md)
- [开发路线](docs/ROADMAP.md)
- [发布检查清单](docs/RELEASE_CHECKLIST.md)
- [版本变更](CHANGELOG.md)

## 官方相关链接

- [上海爱丽丝幻乐团官方网站](http://www16.big.or.jp/~zun/)
- [《东方红魔乡 ～ the Embodiment of Scarlet Devil.》官方作品页](https://www16.big.or.jp/~zun/html/th06.html)

《东方 Project》《东方红魔乡》及相关名称、角色和游戏资源的权利归其各自权利人
所有。请通过合法渠道取得游戏，并遵守东方 Project 二次创作规则及所在地法律。

## 发布与许可

- 商业游戏文件、Wine 前缀、运行截图、用户配置和数据集不得进入发布包。
- `models/best.pt` 的训练数据来源、授权和权重再分发权仍需发布者确认。
- 当前仓库尚未提供开源许可证；确定许可证前，请勿将代码可用性误解为已获得
  复制、修改或再分发许可。
- 发布前请完成 `docs/RELEASE_CHECKLIST.md`。
