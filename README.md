# Touhou AI v1.1.0

一个运行在 Linux/X11 上的《东方红魔乡》视觉检测与规则控制原型。

当前链路：

```text
Wine游戏窗口 → MSS截图 → YOLO/PyTorch检测
→ 规则决策 → 窗口焦点守护 → PyAutoGUI输入
```

本目录是从历史工程中分离出的干净发布版，只保留新的统一 GUI、自动控制、
安全观察、窗口定位、输入测试、会话记录和离线回放。旧 GUI、远程部署脚本、
在线大模型客户端、历史密钥配置和游戏本体均不包含在内。

## 快速开始

系统要求：

- Linux X11 桌面
- Python 3.10 或更高版本
- Wine、`xdotool`、`xwininfo`
- Tkinter

Ubuntu/Debian 可先安装系统组件：

```bash
sudo apt install python3-venv python3-tk wine xdotool x11-utils
```

创建隔离环境并安装 Python 依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

将你合法取得的游戏文件放进 `game/`。程序会依次尝试
`vpatch.exe`、`th06c.exe`、`th06.exe`、`東方紅魔郷.exe`。

运行检查和测试：

```bash
python touhou_ai.py check
python touhou_ai.py test
```

启动 GUI：

```bash
python touhou_ai.py gui
```

在支持可执行脚本的文件管理器中，也可以双击 `启动控制中心.sh`。

## 路径约定

本目录可以整体移动或改名，不依赖启动时的当前工作目录。程序默认只使用本目录内的
以下位置：

- `models/best.pt`：YOLO 模型
- `game/`：本机游戏文件
- `datasets/`：本机训练或验证数据集
- `settings.json`：GUI 设置
- `sessions/`：AI 会话、抽样画面和报告
- `runs/`：游戏日志、观察截图和模型评估结果

请移动整个 `touhou_ai/`，不要只移动其中某个 Python 文件。命令行中明确传入的
数据集或导出目录仍以用户提供的位置为准。

## GUI 工作流

控制首页顶部有两个明确的AI入口：

- **AI模拟观察（不按键）**：运行完整的检测、自机/敌弹跟踪、轨迹预测、风险规划
  和动作决策，但输入模块被强制禁用。
- **AI自动控制（会按键）**：运行同一套AI，并将通过安全门控的动作发送给游戏。

红色“立即停止 AI / 模拟观察并释放按键”按钮在任一模式运行时都会启用。首页会用
绿色或红色状态文字持续标明当前是否正在发送按键，启动后不会自动跳离控制首页。

1. 启动游戏。
2. 定位窗口。
3. 运行AI模拟观察，确认自机、弹幕、轨迹和计划动作。
4. 点击红色按钮停止模拟观察。
5. 执行短暂输入测试。
6. 启动AI自动控制。

观察模式不会发送键盘或鼠标事件。正式 AI 会发送全局按键，并持续确认游戏窗口
焦点；无法确认焦点时不会执行动作。PyAutoGUI 的屏幕角落 failsafe 保持启用。

GUI 默认记录会话。`sessions/` 中会保存检测、状态、动作和抽样画面，可在
“会话与回放”页浏览并生成 JSON/Markdown 报告。新版还会记录自机跟踪来源、
敌弹速度、未来轨迹、碰撞风险、场景状态、候选方向代价和实际按键差异。

控制核心已经加入：

- 四边安全距离和方向切换冷却
- 只发送状态差异的按键状态机
- 自机短时预测与超时强制停控
- 敌弹跨帧速度估计、TTC和未来碰撞风险
- 保守战斗场景确认；未知、菜单和过场不发送动作
- 焦点丢失立即释放已持有按键

分析最新会话：

```bash
python touhou_ai.py analyze
```

分析并导出待人工检查的图片与YOLO预标注：

```bash
python touhou_ai.py analyze --export-review
```

预标注保存在 `prelabels/`，不会被当成正式 `labels/`。人工建立验证集后可以运行：

```bash
python touhou_ai.py model-eval datasets/validation/data.yaml
```

## 目录

```text
.
├── touhou_ai.py          # 统一入口
├── desktop_gui.py        # 唯一桌面 GUI
├── autopilot.py          # YOLO检测与规则控制
├── observe_game.py       # 无输入的安全观察
├── control_test.py       # 短按键诊断
├── launch_game.py        # Wine游戏启动器
├── window_controller.py  # X11窗口发现、定位和聚焦
├── session_recorder.py   # 记录、汇总、回放数据
├── session_analysis.py   # 会话指标和标注候选
├── control_logic.py      # 跟踪、规划、场景和按键状态
├── model_evaluation.py   # 人工标注集上的YOLO评估
├── models/best.pt        # 当前YOLO权重
├── game/                 # 用户自行放置游戏文件
├── datasets/             # 用户自行放置训练或验证数据集
├── docs/                 # 路线与发布检查
└── tests/                # 不操作游戏的回归测试
```

完整后续路线见 `docs/ROADMAP.md`，模块关系与输入安全边界见
`docs/ARCHITECTURE.md`。

## 发布前注意

- 本仓库不应包含商业游戏文件、Wine 前缀、运行录像或用户配置。
- 当前模型来源、数据授权和再分发权需要发布者确认。
- 项目尚未选定开源许可证；公开发布前请完成 `docs/RELEASE_CHECKLIST.md`。
- 这是实验性自动控制程序，尚未证明能够稳定生存或通关。
