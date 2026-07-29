# 发布检查清单

## 必须确认

- [x] 源代码与 Hugging Face 模型权重采用 GNU AGPL v3.0
- [x] 确认 `models/best.pt` 由项目作者使用自行整理和标注的数据训练
- [x] 将模型权重与模型卡发布到 `wcycn/touhou-ai-yolo`
- [x] 从 Hugging Face 回下载权重并核对 SHA-256
- [x] 将 1,834 张训练/验证图片与对应标签备份到私人 Dataset 仓库
- [x] 在 GitHub、模型仓库与私人数据集说明中添加交叉链接
- [x] 确认 Git 历史中没有游戏文件、API密钥、密码或个人路径
- [x] 确认 Git 只跟踪 `game/README.md` 与 `game/.gitkeep`
- [x] 确认 `sessions/`、`runs/`、设置、Wine前缀和用户截图未被跟踪
- [ ] 在一台干净的 Linux/X11 机器上按 README 完成安装
- [ ] 运行 `python touhou_ai.py check`
- [x] 运行 `python touhou_ai.py test`（58项通过）
- [x] 依次验证 GUI 的定位、观察、输入测试、AI启停和会话回放

## 推荐补齐

- [ ] 添加模型类别表、输入尺寸和验证集指标
- [x] 添加一段压缩后的真实会话演示，不包含游戏文件或训练数据
- [ ] 明确支持的发行版、Python、Wine和GPU版本
- [ ] 设置版本标签并生成校验和
- [x] 编写可直接用于 GitHub Release 的 v1.1.0 发行说明
- [ ] 在项目主页注明实验性质与全局键盘输入风险

## 建议发布内容

GitHub 只发布当前目录中的源码和文档，不跟踪 `models/best.pt`。权重由 Hugging
Face 托管，原始游戏截图数据不公开发布。不要将旧工程目录整体打包发布。
