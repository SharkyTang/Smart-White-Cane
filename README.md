# Smart White Cane Lab

这是一个独立的个人学习原型，用 Mac 摄像头先验证智能导盲杖最核心的软件闭环：

```text
Camera / Video
      ↓
Object Detection
      ↓
Position + Risk Assessment
      ↓
Visual Overlay + Optional Audio Alert
```

当前版本是 **Milestone 0.1**。它不是医疗器械或可独立依赖的导航设备，不应用于真实道路安全决策。

## 当前范围

- Mac 摄像头或本地视频输入
- Ultralytics YOLO 预训练模型
- 左 / 中 / 右空间位置判断
- 低 / 中 / 高风险规则
- macOS `say` 语音提醒与冷却时间
- 纯 Python 风险逻辑单元测试
- 无摄像头也能运行的 `--demo` 模式

首版暂不做 GPS、SLAM、手机 App、云端服务、自训练数据集或真实用户测试。

## 安装

项目使用 `uv` 管理独立的 Python 3.12 环境：

```bash
uv sync
```

首次运行 YOLO 时会下载轻量模型权重，因此需要网络连接。

## 先验证纯软件闭环

```bash
uv run smart-cane --demo
```

带语音运行：

```bash
uv run smart-cane --demo --audio
```

## 使用 Mac 摄像头

```bash
uv run smart-cane --source 0
```

同时启用语音：

```bash
uv run smart-cane --source 0 --audio
```

窗口中按 `q` 或 `Esc` 退出。macOS 第一次访问摄像头时会请求权限；若未出现画面，请在“系统设置 → 隐私与安全性 → 摄像头”中允许你使用的终端或 Codex。

## 使用图片或视频文件

```bash
uv run smart-cane --source /absolute/path/to/video.mp4 --no-audio
```

单张图片也可以作为一次性输入：

```bash
uv run smart-cane --source /absolute/path/to/image.jpg --device cpu --no-display
```

## 测试

```bash
uv run python -m unittest discover -s tests -v
```

## 工程结构

```text
src/smart_cane/
├── perception/   # 摄像头与目标检测
├── decision/     # 位置和风险判断
├── feedback/     # 语音反馈
├── app.py        # 实时处理闭环
├── cli.py        # 命令行入口
├── config.py     # 运行配置
└── models.py     # 共享数据模型
tests/            # 不依赖摄像头和模型的单元测试
docs/             # 架构与决策记录
```

## 下一步

1. 记录 FPS、推理延迟、误报和提醒次数。
2. 给检测结果加入连续帧稳定性，减少闪烁与重复播报。
3. 实验单目深度估计，或用已知距离数据校准风险规则。
4. 把摄像头、距离传感器和反馈设备保持为可替换适配器，方便迁移至 Raspberry Pi。
