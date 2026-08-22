# Smart White Cane Lab

这是一个独立的个人学习原型：先用 Mac 摄像头验证“看见障碍物 → 稳定判断 → 风险分级 → 提醒”的完整软件闭环，再考虑迁移到真实导盲杖硬件。

```text
Camera / Video
      ↓
YOLO Object Detection
      ↓
Temporal Stability + Direction Hysteresis
      ↓
Risk Assessment
      ↓
Alert Deduplication
      ↓
Visual Overlay + Optional Audio
```

当前版本是 **Milestone 0.2**。它不是医疗器械或可独立依赖的导航设备，不能用于真实道路安全决策。

## 0.2 已完成

- Mac 摄像头、本地视频或单张图片输入
- Ultralytics YOLO 预训练模型，默认 `yolo11n.pt`、`640 × 640`
- 左 / 中 / 右方向及低 / 中 / 高风险判断
- 5 帧窗口内 3 次确认，过滤单帧误检
- 大面积目标连续 2 帧快速确认
- 目标短暂丢失最多保留 2 帧，第 3 次连续漏检时清除；边界方向带迟滞防抖
- 同一目标同一提醒默认 3 秒冷却
- 风险从中升高、方向变化时立即提醒
- 多目标按风险顺序提醒，不让第二目标被冷却中的第一目标长期遮蔽
- 语音真正接受后才提交冷却；失败会在后续帧重试
- 冷启动、采集、推理、决策、渲染和完整帧耗时统计
- CPU 稳定默认；显式请求 MPS 但不可用时安全回退 CPU
- 不保存摄像头画面

暂不做 GPS、SLAM、手机 App、云端服务、自训练数据集、物理距离估计或真实用户道路测试。

## 安装

项目使用 `uv` 管理独立的 Python 3.12 环境：

```bash
uv sync --dev
```

首次运行 YOLO 时会下载轻量模型权重，因此需要网络连接。

## 先运行纯软件演示

```bash
uv run smart-cane --demo
```

带语音运行：

```bash
uv run smart-cane --demo --audio
```

演示不读取摄像头、不加载 YOLO，也不访问网络。

## 使用 Mac 摄像头

```bash
uv run smart-cane --source 0
```

同时启用 macOS 语音：

```bash
uv run smart-cane --source 0 --audio
```

窗口中按 `q` 或 `Esc` 退出。macOS 第一次访问摄像头时会请求权限；若未出现画面，请在“系统设置 → 隐私与安全性 → 摄像头”中允许 Codex 或你使用的终端，然后重启对应应用。

无窗口、固定帧数的可复现实验：

```bash
uv run smart-cane --source 0 --device cpu --no-display --max-frames 1800
```

## 使用图片或视频

```bash
uv run smart-cane --source /absolute/path/to/video.mp4 --no-display
uv run smart-cane --source /absolute/path/to/image.jpg --device cpu --no-display
```

单张图片只有冷启动样本。默认前 5 帧属于 warmup，因此程序会明确显示“没有稳态样本”，不会把 `0 FPS` 冒充成测量结果。

## 默认稳定与提醒参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--confirmation-window` | 5 | 普通目标的观察窗口 |
| `--minimum-confirmations` | 3 | 窗口内达到稳定所需次数 |
| `--high-risk-confirmations` | 2 | 大面积目标快速确认的连续帧数 |
| `--high-risk-area-ratio` | 0.18 | 快速确认与高风险分类共用的画面面积占比 |
| `--max-missing-frames` | 3 | 连续缺失后清除目标 |
| `--direction-hysteresis` | 0.05 | 左 / 中 / 右边界迟滞宽度 |
| `--alert-cooldown` | 3.0 秒 | 同一 track、类别、方向和风险的重复冷却 |
| `--warmup-frames` | 5 | 从稳态统计中排除的开头帧数 |
| `--image-size` | 640 | YOLO 推理尺寸 |

旧参数 `--audio-cooldown` 仍可用，作为 `--alert-cooldown` 的兼容别名。查看所有参数：

```bash
uv run smart-cane --help
```

## 性能输出怎么读

一次实时运行结束后会输出：

- `raw`：YOLO 每帧产生的原始框总数。
- `stable`：确认后送入风险引擎的目标帧次；短暂漏检会保留目标，因此它可能略高于 `raw`。
- `alert events`：提醒策略已成功交付的事件；关闭语音时表示逻辑提醒事件。
- `spoken`：macOS `say` 实际成功启动的次数。
- `cold inference`：第一次推理，包含模型惰性加载。
- `steady FPS`：排除 warmup 后的完整处理帧吞吐。
- `mean / median / p95`：稳态推理延迟分布。

`total` 是管线服务时间，包含采集、推理、决策、提醒调用、渲染及少量记账开销；各分阶段值不要求精确相加等于 `total`。

## 0.2 验收快照

2026-08-23 在当前 Mac/Codex 环境中，使用 CPU、摄像头、关闭窗口和语音、且不保存画面，完成了 1800 帧（约 60 秒）验收：

- 冷启动推理：`890.9 ms`
- 稳态处理：`30.0 FPS`
- 稳态推理 mean / median / p95：`16.8 / 16.7 / 17.4 ms`
- 原始 / 稳定目标帧次：`1074 / 1089`
- 去重后逻辑提醒事件：`25`
- 崩溃、摄像头中断、画面落盘：均未发生

这些数字是当前设备上的一次工程基线，不代表真实导盲安全效果。

## 测试与代码检查

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check src tests
uv run ruff format --check src tests
```

测试不需要摄像头、模型下载或外部服务。真实摄像头验收需要单独运行上面的固定帧数命令。

## 工程结构

```text
src/smart_cane/
├── perception/       # 摄像头、视频和 YOLO 适配器
├── decision/         # 风险规则与连续帧稳定器
├── feedback/         # 提醒策略与 macOS 语音
├── metrics.py        # 性能采样与稳态统计
├── app.py            # 实时处理闭环
├── cli.py            # 命令行入口
├── config.py         # 运行配置
└── models.py         # 共享数据模型
tests/                # 硬件无关的自动化测试
docs/                 # 架构、决策和验收记录
```

## 下一阶段候选方向

1. 建立可重复的视频场景集，记录漏检、误报和提醒时机，而不只看 FPS。
2. 给风险规则加入真实距离证据，优先尝试可替换的超声波或 ToF 传感器适配器。
3. 设计振动反馈接口，并与语音保持同一提醒策略。
4. 只有在有可验证数据后，再评估单目深度、自训练模型或 Raspberry Pi 部署。
