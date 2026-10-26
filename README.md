# Smart White Cane · 智能导盲杖学习原型

[GitHub 仓库](https://github.com/SharkyTang/Smart-White-Cane) · Python 3.12 · iOS 18+ · 当前包版本 `0.4.0`

这是一个独立的个人学习原型：用 Mac 摄像头、iPhone Safari 或
iPhone LiDAR 验证“检测障碍物 → 稳定目标 → 距离/风险 → 提醒”的软件
闭环。iPhone 负责同步 RGB-D 采集、显示和播报，YOLO 仍在同一局域网内的
Mac 上运行。

```text
Mac Camera / Video ─┐
iPhone Safari RGB ──├─> Mac YOLO Object Detection
iPhone ARKit RGB-D ─┘             + registered LiDAR depth
      ↓
Temporal Stability + Direction Hysteresis
      ↓
Distance Fusion + Metric/Area Risk
      ↓
Alert Deduplication
      ↓
Visual Overlay + Audio Alert
```

当前源码是 **Milestone 0.4 工程验证版**，默认检测模型为 `yolo26n.pt`。
自动化只能验证协议和策略；iPhone 16 Pro Max 的完整距离、配准、延迟和播报门槛
尚需按真机矩阵完成。
它不是医疗器械或可独立依赖的导航设备，不能用于真实道路安全决策。

## 当前状态与已知问题

- **已实现**：Mac 摄像头/图片/视频、iPhone Safari RGB、原生 iPhone LiDAR
  RGB-D 三种输入路径，以及目标稳定、风险规则、提醒去重和元数据验证工具。
- **已有局部历史结果**：YOLO11n 的 1.5 米椅子三轮采集已有记录；该结果只覆盖
  当时的模型、目标和距离，不能代表当前 YOLO26n 或完整 LiDAR 验收通过。
- **仍待修正与复测**：历史采集发现播报等待、过时米数提醒、类别混淆和处理率
  未达计划门槛。`0.4.1` 的队列、提醒新鲜度和风险回退修正，以及 YOLO26n
  的重新验收尚未完成，详见 [下一阶段计划](docs/milestone-0.4.1-next-stage-zh-CN.md)。

以下列表说明源码中已有的能力；真实设备上的可靠性以对应验收记录为准。

## 已实现功能

- iOS 18 SwiftUI RGB-D 客户端：同一 `ARFrame` 采集 RGB、`sceneDepth`
  和 `confidenceMap`
- 版本化 `SCF1` 传输、严格 2 MB/尺寸/帧序校验、单请求在途与最新帧替换
- 稳定框内缩 15% 后过滤低置信/无效深度，使用近端 20 分位数和
  3 个有效帧中位数
- 默认 1 m / 2 m 风险阈值、0.15 m 降级迟滞、无效深度面积降级
- 稳定 `alert_id` 与语音实际开口 ACK；丢失 ACK 可幂等重试且不重复播报
- 默认关闭的验证模式只导出元数据 JSONL，不保存 RGB/深度画面
- iPhone Safari 后置摄像头输入、手机识别框和中文语音 / 提示音提醒
- 手机帧只发送给同一局域网内的 Mac，内存处理后立即释放，不保存画面
- 串行传帧，不积压旧画面；手机断线时未送达提醒不会进入冷却
- Mac 摄像头、本地视频或单张图片输入
- Ultralytics YOLO26 预训练检测模型，默认 `yolo26n.pt`、`640 × 640`
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

本轮不做 Core ML 迁移、GPS、SLAM、触觉、道路、楼梯、坑洞、负障碍或
独立导航验证。LiDAR 只为已被 YOLO 识别的框提供“最近可靠表面”证据，
不会修复漏检或错误框。

## 快速开始

先准备 `uv` 和 Python 3.12。项目要求 Python `>=3.12,<3.13`，使用 `uv`
管理独立环境和锁定依赖。在终端克隆并安装：

```bash
git clone https://github.com/SharkyTang/Smart-White-Cane.git
cd Smart-White-Cane
uv sync --dev
uv run smart-cane --demo
```

如果已经在本地项目目录中，只需执行后两行，无需再次克隆。
安装依赖需要网络；首次使用尚未下载的 YOLO 模型时，也需要网络下载权重。
纯软件演示安装后不访问网络。

| 使用方式 | 额外条件 |
|---|---|
| 纯软件演示与 Python 自动化测试 | 不需要摄像头、LiDAR 或模型权重 |
| Mac 摄像头、图片与视频 | 模型权重；摄像头模式另需系统权限；macOS 语音使用 `say` |
| iPhone Safari | Mac 服务、同一可信局域网、iPhone 摄像头权限和可信 HTTPS |
| iPhone LiDAR | 上述局域网与 HTTPS 条件，以及 Xcode、iOS 18+、支持 LiDAR 的真机；当前验收目标为 iPhone 16 Pro Max |

## 先运行纯软件演示

```bash
uv run smart-cane --demo
```

带语音运行：

```bash
uv run smart-cane --demo --audio
```

演示不读取摄像头、不加载 YOLO，也不访问网络。

## 使用 iPhone LiDAR

LiDAR 需要原生 ARKit，模拟器不会产生真实深度。先按下方 Safari 部分生成、
安装并信任本地 CA，然后启动 Mac 服务：

```bash
uv run smart-cane-mobile
```

默认加载 `yolo26n.pt`。终端中的 `YOLO model: yolo26n.pt` 用于确认本次
配置；模型仍会在第一帧执行冷加载，因此开始验收前先等实时检测稳定。全新环境
还会先下载权重。仍可用 `--model` 显式指定其他兼容权重，但正式复测不要覆盖
默认值。

在 Xcode 中打开：

```text
ios/SmartCaneLiDAR/SmartCaneLiDAR.xcodeproj
```

选择自己的 Team，连接并信任 iPhone 16 Pro Max，开启 Developer Mode 后安装。
在 App 中输入终端显示的 `https://Mac局域网IP:8443`，先做声音测试，再点开始。
界面中的示例地址需要替换为自己的 Mac 地址；App 使用系统证书校验，不会绕过
HTTPS 信任失败。

第一次使用 Xcode，请从[中文版操作与验收手册](docs/milestone-0.4-lidar-validation-zh-CN.md)开始，
其中包含打开现有工程、选择 Team、安装到 iPhone、证书信任和逐项验收步骤。

验收时必须用三脚架和卷尺，并按
[0.4 LiDAR 技术验收规范（英文）](docs/milestone-0.4-lidar-validation.md) 执行。“验证模式”默认关闭；
每个场景填写 3 个不同轮次编号、每轮收集至少 30 个稳定帧，并为每个预期播报
填写唯一事件编号。导出的 JSONL 可生成报告：

```bash
uv run python scripts/analyze_lidar_validation.py \
  /absolute/path/to/export.jsonl \
  --output docs/milestone-0.4-lidar-results.md
```

## 使用 iPhone Safari 摄像头

iPhone Safari 只有在可信 HTTPS 页面中才能连续调用摄像头。以下设置只需在第一次使用、局域网 IP 变化或证书过期时处理。

1. 确保 Mac 和 iPhone 连接同一个可信 Wi-Fi，不要在公共网络或端口映射环境中运行。
2. 在项目目录生成只用于本地测试的证书：

   ```bash
   scripts/create_mobile_cert.sh
   ```

   如果脚本无法自动识别 Mac 的局域网 IP，可以手动传入：

   ```bash
   scripts/create_mobile_cert.sh 192.168.x.x
   ```

3. 通过隔空投送把 `.mobile-certs/mobile-ca.cer` 发到 iPhone。只发送这个公钥证书，绝不能发送 `.key` 文件。
4. 在 iPhone 中安装描述文件，然后进入“设置 → 通用 → 关于本机 → 证书信任设置”，为 `Smart Cane Local Test CA` 打开完全信任。
5. 在 Mac 启动服务：

   ```bash
   uv run smart-cane-mobile
   ```

6. 用 iPhone Safari 打开终端显示的 `https://Mac局域网IP:8443`，允许访问本地网络和摄像头，然后点“开始检测”。页面必须保持前台，锁屏或切到后台会暂停摄像头。

停止服务时在 Mac 终端按 `Control-C`。局域网 IP 变化时重新运行证书脚本即可；它会复用原来的本地 CA，因此通常不需要在 iPhone 重装描述文件。

只在 Mac 浏览器检查页面、不调用 iPhone 摄像头时，可显式使用：

```bash
uv run smart-cane-mobile --insecure-http --host 127.0.0.1
```

HTTP 模式不能用于 iPhone 摄像头。若不再测试，请在 iPhone 的“设置 → 通用 → VPN 与设备管理”中移除该描述文件，并删除 Mac 上的 `.mobile-certs/`；自建根证书私钥必须留在本机且不得分享。

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

下表中 `--distance-*` 参数仅用于 `smart-cane-mobile`；其余列出的参数可在
`smart-cane` 中使用，手机服务支持的参数以其 `--help` 输出为准。

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `--confirmation-window` | 5 | 普通目标的观察窗口 |
| `--minimum-confirmations` | 3 | 窗口内达到稳定所需次数 |
| `--high-risk-confirmations` | 2 | 大面积目标快速确认的连续帧数 |
| `--high-risk-area-ratio` | 0.18 | 快速确认与高风险分类共用的画面面积占比 |
| `--max-missing-frames` | 3 | 连续缺失后清除目标 |
| `--direction-hysteresis` | 0.05 | 左 / 中 / 右边界迟滞宽度 |
| `--alert-cooldown` | 3.0 秒 | 同一 track、类别、方向和风险的重复冷却 |
| `--distance-high` | 1.0 m | LiDAR 高风险上界 |
| `--distance-medium` | 2.0 m | LiDAR 中风险上界 |
| `--distance-hysteresis` | 0.15 m | 米制风险降级迟滞 |
| `--distance-max` | 5.0 m | 深度样本可用上界 |
| `--warmup-frames` | 5 | 从稳态统计中排除的开头帧数 |
| `--image-size` | 640 | YOLO 推理尺寸 |

旧参数 `--audio-cooldown` 仍可用，作为 `--alert-cooldown` 的兼容别名。查看所有参数：

```bash
uv run smart-cane --help
uv run smart-cane-mobile --help
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
它们属于旧模型的 0.2 历史记录，不是 YOLO26n 或 iPhone LiDAR 链路的性能结果。

## 测试与代码检查

```bash
uv run python -m unittest discover -s tests -v
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
node --check src/smart_cane/mobile/app.js
bash -n scripts/create_mobile_cert.sh
uv build
```

自动化测试不需要摄像头、LiDAR、模型下载或外部服务。真实 Mac 摄像头验收需要
单独运行上面的固定帧数命令；Safari 路径按
[0.3 验收记录](docs/milestone-0.3-mobile-validation.md) 核验；LiDAR 路径必须按
[0.4 验收记录](docs/milestone-0.4-lidar-validation.md) 在 iPhone 16 Pro Max 上完成，
模拟器不能替代。

## 工程结构

```text
src/smart_cane/
├── perception/       # 摄像头、视频和 YOLO 适配器
├── decision/         # 风险规则与连续帧稳定器
├── feedback/         # 提醒策略与 macOS 语音
├── mobile/           # iPhone Safari 页面
├── depth.py         # 注册深度采样、质量与 track 平滑
├── metrics.py        # 性能采样与稳态统计
├── rgbd_protocol.py  # SCF1 RGB-D 协议解析与校验
├── validation.py     # 元数据验收指标汇总
├── app.py            # 实时处理闭环
├── cli.py            # 命令行入口
├── mobile_cli.py     # 手机测试服务入口
├── mobile_server.py  # HTTPS、帧解码与手机响应
├── config.py         # 运行配置
└── models.py         # 共享数据模型
tests/                # 硬件无关的自动化测试
ios/SmartCaneLiDAR/   # iOS 18 ARKit / SwiftUI RGB-D 客户端
scripts/              # 本地证书和 LiDAR 验收报告工具
docs/                 # 架构、决策和验收记录
```

## 详细文档

- [架构说明](docs/architecture.md)：模块职责与 RGB / RGB-D 处理链路。
- [设计决策](docs/decisions.md)：关键选择和工程限制。
- [0.2 Mac 验收记录](docs/milestone-0.2-validation.md)：历史摄像头性能基线。
- [0.3 Safari 验收记录](docs/milestone-0.3-mobile-validation.md)：手机网页输入与提醒验证。
- [0.4 LiDAR 中文手册](docs/milestone-0.4-lidar-validation-zh-CN.md)：Xcode、证书和真机操作。
- [0.4 LiDAR 技术验收规范](docs/milestone-0.4-lidar-validation.md)：距离、配准、时延与播报门槛。
- [0.4.1 下一阶段计划](docs/milestone-0.4.1-next-stage-zh-CN.md)：历史问题、稳定性修正和 YOLO26n 复测。
- [项目工作记录](docs/project-log/README.md)：计划、改动和验证过程。

## GitHub 连接与后续上传

本地 Git 的 `origin` 远端用于持久关联此仓库。可以在项目目录查看连接和当前状态：

```bash
git remote -v
git status
```

后续修改文件后，在项目目录执行：

```bash
git add .
git commit -m "Describe your changes"
git push
```

`main` 配置上游后，`git push` 会推送到 `origin/main`。远端连接会保存在本地
Git 配置中；文件改动仍需要提交和推送才会出现在 GitHub。

仓库包含源码、iOS 工程、测试、脚本、文档和依赖锁文件。`.gitignore` 排除本地
虚拟环境、缓存、构建产物、Xcode 用户状态、私钥/证书目录、验证工作目录和可重新
下载的模型权重；运行时请按前述步骤在本机生成证书、下载模型。

## 下一阶段验证顺序

1. 先完成 0.3 Safari 真机回归，保证旧输入模式仍可独立运行。
2. 用三脚架、卷尺和静态不透明目标完成 0.5-5 m LiDAR 矩阵。
3. 只有普通目标的配准、误差、覆盖、延迟和错误米数播报都达标后，
   才执行动态阈值、断网和热状态测试。
4. 玻璃、镜面、深色布和细杆单独报告；失败场景不得宣称精确米数可用。
