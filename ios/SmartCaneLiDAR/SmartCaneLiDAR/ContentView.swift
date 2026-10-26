import SwiftUI

struct ContentView: View {
    @ObservedObject var model: AppModel
    @AppStorage("smartCaneServerURL") private var serverAddress = "https://192.168.1.2:8443"
    @State private var showValidation = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    statusCard
                    cameraCard
                    resultCard
                    urgentCard
                    controls
                    connectionCard
                    validationCard
                    safetyNotice
                }
                .padding()
            }
            .background(Color(.systemGroupedBackground))
            .navigationTitle("LiDAR 导盲验证")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    private var statusCard: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(statusColor)
                .frame(width: 12, height: 12)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 3) {
                Text(model.statusText).font(.headline)
                Text("温度：\(model.thermalText) · 往返：\(model.roundTripText)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
        .padding()
        .background(.background, in: RoundedRectangle(cornerRadius: 16))
        .accessibilityElement(children: .combine)
    }

    private var cameraCard: some View {
        GeometryReader { geometry in
            ZStack {
                Color.black
                if let image = model.previewImage {
                    Image(decorative: image, scale: 1)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: "camera.metering.matrix")
                            .font(.system(size: 38))
                        Text(model.lidarSupported ? "点击开始读取后置 LiDAR" : "此设备不支持场景深度")
                            .multilineTextAlignment(.center)
                    }
                    .foregroundStyle(.white.opacity(0.8))
                }
                ForEach(model.overlays) { item in
                    if let rect = CoordinateMapper.aspectFitRect(
                        forRGBBox: item.rect,
                        sourceSize: item.frameSize,
                        viewportSize: geometry.size
                    ) {
                        DetectionOverlay(item: item, rect: rect)
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 20))
            .overlay(RoundedRectangle(cornerRadius: 20).stroke(.white.opacity(0.18), lineWidth: 1))
        }
        .frame(height: 430)
        .accessibilityLabel("LiDAR 与 YOLO 检测画面")
    }

    private var resultCard: some View {
        HStack(spacing: 12) {
            metric(title: "稳定目标", value: "\(model.objectCount) 个", symbol: "viewfinder")
            metric(title: "最近距离", value: model.latestDistanceText, symbol: "ruler")
        }
    }

    private var urgentCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("最紧急目标", systemImage: "exclamationmark.shield")
                .font(.headline)
            Text(model.urgentTargetText)
                .font(.title3.bold())
            LabeledContent("方向与风险", value: model.urgentRiskText)
            LabeledContent("前向距离", value: model.forwardDistanceText)
            LabeledContent("深度质量", value: model.depthQualityText)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(.background, in: RoundedRectangle(cornerRadius: 16))
        .accessibilityElement(children: .combine)
    }

    private var controls: some View {
        HStack(spacing: 12) {
            Button {
                Task { await model.start(serverAddress: serverAddress) }
            } label: {
                Label("开始检测", systemImage: "play.fill")
                    .frame(maxWidth: .infinity, minHeight: 48)
            }
            .buttonStyle(.borderedProminent)
            .disabled(model.isRunning || model.isStarting || model.isRecovering || !model.lidarSupported)

            Button(role: .destructive) { model.stop() } label: {
                Label("停止", systemImage: "stop.fill")
                    .frame(maxWidth: .infinity, minHeight: 48)
            }
            .buttonStyle(.bordered)
            .disabled(!model.isRunning && !model.isStarting && !model.isRecovering)
        }
    }

    private var connectionCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Mac 检测服务", systemImage: "laptopcomputer.and.iphone")
                .font(.headline)
            TextField("https://Mac局域网IP:8443", text: $serverAddress)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
                .keyboardType(.URL)
                .textFieldStyle(.roundedBorder)
                .disabled(model.isRunning)
            HStack {
                Button("声音测试") { model.testSound() }
                Spacer()
                Text("只接受 HTTPS，不绕过证书验证")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let error = model.errorText {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .accessibilityLabel("错误：\(error)")
            }
        }
        .padding()
        .background(.background, in: RoundedRectangle(cornerRadius: 16))
    }

    private var validationCard: some View {
        DisclosureGroup("受控验证日志", isExpanded: $showValidation) {
            VStack(alignment: .leading, spacing: 10) {
                Toggle("记录元数据 JSONL", isOn: $model.loggingEnabled)
                    .onChange(of: model.loggingEnabled) { _, _ in model.updateLoggingConfiguration() }
                Toggle("调试 A/B：使用 smoothedSceneDepth", isOn: $model.useSmoothedDepth)
                    .disabled(
                        model.isRunning || model.isStarting || model.isRecovering
                            || !model.smoothedDepthSupported
                    )
                Text(model.useSmoothedDepth
                     ? "当前深度源：smoothed_scene_depth（仅调试）"
                     : "当前深度源：raw_scene_depth（正式默认）")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("场景编号，例如 chair-center-1m", text: $model.scenarioID)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { model.updateLoggingConfiguration() }
                TextField("本轮编号，例如 run-1", text: $model.validationRunID)
                    .textFieldStyle(.roundedBorder)
                TextField("卷尺真值（米，可选）", text: $model.groundTruthText)
                    .keyboardType(.decimalPad)
                    .textFieldStyle(.roundedBorder)
                Picker("材质分组", selection: $model.validationGroup) {
                    Text("普通不透明").tag("ordinary")
                    Text("边界材质").tag("boundary")
                }
                .pickerStyle(.segmented)
                Toggle("本场景预期应播报", isOn: $model.expectedAlert)
                TextField("预期事件编号，例如 cross-1m-in", text: $model.expectedAlertEventID)
                    .textFieldStyle(.roundedBorder)
                    .disabled(!model.expectedAlert)
                HStack {
                    Button("更新记录条件") { model.updateLoggingConfiguration() }
                    Button("准备导出") { model.prepareExport() }
                    if let url = model.exportURL {
                        ShareLink(item: url) { Label("导出", systemImage: "square.and.arrow.up") }
                    }
                }
                Text("每个组合使用 3 个不同轮次编号，每轮至少 30 个稳定帧；每个脚本化播报事件使用唯一事件编号。只记录元数据，不写入 RGB、深度图或音频。")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(.top, 10)
        }
        .padding()
        .background(.background, in: RoundedRectangle(cornerRadius: 16))
    }

    private var safetyNotice: some View {
        Label(
            "仅限静止、受控室内验证。不能用于道路、楼梯、坑洞或独立导航。",
            systemImage: "exclamationmark.triangle.fill"
        )
        .font(.footnote.weight(.semibold))
        .foregroundStyle(.orange)
        .padding(.vertical, 6)
    }

    private func metric(title: String, value: String, symbol: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Label(title, systemImage: symbol).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.bold()).monospacedDigit()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(.background, in: RoundedRectangle(cornerRadius: 16))
    }

    private var statusColor: Color {
        switch model.captureState {
        case .running: .green
        case .starting: .yellow
        case .interrupted: .orange
        case .unsupported: .red
        case .stopped: .gray
        }
    }
}

private struct DetectionOverlay: View {
    let item: OverlayItem
    let rect: CGRect

    var body: some View {
        ZStack(alignment: .topLeading) {
            Rectangle()
                .stroke(color, lineWidth: 4)
                .frame(width: rect.width, height: rect.height)
            Text(label)
                .font(.caption.bold())
                .foregroundStyle(.black)
                .padding(.horizontal, 7)
                .padding(.vertical, 4)
                .background(color)
                .fixedSize()
        }
        .frame(width: rect.width, height: rect.height, alignment: .topLeading)
        .position(x: rect.midX, y: rect.midY)
        .accessibilityLabel(label)
    }

    private var label: String {
        let distance = DistanceFormatter.displayText(item.distanceMeters)
        return "\(chineseLabel(item.label)) · \(distance)"
    }

    private var color: Color {
        switch item.level {
        case "high": .red
        case "medium": .yellow
        default: .cyan
        }
    }

    private func chineseLabel(_ label: String) -> String {
        let values = [
            "person": "行人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车",
            "bus": "公交车", "truck": "卡车", "chair": "椅子", "bench": "长椅",
            "dog": "狗", "cat": "猫", "suitcase": "行李箱", "backpack": "背包",
            "umbrella": "雨伞", "bottle": "瓶子", "dining table": "餐桌", "couch": "沙发",
        ]
        return values[label.lowercased()] ?? label
    }
}
