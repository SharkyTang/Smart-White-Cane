import Combine
import Foundation
import SwiftUI

@MainActor
final class AppModel: ObservableObject {
    private struct FramePresentation {
        let previewImage: CGImage
        let overlays: [OverlayItem]
    }

    @Published private(set) var isRunning = false
    @Published private(set) var isStarting = false
    @Published private(set) var isRecovering = false
    @Published private(set) var captureState: CaptureState = .stopped
    @Published private(set) var statusText = "等待开始"
    @Published private(set) var errorText: String?
    @Published private var framePresentation: FramePresentation?
    @Published private(set) var objectCount = 0
    @Published private(set) var latestDistanceText = "距离不可用"
    @Published private(set) var urgentTargetText = "暂无稳定目标"
    @Published private(set) var urgentRiskText = "—"
    @Published private(set) var forwardDistanceText = "距离不可用"
    @Published private(set) var depthQualityText = "unavailable · 覆盖 0% · 0 样本"
    @Published private(set) var roundTripText = "—"
    @Published private(set) var thermalText = "正常"
    @Published private(set) var exportURL: URL?
    @Published var loggingEnabled = false
    @Published var scenarioID = ""
    @Published var validationRunID = ""
    @Published var groundTruthText = ""
    @Published var validationGroup = "ordinary"
    @Published var expectedAlert = false
    @Published var expectedAlertEventID = ""
    @Published var useSmoothedDepth = false

    var previewImage: CGImage? { framePresentation?.previewImage }
    var overlays: [OverlayItem] { framePresentation?.overlays ?? [] }

    let lidarSupported = ARFrameCaptureService.supportsSceneDepth
    let smoothedDepthSupported = ARFrameCaptureService.supportsSmoothedSceneDepth

    private let capture = ARFrameCaptureService()
    private let speech = SpeechCoordinator()
    private let logger = ValidationLogger()
    private var transport: FrameTransport?
    private var sequenceGate = FrameSequenceGate()
    private var sessionID = ""
    private var latestCaptureTimestampByFrame: [Int: Double] = [:]
    private var lastServerAddress = ""
    private var resumeOnActive = false
    private var interruptedNeedsReset = false
    private var thermalPaused = false
    private var thermalCancellable: AnyCancellable?
    private var sceneIsActive = true
    private var connectionGeneration = 0
    private var connectRequested = false
    private var reconnectAttempt = 0
    private var reconnectTask: Task<Void, Never>?

    init() {
        capture.onFrame = { [weak self] frame in
            Task { @MainActor [weak self] in await self?.process(frame) }
        }
        capture.onStateChange = { [weak self] state in
            Task { @MainActor [weak self] in self?.handleCaptureState(state) }
        }
        capture.onError = { [weak self] error in
            Task { @MainActor [weak self] in self?.handleCaptureError(error) }
        }
        capture.onFatalError = { [weak self] error in
            Task { @MainActor [weak self] in self?.handleFatalCaptureError(error) }
        }
        speech.onSpeechStarted = { [weak self] event in
            Task { [weak self] in
                guard let self else { return }
                try? await self.logger.recordSpeech(event)
                await MainActor.run { self.refreshExportURL() }
            }
        }
        thermalCancellable = NotificationCenter.default.publisher(
            for: ProcessInfo.thermalStateDidChangeNotification
        ).receive(on: DispatchQueue.main).sink { [weak self] _ in
            self?.applyThermalState(ProcessInfo.processInfo.thermalState)
        }
        applyThermalState(ProcessInfo.processInfo.thermalState)
    }

    func start(serverAddress: String) async {
        guard !thermalPaused else {
            statusText = "设备过热，降温后才能启动"
            return
        }
        reconnectTask?.cancel()
        reconnectTask = nil
        reconnectAttempt = 0
        isRecovering = false
        await connect(serverAddress: serverAddress)
    }

    private func connect(serverAddress: String) async {
        guard sceneIsActive, !thermalPaused, !isRunning else { return }
        if isStarting {
            connectRequested = true
            return
        }
        isStarting = true
        connectionGeneration += 1
        let generation = connectionGeneration
        defer {
            isStarting = false
            let shouldReconnect = connectRequested
                && sceneIsActive
                && !thermalPaused
                && !isRunning
            connectRequested = false
            if shouldReconnect {
                Task { await self.start(serverAddress: self.lastServerAddress) }
            }
        }
        errorText = nil
        lastServerAddress = serverAddress.trimmingCharacters(in: .whitespacesAndNewlines)
        guard lidarSupported else {
            statusText = "设备不支持 LiDAR 场景深度"
            errorText = CaptureError.lidarUnavailable.localizedDescription
            captureState = .unsupported
            return
        }

        do {
            guard let url = URL(string: lastServerAddress) else {
                throw FrameTransportError.invalidServerURL
            }
            let transport = try FrameTransport(baseURL: url)
            statusText = "正在连接 Mac…"
            let newSessionID = try await transport.reset()
            guard generation == connectionGeneration, sceneIsActive, !thermalPaused else { return }
            self.transport = transport
            sessionID = newSessionID
            sequenceGate.reset()
            speech.reset()
            latestCaptureTimestampByFrame.removeAll()
            do {
                try await configureLogger()
            } catch {
                loggingEnabled = false
                errorText = "验证日志无法创建，已关闭记录：\(error.localizedDescription)"
            }
            guard generation == connectionGeneration, sceneIsActive, !thermalPaused else { return }
            interruptedNeedsReset = false
            capture.setUseSmoothedDepth(useSmoothedDepth)
            try capture.start(sessionID: newSessionID)
            isRunning = true
            isRecovering = false
            reconnectAttempt = 0
            resumeOnActive = false
            statusText = "正在启动 LiDAR…"
        } catch {
            guard generation == connectionGeneration, sceneIsActive, !thermalPaused else { return }
            isRunning = false
            if error is CaptureError || error is SCF1PacketError {
                statusText = "启动失败"
                errorText = error.localizedDescription
                return
            }
            if case FrameTransportError.invalidServerURL = error {
                statusText = "启动失败"
                errorText = error.localizedDescription
                return
            }
            scheduleReconnect(after: error)
        }
    }

    func stop() {
        connectionGeneration += 1
        connectRequested = false
        reconnectTask?.cancel()
        reconnectTask = nil
        reconnectAttempt = 0
        isRecovering = false
        resumeOnActive = false
        interruptedNeedsReset = false
        thermalPaused = ProcessInfo.processInfo.thermalState == .critical
        isRunning = false
        capture.stop()
        speech.reset()
        sequenceGate.reset()
        latestCaptureTimestampByFrame.removeAll()
        clearDetectionResults()
        statusText = "已停止"
    }

    func testSound() { speech.testSound() }

    func updateLoggingConfiguration() {
        Task { try? await configureLogger() }
    }

    func prepareExport() {
        Task {
            let url = await logger.exportURL()
            await MainActor.run { self.exportURL = url }
        }
    }

    func handleScenePhase(_ phase: ScenePhase) {
        switch phase {
        case .background:
            sceneIsActive = false
            guard isRunning || isStarting || isRecovering else { return }
            connectionGeneration += 1
            reconnectTask?.cancel()
            reconnectTask = nil
            isRecovering = false
            resumeOnActive = true
            isRunning = false
            capture.stop()
            speech.reset()
            sequenceGate.reset()
            latestCaptureTimestampByFrame.removeAll()
            clearDetectionResults()
            statusText = "后台已暂停，返回后重新连接"
        case .active:
            sceneIsActive = true
            guard resumeOnActive, !thermalPaused else { return }
            Task { await start(serverAddress: lastServerAddress) }
        case .inactive:
            // Permission prompts and other short system interruptions can make the
            // scene inactive without taking the app to the background.
            break
        @unknown default:
            break
        }
    }

    private func process(_ frame: RGBDFrame) async {
        guard isRunning, let transport else {
            capture.completeRequest(sessionID: frame.sessionID, frameID: frame.frameID)
            return
        }
        sequenceGate.markSent(frame.frameID)
        latestCaptureTimestampByFrame[frame.frameID] = frame.timestampSeconds
        var metadata = frame.metadata
        metadata.scenarioID = scenarioID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? nil : scenarioID.trimmingCharacters(in: .whitespacesAndNewlines)
        let packet = SCF1Packet(
            metadata: metadata,
            jpegData: frame.jpegData,
            depthData: frame.depthData,
            confidenceData: frame.confidenceData
        )

        defer {
            capture.completeRequest(sessionID: frame.sessionID, frameID: frame.frameID)
        }
        do {
            let result = try await transport.send(packet)
            guard isRunning, frame.sessionID == sessionID else { return }
            guard sequenceGate.accept(responseFrameID: result.response.frameID) else {
                statusText = "已忽略旧帧响应"
                return
            }
            errorText = nil
            statusText = "LiDAR + YOLO 检测中"
            // Publish the image and its server-derived overlays in one MainActor
            // transaction so a newer image is never shown with older boxes.
            framePresentation = FramePresentation(
                previewImage: frame.previewImage,
                overlays: result.response.overlayItems
            )
            objectCount = result.response.assessments.isEmpty
                ? result.response.detections.count : result.response.assessments.count
            roundTripText = String(format: "%.0f ms", result.roundTripMilliseconds)
            let nearest = result.response.assessments.compactMap(\.distanceMeters).min()
            latestDistanceText = DistanceFormatter.displayText(
                nearest,
                validationPrecision: loggingEnabled
            )
            updateUrgentSummary(result.response)
            try? await logger.recordFrame(result)
            refreshExportURL()

            if let alert = result.response.alert,
               alert.shouldEmit,
               let alertID = alert.alertID,
               let alertFrameID = alert.alertFrameID {
                let matchingAssessment = result.response.assessments.first {
                    $0.trackKey == alert.trackKey
                }
                let matchingDistance = matchingAssessment?.speechDistanceMeters
                let timestamp = latestCaptureTimestampByFrame[alertFrameID]
                    ?? frame.timestampSeconds
                speech.handle(
                    alert: alert,
                    alertID: alertID,
                    frameID: alertFrameID,
                    captureTimestampSeconds: timestamp,
                    distanceMeters: matchingDistance,
                    distanceStatus: matchingAssessment?.distance?.derivedStatus ?? "unavailable",
                    riskSource: matchingAssessment?.riskSource ?? "unknown",
                    acknowledge: { [weak self] in
                        Task { @MainActor [weak self] in
                            await self?.acknowledge(alertID: alertID, frameID: alertFrameID)
                        }
                    }
                )
            }
            latestCaptureTimestampByFrame = latestCaptureTimestampByFrame.filter {
                $0.key >= result.response.frameID - 8
            }
        } catch {
            guard isRunning, frame.sessionID == sessionID else { return }
            sequenceGate.clearInFlight(frame.frameID)
            if error is SCF1PacketError {
                isRunning = false
                capture.stop()
                statusText = "RGB-D 封包失败，已停止"
                errorText = error.localizedDescription
                return
            }
            scheduleReconnect(after: error)
        }
    }

    private func scheduleReconnect(after error: Error) {
        connectionGeneration += 1
        isRunning = false
        capture.stop()
        speech.reset()
        sequenceGate.reset()
        latestCaptureTimestampByFrame.removeAll()
        clearDetectionResults()
        reconnectTask?.cancel()
        reconnectAttempt += 1
        let delaySeconds = ReconnectBackoff.delaySeconds(forAttempt: reconnectAttempt)
        isRecovering = true
        errorText = error.localizedDescription
        statusText = "连接中断，\(delaySeconds) 秒后重置会话重试"
        let generation = connectionGeneration
        reconnectTask = Task { [weak self] in
            do {
                try await Task.sleep(for: .seconds(delaySeconds))
            } catch {
                return
            }
            guard !Task.isCancelled else { return }
            await self?.attemptReconnect(generation: generation)
        }
    }

    private func attemptReconnect(generation: Int) async {
        guard generation == connectionGeneration,
              sceneIsActive,
              !thermalPaused,
              isRecovering else { return }
        reconnectTask = nil
        await connect(serverAddress: lastServerAddress)
    }

    private func acknowledge(alertID: String, frameID: Int) async {
        guard let transport else { return }
        do {
            try await transport.acknowledge(
                sessionID: sessionID,
                frameID: frameID,
                alertID: alertID
            )
        } catch {
            errorText = "提醒 ACK 暂未送达，将在相同 alert_id 再次出现时重试"
        }
    }

    private func updateUrgentSummary(_ response: FrameResponse) {
        let alertTarget = response.alert?.trackKey.flatMap { key in
            response.assessments.first { $0.trackKey == key }
        }
        let urgent = alertTarget ?? response.assessments.max { lhs, rhs in
            Self.riskRank(lhs.level) < Self.riskRank(rhs.level)
        }
        guard let urgent else {
            urgentTargetText = "暂无稳定目标"
            urgentRiskText = "—"
            forwardDistanceText = "距离不可用"
            depthQualityText = "unavailable · 覆盖 0% · 0 样本"
            return
        }

        let direction = ["left": "左侧", "center": "前方", "right": "右侧"][urgent.direction]
            ?? urgent.direction
        let risk = ["low": "低", "medium": "中", "high": "高"][urgent.level]
            ?? urgent.level
        urgentTargetText = "\(urgent.label) · \(direction)"
        urgentRiskText = "\(risk)风险 · \(urgent.riskSource ?? "未知来源")"
        let forward = response.assessments
            .filter { $0.direction == "center" }
            .compactMap(\.distanceMeters)
            .min()
        forwardDistanceText = DistanceFormatter.displayText(
            forward,
            validationPrecision: loggingEnabled
        )
        let distance = urgent.distance
        let source = distance?.depthSource ?? "unknown_source"
        let quality = distance?.quality ?? "unavailable"
        let freshness = distance?.freshnessFrames ?? 0
        depthQualityText = String(
            format: "%@/%@ · %@ · 新鲜度 %d 帧 · 覆盖 %.0f%% · %d 样本",
            quality,
            distance?.derivedStatus ?? "unavailable",
            source,
            freshness,
            (distance?.coverageRatio ?? 0) * 100,
            distance?.sampleCount ?? 0
        )
    }

    private func clearDetectionResults() {
        framePresentation = nil
        objectCount = 0
        latestDistanceText = "距离不可用"
        urgentTargetText = "暂无稳定目标"
        urgentRiskText = "—"
        forwardDistanceText = "距离不可用"
        depthQualityText = "unavailable · 覆盖 0% · 0 样本"
        roundTripText = "—"
    }

    private static func riskRank(_ level: String) -> Int {
        ["low": 0, "medium": 1, "high": 2][level] ?? -1
    }

    private func configureLogger() async throws {
        let groundTruth = Double(groundTruthText)
        try await logger.configure(
            enabled: loggingEnabled,
            sessionID: sessionID,
            scenarioID: scenarioID,
            runID: validationRunID,
            groundTruthMeters: groundTruth,
            validationGroup: validationGroup,
            expectedAlert: expectedAlert,
            expectedAlertEventID: expectedAlertEventID
        )
        refreshExportURL()
    }

    private func refreshExportURL() {
        Task {
            let url = await logger.exportURL()
            await MainActor.run { self.exportURL = url }
        }
    }

    private func handleCaptureState(_ state: CaptureState) {
        captureState = state
        switch state {
        case .stopped:
            if !resumeOnActive, !thermalPaused, !isRecovering, !isStarting {
                statusText = "已停止"
            }
        case .starting:
            statusText = "正在初始化 ARKit 场景深度…"
            if interruptedNeedsReset {
                interruptedNeedsReset = false
                capture.stop()
                isRunning = false
                Task { await start(serverAddress: lastServerAddress) }
            }
        case .running:
            statusText = "LiDAR + YOLO 检测中"
        case .interrupted:
            connectionGeneration += 1
            isRunning = false
            sequenceGate.reset()
            speech.reset()
            resumeOnActive = true
            interruptedNeedsReset = true
            statusText = "相机被系统中断，恢复后将重置会话"
        case .unsupported:
            statusText = "设备不支持 LiDAR 场景深度"
        }
    }

    private func handleCaptureError(_ error: Error) {
        errorText = error.localizedDescription
        statusText = "传感器暂不可用"
    }

    private func handleFatalCaptureError(_ error: Error) {
        connectionGeneration += 1
        connectRequested = false
        reconnectTask?.cancel()
        reconnectTask = nil
        isRecovering = false
        isRunning = false
        isStarting = false
        resumeOnActive = false
        interruptedNeedsReset = false
        capture.stop()
        speech.reset()
        sequenceGate.reset()
        latestCaptureTimestampByFrame.removeAll()
        clearDetectionResults()
        errorText = error.localizedDescription
        statusText = "相机会话失败，已停止；重新开始会先重置会话"
    }

    private func applyThermalState(_ state: ProcessInfo.ThermalState) {
        switch state {
        case .nominal:
            thermalText = "正常 · 8 Hz"
            capture.setTargetFPS(8)
        case .fair:
            thermalText = "轻微升温 · 8 Hz"
            capture.setTargetFPS(8)
        case .serious:
            thermalText = "温度较高 · 已降至 5 Hz"
            capture.setTargetFPS(5)
        case .critical:
            thermalText = "过热 · 已暂停"
            thermalPaused = true
            if isRunning || isStarting || isRecovering {
                connectionGeneration += 1
                reconnectTask?.cancel()
                reconnectTask = nil
                isRecovering = false
                thermalPaused = true
                resumeOnActive = true
                isRunning = false
                capture.stop()
                speech.reset()
                sequenceGate.reset()
                latestCaptureTimestampByFrame.removeAll()
                clearDetectionResults()
                statusText = "设备过热，降温后自动恢复"
            }
        @unknown default:
            thermalText = "未知"
            capture.setTargetFPS(5)
        }

        if state != .critical, thermalPaused {
            thermalPaused = false
            if resumeOnActive, sceneIsActive {
                Task { await start(serverAddress: lastServerAddress) }
            }
        }
    }
}

enum ReconnectBackoff {
    static func delaySeconds(forAttempt attempt: Int) -> Int {
        guard attempt > 0 else { return 0 }
        return min(8, 1 << min(attempt - 1, 3))
    }
}
