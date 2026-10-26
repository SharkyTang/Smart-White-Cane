import ARKit
import CoreImage
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

enum CaptureState: Equatable, Sendable {
    case stopped
    case starting
    case running
    case interrupted
    case unsupported
}

enum CaptureError: LocalizedError {
    case lidarUnavailable
    case smoothedDepthUnavailable
    case missingDepth
    case unsupportedDepthFormat
    case imageConversionFailed

    var errorDescription: String? {
        switch self {
        case .lidarUnavailable:
            return "此设备不支持 ARKit 场景深度，需要带 LiDAR 的 iPhone Pro"
        case .smoothedDepthUnavailable:
            return "此设备不支持平滑场景深度，请关闭调试 A/B 开关"
        case .missingDepth:
            return "当前帧没有可用的 LiDAR 深度"
        case .unsupportedDepthFormat:
            return "LiDAR 深度格式不是 Float32"
        case .imageConversionFailed:
            return "摄像头画面转换失败"
        }
    }
}

final class ARFrameCaptureService: NSObject, ARSessionDelegate, @unchecked Sendable {
    private struct Admission {
        let frameID: Int
        let sessionID: String
        let useSmoothedDepth: Bool
    }

    var onFrame: (@Sendable (RGBDFrame) -> Void)?
    var onStateChange: (@Sendable (CaptureState) -> Void)?
    var onError: (@Sendable (Error) -> Void)?
    var onFatalError: (@Sendable (Error) -> Void)?

    private let session = ARSession()
    private let captureQueue = DispatchQueue(label: "com.shaoqitang.SmartCaneLiDAR.capture", qos: .userInitiated)
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private let stateLock = NSLock()

    private var active = false
    private var interrupted = false
    private var requestInFlight = false
    private var inFlightFrameID: Int?
    private var nextFrameID = 0
    private var currentSessionID = ""
    private var lastAcceptedTimestamp = -Double.infinity
    private var nextEligibleTimestamp = -Double.infinity
    private var targetFPS = 8.0
    private var configuredUseSmoothedDepth = false
    private var activeUseSmoothedDepth = false
    private var pendingFrame = LatestFrameSlot<ARFrame>()

    override init() {
        super.init()
        session.delegate = self
        session.delegateQueue = captureQueue
    }

    static var supportsSceneDepth: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
    }

    static var supportsSmoothedSceneDepth: Bool {
        ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth)
    }

    func setUseSmoothedDepth(_ enabled: Bool) {
        stateLock.withLock { configuredUseSmoothedDepth = enabled }
    }

    func start(sessionID: String) throws {
        guard Self.supportsSceneDepth else {
            onStateChange?(.unsupported)
            throw CaptureError.lidarUnavailable
        }

        let useSmoothedDepth = stateLock.withLock {
            let selected = configuredUseSmoothedDepth
            active = true
            interrupted = false
            requestInFlight = false
            inFlightFrameID = nil
            pendingFrame.clear()
            nextFrameID = 0
            currentSessionID = sessionID
            lastAcceptedTimestamp = -Double.infinity
            nextEligibleTimestamp = -Double.infinity
            activeUseSmoothedDepth = selected
            return selected
        }
        guard !useSmoothedDepth || Self.supportsSmoothedSceneDepth else {
            stateLock.withLock { active = false }
            throw CaptureError.smoothedDepthUnavailable
        }
        onStateChange?(.starting)
        let configuration = ARWorldTrackingConfiguration()
        configuration.frameSemantics.insert(.sceneDepth)
        if useSmoothedDepth {
            configuration.frameSemantics.insert(.smoothedSceneDepth)
        }
        session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }

    func stop() {
        stateLock.withLock {
            active = false
            interrupted = false
            requestInFlight = false
            inFlightFrameID = nil
            pendingFrame.clear()
        }
        session.pause()
        onStateChange?(.stopped)
    }

    func completeRequest(sessionID: String, frameID: Int) {
        let pendingWork: (ARFrame, Admission)? = stateLock.withLock {
            guard currentSessionID == sessionID, inFlightFrameID == frameID else { return nil }
            requestInFlight = false
            inFlightFrameID = nil
            guard let latest = pendingFrame.take(),
                  active,
                  !interrupted,
                  latest.timestamp >= nextEligibleTimestamp else {
                return nil
            }
            return (latest, reserveLocked(timestamp: latest.timestamp))
        }
        if let pendingWork {
            captureQueue.async { [weak self] in
                self?.process(pendingWork.0, admission: pendingWork.1)
            }
        }
    }

    func setTargetFPS(_ framesPerSecond: Double) {
        stateLock.withLock {
            targetFPS = max(1, min(8, framesPerSecond))
            if lastAcceptedTimestamp.isFinite {
                nextEligibleTimestamp = lastAcceptedTimestamp + 1 / targetFPS
            }
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard selectedDepth(in: frame) != nil else { return }
        guard let admission = admit(frame) else { return }
        process(frame, admission: admission)
    }

    private func process(_ frame: ARFrame, admission: Admission) {
        guard let sceneDepth = admission.useSmoothedDepth
            ? frame.smoothedSceneDepth : frame.sceneDepth else {
            completeRequest(sessionID: admission.sessionID, frameID: admission.frameID)
            return
        }
        do {
            let image = try makePortraitJPEG(from: frame.capturedImage)
            let depth = try makePortraitDepth(from: sceneDepth.depthMap)
            let confidence = makePortraitConfidence(
                from: sceneDepth.confidenceMap,
                sourceWidth: CVPixelBufferGetWidth(sceneDepth.depthMap),
                sourceHeight: CVPixelBufferGetHeight(sceneDepth.depthMap)
            )
            let captured = RGBDFrame(
                sessionID: admission.sessionID,
                frameID: admission.frameID,
                timestampSeconds: frame.timestamp,
                jpegData: image.jpeg,
                depthData: depth.data,
                confidenceData: confidence,
                rgbWidth: image.width,
                rgbHeight: image.height,
                depthWidth: depth.width,
                depthHeight: depth.height,
                depthSource: admission.useSmoothedDepth
                    ? "smoothed_scene_depth" : "raw_scene_depth",
                previewImage: image.preview
            )
            onStateChange?(.running)
            onFrame?(captured)
        } catch {
            completeRequest(sessionID: admission.sessionID, frameID: admission.frameID)
            onError?(error)
        }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        stateLock.withLock {
            interrupted = true
            requestInFlight = false
            inFlightFrameID = nil
            pendingFrame.clear()
        }
        onStateChange?(.interrupted)
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        stateLock.withLock { interrupted = false }
        onStateChange?(.starting)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        stateLock.withLock {
            active = false
            requestInFlight = false
            inFlightFrameID = nil
            pendingFrame.clear()
        }
        session.pause()
        onFatalError?(error)
    }

    private func selectedDepth(in frame: ARFrame) -> ARDepthData? {
        stateLock.withLock {
            activeUseSmoothedDepth ? frame.smoothedSceneDepth : frame.sceneDepth
        }
    }

    private func admit(_ frame: ARFrame) -> Admission? {
        stateLock.withLock {
            guard active, !interrupted else { return nil }
            if requestInFlight {
                pendingFrame.replace(with: frame)
                return nil
            }
            guard frame.timestamp >= nextEligibleTimestamp else { return nil }
            return reserveLocked(timestamp: frame.timestamp)
        }
    }

    private func reserveLocked(timestamp: Double) -> Admission {
        let frameID = nextFrameID
        nextFrameID += 1
        lastAcceptedTimestamp = timestamp
        let interval = 1 / targetFPS
        if nextEligibleTimestamp.isFinite {
            nextEligibleTimestamp += interval
            if nextEligibleTimestamp <= timestamp {
                nextEligibleTimestamp = timestamp + interval
            }
        } else {
            nextEligibleTimestamp = timestamp + interval
        }
        requestInFlight = true
        inFlightFrameID = frameID
        return Admission(
            frameID: frameID,
            sessionID: currentSessionID,
            useSmoothedDepth: activeUseSmoothedDepth
        )
    }

    private func makePortraitJPEG(from pixelBuffer: CVPixelBuffer) throws
        -> (jpeg: Data, preview: CGImage, width: Int, height: Int)
    {
        let portrait = CIImage(cvPixelBuffer: pixelBuffer).oriented(.right)
        let maximumDimension = max(portrait.extent.width, portrait.extent.height)
        guard maximumDimension > 0 else { throw CaptureError.imageConversionFailed }
        let scale = min(1, 640 / maximumDimension)
        let scaled = portrait.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        let normalized = scaled.transformed(
            by: CGAffineTransform(translationX: -scaled.extent.minX, y: -scaled.extent.minY)
        )
        let extent = normalized.extent.integral
        guard let preview = ciContext.createCGImage(normalized, from: extent) else {
            throw CaptureError.imageConversionFailed
        }

        guard let encoded = CFDataCreateMutable(nil, 0),
              let destination = CGImageDestinationCreateWithData(
                  encoded,
                  UTType.jpeg.identifier as CFString,
                  1,
                  nil
              ) else {
            throw CaptureError.imageConversionFailed
        }
        CGImageDestinationAddImage(
            destination,
            preview,
            [kCGImageDestinationLossyCompressionQuality: 0.68] as CFDictionary
        )
        guard CGImageDestinationFinalize(destination) else {
            throw CaptureError.imageConversionFailed
        }
        return (encoded as Data, preview, preview.width, preview.height)
    }

    private func makePortraitDepth(from pixelBuffer: CVPixelBuffer) throws
        -> (data: Data, width: Int, height: Int)
    {
        guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_DepthFloat32 else {
            throw CaptureError.unsupportedDepthFormat
        }
        let sourceWidth = CVPixelBufferGetWidth(pixelBuffer)
        let sourceHeight = CVPixelBufferGetHeight(pixelBuffer)
        let outputWidth = sourceHeight
        let outputHeight = sourceWidth
        var output = [UInt16](repeating: 0, count: outputWidth * outputHeight)

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            throw CaptureError.missingDepth
        }
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        for y in 0..<sourceHeight {
            let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: Float32.self)
            for x in 0..<sourceWidth {
                let destinationX = sourceHeight - 1 - y
                let destinationY = x
                let sourceMeters = row[x]
                let halfMeters = sourceMeters.isFinite && sourceMeters > 0
                    ? Float16(sourceMeters) : Float16.zero
                let safeMeters = halfMeters.isFinite ? halfMeters : Float16.zero
                output[destinationY * outputWidth + destinationX] = safeMeters.bitPattern.littleEndian
            }
        }
        return (output.withUnsafeBytes { Data($0) }, outputWidth, outputHeight)
    }

    private func makePortraitConfidence(
        from pixelBuffer: CVPixelBuffer?,
        sourceWidth: Int,
        sourceHeight: Int
    ) -> Data {
        let outputWidth = sourceHeight
        let outputHeight = sourceWidth
        guard let pixelBuffer,
              CVPixelBufferGetWidth(pixelBuffer) == sourceWidth,
              CVPixelBufferGetHeight(pixelBuffer) == sourceHeight,
              CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_OneComponent8 else {
            return Data(repeating: 0, count: outputWidth * outputHeight)
        }

        var output = Data(repeating: 0, count: outputWidth * outputHeight)
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return output }
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        output.withUnsafeMutableBytes { destination in
            guard let target = destination.bindMemory(to: UInt8.self).baseAddress else { return }
            for y in 0..<sourceHeight {
                let row = base.advanced(by: y * rowBytes).assumingMemoryBound(to: UInt8.self)
                for x in 0..<sourceWidth {
                    let destinationX = sourceHeight - 1 - y
                    let destinationY = x
                    target[destinationY * outputWidth + destinationX] = row[x] <= 2 ? row[x] : 0
                }
            }
        }
        return output
    }
}

struct LatestFrameSlot<Element> {
    private(set) var value: Element?

    mutating func replace(with value: Element) {
        self.value = value
    }

    mutating func take() -> Element? {
        defer { value = nil }
        return value
    }

    mutating func clear() {
        value = nil
    }
}

private extension NSLock {
    @discardableResult
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
