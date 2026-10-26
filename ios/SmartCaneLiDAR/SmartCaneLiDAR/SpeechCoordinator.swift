import AVFAudio
import Foundation

struct SpeechStartEvent: Sendable {
    let frameID: Int
    let alertID: String
    let speechStartLatencyMilliseconds: Double
    let spokenText: String
    let spokenDistanceMeters: Double?
    let distanceStatus: String
    let riskSource: String
    let level: String
}

@MainActor
final class SpeechCoordinator: NSObject, @preconcurrency AVSpeechSynthesizerDelegate {
    var onSpeechStarted: ((SpeechStartEvent) -> Void)?

    private struct Pending {
        let frameID: Int
        let alertID: String
        let captureTimestampSeconds: Double
        let spokenText: String
        let spokenDistanceMeters: Double?
        let distanceStatus: String
        let riskSource: String
        let level: String
        let acknowledge: () -> Void
    }

    private let synthesizer = AVSpeechSynthesizer()
    private var pendingByUtterance: [ObjectIdentifier: Pending] = [:]
    private var pendingAlertIDs: Set<String> = []
    private var startedAlertIDs: Set<String> = []
    private var activeUtteranceID: ObjectIdentifier?
    private var activeLevel: String?

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    func reset() {
        synthesizer.stopSpeaking(at: .immediate)
        pendingByUtterance.removeAll()
        pendingAlertIDs.removeAll()
        startedAlertIDs.removeAll()
        activeUtteranceID = nil
        activeLevel = nil
        deactivateAudioSession()
    }

    func handle(
        alert: AlertPayload,
        alertID: String,
        frameID: Int,
        captureTimestampSeconds: Double,
        distanceMeters: Double?,
        distanceStatus: String,
        riskSource: String,
        acknowledge: @escaping () -> Void
    ) {
        guard alert.shouldEmit else { return }
        if startedAlertIDs.contains(alertID) {
            acknowledge()
            return
        }
        guard !pendingAlertIDs.contains(alertID) else { return }

        prepareAudioSession()
        let spokenText = Self.spokenMessage(alert: alert, distanceMeters: distanceMeters)
        let utterance = AVSpeechUtterance(string: spokenText)
        utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN")
        utterance.rate = 0.48
        utterance.volume = 1
        let pending = Pending(
            frameID: frameID,
            alertID: alertID,
            captureTimestampSeconds: captureTimestampSeconds,
            spokenText: spokenText,
            spokenDistanceMeters: distanceMeters,
            distanceStatus: distanceStatus,
            riskSource: riskSource,
            level: alert.level,
            acknowledge: acknowledge
        )
        pendingByUtterance[ObjectIdentifier(utterance)] = pending
        pendingAlertIDs.insert(alertID)
        let wasSpeaking = synthesizer.isSpeaking
        let shouldInterrupt = wasSpeaking
            && SpeechInterruptionPolicy.shouldInterrupt(
                currentLevel: activeLevel,
                incomingLevel: alert.level
            )
        if shouldInterrupt {
            synthesizer.stopSpeaking(at: .immediate)
        }
        synthesizer.speak(utterance)
        if !wasSpeaking || shouldInterrupt {
            activeUtteranceID = ObjectIdentifier(utterance)
            activeLevel = alert.level
        }
    }

    func testSound() {
        prepareAudioSession()
        synthesizer.stopSpeaking(at: .immediate)
        let utterance = AVSpeechUtterance(string: "声音测试正常")
        utterance.voice = AVSpeechSynthesisVoice(language: "zh-CN")
        utterance.rate = 0.48
        synthesizer.speak(utterance)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        guard let pending = pendingByUtterance[ObjectIdentifier(utterance)] else { return }
        activeUtteranceID = ObjectIdentifier(utterance)
        activeLevel = pending.level
        pendingAlertIDs.remove(pending.alertID)
        startedAlertIDs.insert(pending.alertID)
        let latency = max(
            0,
            (ProcessInfo.processInfo.systemUptime - pending.captureTimestampSeconds) * 1_000
        )
        onSpeechStarted?(
            SpeechStartEvent(
                frameID: pending.frameID,
                alertID: pending.alertID,
                speechStartLatencyMilliseconds: latency,
                spokenText: pending.spokenText,
                spokenDistanceMeters: pending.spokenDistanceMeters,
                distanceStatus: pending.distanceStatus,
                riskSource: pending.riskSource,
                level: pending.level
            )
        )
        pending.acknowledge()
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        let utteranceID = ObjectIdentifier(utterance)
        pendingByUtterance.removeValue(forKey: utteranceID)
        if activeUtteranceID == utteranceID {
            activeUtteranceID = nil
            activeLevel = nil
        }
        if pendingByUtterance.isEmpty { deactivateAudioSession() }
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        let utteranceID = ObjectIdentifier(utterance)
        if let pending = pendingByUtterance.removeValue(forKey: utteranceID),
           !startedAlertIDs.contains(pending.alertID) {
            pendingAlertIDs.remove(pending.alertID)
        }
        if activeUtteranceID == utteranceID {
            activeUtteranceID = nil
            activeLevel = nil
        }
        if pendingByUtterance.isEmpty { deactivateAudioSession() }
    }

    static func spokenMessage(alert: AlertPayload, distanceMeters: Double?) -> String {
        let urgency = alert.level == "high" ? "危险" : "注意"
        let direction = ["left": "左侧", "center": "前方", "right": "右侧"][alert.direction] ?? "附近"
        let label = chineseLabel(alert.label)
        // The caller passes a value only for fresh, valid LiDAR evidence. Never
        // fall back to an alert-carried number for area-based risk.
        let distance = DistanceFormatter.speechText(distanceMeters)
        return [urgency, "\(direction)有\(label)", distance].compactMap { $0 }.joined(separator: "，")
    }

    private static func chineseLabel(_ label: String) -> String {
        let mapping = [
            "person": "行人", "bicycle": "自行车", "car": "汽车", "motorcycle": "摩托车",
            "bus": "公交车", "truck": "卡车", "chair": "椅子", "bench": "长椅",
            "dog": "狗", "cat": "猫", "suitcase": "行李箱", "backpack": "背包",
            "umbrella": "雨伞", "bottle": "瓶子", "dining table": "餐桌",
            "couch": "沙发", "potted plant": "盆栽", "traffic light": "红绿灯",
            "stop sign": "停止标志",
        ]
        if label.range(of: "[\\u{3400}-\\u{9FFF}]", options: .regularExpression) != nil {
            return label
        }
        return mapping[label.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()] ?? "障碍物"
    }

    private func prepareAudioSession() {
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
        try? session.setActive(true)
    }

    private func deactivateAudioSession() {
        try? AVAudioSession.sharedInstance().setActive(
            false,
            options: [.notifyOthersOnDeactivation]
        )
    }
}

enum SpeechInterruptionPolicy {
    static func shouldInterrupt(currentLevel: String?, incomingLevel: String) -> Bool {
        incomingLevel == "high"
            && currentLevel.map { ["low", "medium"].contains($0) } == true
    }
}
