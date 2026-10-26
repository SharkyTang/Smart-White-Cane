import Foundation

actor ValidationLogger {
    private var enabled = false
    private var sessionID = ""
    private var scenarioID: String?
    private var runID: String?
    private var groundTruthMeters: Double?
    private var validationGroup = "ordinary"
    private var expectedAlert = false
    private var expectedAlertEventID: String?
    private var fileURL: URL?

    func configure(
        enabled: Bool,
        sessionID: String,
        scenarioID: String?,
        runID: String?,
        groundTruthMeters: Double?,
        validationGroup: String,
        expectedAlert: Bool,
        expectedAlertEventID: String?
    ) throws {
        let sessionChanged = self.sessionID != sessionID
        self.enabled = enabled
        self.sessionID = sessionID
        self.scenarioID = scenarioID?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        self.runID = runID?.trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        self.groundTruthMeters = groundTruthMeters.flatMap { $0.isFinite && $0 > 0 ? $0 : nil }
        self.validationGroup = validationGroup == "boundary" ? "boundary" : "ordinary"
        self.expectedAlert = expectedAlert
        self.expectedAlertEventID = expectedAlertEventID?
            .trimmingCharacters(in: .whitespacesAndNewlines).nilIfEmpty
        if sessionChanged { fileURL = nil }
        guard enabled, !sessionID.isEmpty else { return }
        if fileURL == nil {
            fileURL = try Self.makeLogURL(sessionID: sessionID)
        }
    }

    func recordFrame(_ result: FrameTransportResult) throws {
        guard enabled, result.response.sessionID == sessionID else { return }
        let assessments = result.response.assessments.map { assessment in
            FrameLogAssessment(
                trackKey: assessment.trackKey,
                label: assessment.label,
                direction: assessment.direction,
                level: assessment.level,
                distanceMeters: assessment.distance?.meters,
                distanceStatus: assessment.distance?.derivedStatus ?? "unavailable",
                riskSource: assessment.riskSource,
                depthSource: assessment.distance?.depthSource,
                quality: assessment.distance?.quality,
                freshnessFrames: assessment.distance?.freshnessFrames,
                candidateStatus: assessment.distance?.candidateStatus,
                validRatio: assessment.distance?.coverageRatio ?? 0,
                sampleCount: assessment.distance?.sampleCount ?? 0,
                candidateP20Meters: assessment.distance?.candidateP20Meters,
                candidateMedianMeters: assessment.distance?.candidateMedianMeters,
                candidateCenterMedianMeters: assessment.distance?.candidateCenterMedianMeters
            )
        }
        let alert = result.response.alert.map {
            let matchingAssessment = $0.trackKey.flatMap { trackKey in
                result.response.assessments.first { $0.trackKey == trackKey }
            }
            return FrameLogAlert(
                alertID: $0.alertID,
                alertFrameID: $0.alertFrameID,
                shouldEmit: $0.shouldEmit,
                trackKey: $0.trackKey,
                label: $0.label,
                direction: $0.direction,
                level: $0.level,
                reason: $0.reason,
                riskSource: matchingAssessment?.riskSource,
                distanceStatus: matchingAssessment?.distance?.derivedStatus ?? "unavailable",
                distanceMeters: matchingAssessment?.distance?.meters
            )
        }
        try append(
            FrameLogRecord(
                recordType: "frame",
                sessionID: sessionID,
                scenarioID: scenarioID,
                runID: runID,
                groundTruthMeters: groundTruthMeters,
                validationGroup: validationGroup,
                expectedAlert: expectedAlert,
                expectedAlertEventID: expectedAlertEventID,
                modelName: result.response.modelName,
                frameID: result.requestFrameID,
                responseFrameID: result.response.frameID,
                captureTimestampSeconds: result.captureTimestampSeconds,
                depthSource: result.depthSource,
                roundTripMilliseconds: result.roundTripMilliseconds,
                assessments: assessments,
                alert: alert
            )
        )
    }

    func recordSpeech(_ event: SpeechStartEvent) throws {
        guard enabled else { return }
        try append(
            SpeechLogRecord(
                recordType: "speech",
                sessionID: sessionID,
                frameID: event.frameID,
                alertID: event.alertID,
                speechStartLatencyMilliseconds: event.speechStartLatencyMilliseconds,
                spokenText: event.spokenText,
                spokenDistanceMeters: event.spokenDistanceMeters,
                distanceStatus: event.distanceStatus,
                riskSource: event.riskSource,
                level: event.level
            )
        )
    }

    func exportURL() -> URL? { fileURL }

    private func append<T: Encodable>(_ value: T) throws {
        guard let fileURL else { return }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var line = try encoder.encode(value)
        line.append(0x0A)
        if !FileManager.default.fileExists(atPath: fileURL.path) {
            _ = FileManager.default.createFile(atPath: fileURL.path, contents: nil)
        }
        let handle = try FileHandle(forWritingTo: fileURL)
        try handle.seekToEnd()
        try handle.write(contentsOf: line)
        try handle.close()
    }

    private static func makeLogURL(sessionID: String) throws -> URL {
        let root = try FileManager.default.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appending(path: "SmartCaneLiDAR", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let url = root.appending(path: "validation-\(sessionID).jsonl")
        if !FileManager.default.fileExists(atPath: url.path) {
            _ = FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        return url
    }
}

private struct FrameLogRecord: Encodable {
    let recordType: String
    let sessionID: String
    let scenarioID: String?
    let runID: String?
    let groundTruthMeters: Double?
    let validationGroup: String
    let expectedAlert: Bool
    let expectedAlertEventID: String?
    let modelName: String?
    let frameID: Int
    let responseFrameID: Int
    let captureTimestampSeconds: Double
    let depthSource: String
    let roundTripMilliseconds: Double
    let assessments: [FrameLogAssessment]
    let alert: FrameLogAlert?

    enum CodingKeys: String, CodingKey {
        case recordType = "record_type"
        case sessionID = "session_id"
        case scenarioID = "scenario_id"
        case runID = "run_id"
        case groundTruthMeters = "ground_truth_meters"
        case validationGroup = "validation_group"
        case expectedAlert = "expected_alert"
        case expectedAlertEventID = "expected_alert_event_id"
        case modelName = "model_name"
        case frameID = "frame_id"
        case responseFrameID = "response_frame_id"
        case captureTimestampSeconds = "capture_timestamp_seconds"
        case depthSource = "depth_source"
        case roundTripMilliseconds = "rtt_ms"
        case assessments
        case alert
    }
}

private struct FrameLogAssessment: Encodable {
    let trackKey: String
    let label: String
    let direction: String
    let level: String
    let distanceMeters: Double?
    let distanceStatus: String
    let riskSource: String?
    let depthSource: String?
    let quality: String?
    let freshnessFrames: Int?
    let candidateStatus: String?
    let validRatio: Double
    let sampleCount: Int
    let candidateP20Meters: Double?
    let candidateMedianMeters: Double?
    let candidateCenterMedianMeters: Double?

    enum CodingKeys: String, CodingKey {
        case trackKey = "track_key"
        case label
        case direction
        case level
        case distanceMeters = "distance_meters"
        case distanceStatus = "distance_status"
        case riskSource = "risk_source"
        case depthSource = "depth_source"
        case quality
        case freshnessFrames = "freshness_frames"
        case candidateStatus = "candidate_status"
        case validRatio = "valid_ratio"
        case sampleCount = "sample_count"
        case candidateP20Meters = "candidate_p20_meters"
        case candidateMedianMeters = "candidate_median_meters"
        case candidateCenterMedianMeters = "candidate_center_median_meters"
    }
}

private struct FrameLogAlert: Encodable {
    let alertID: String?
    let alertFrameID: Int?
    let shouldEmit: Bool
    let trackKey: String?
    let label: String
    let direction: String
    let level: String
    let reason: String
    let riskSource: String?
    let distanceStatus: String
    let distanceMeters: Double?

    enum CodingKeys: String, CodingKey {
        case alertID = "alert_id"
        case alertFrameID = "alert_frame_id"
        case shouldEmit = "should_emit"
        case trackKey = "track_key"
        case label
        case direction
        case level
        case reason
        case riskSource = "risk_source"
        case distanceStatus = "distance_status"
        case distanceMeters = "distance_meters"
    }

    func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encodeIfPresent(alertID, forKey: .alertID)
        try values.encodeIfPresent(alertFrameID, forKey: .alertFrameID)
        try values.encode(shouldEmit, forKey: .shouldEmit)
        if let trackKey {
            try values.encode(trackKey, forKey: .trackKey)
        } else {
            try values.encodeNil(forKey: .trackKey)
        }
        try values.encode(label, forKey: .label)
        try values.encode(direction, forKey: .direction)
        try values.encode(level, forKey: .level)
        try values.encode(reason, forKey: .reason)
        if let riskSource {
            try values.encode(riskSource, forKey: .riskSource)
        } else {
            try values.encodeNil(forKey: .riskSource)
        }
        try values.encode(distanceStatus, forKey: .distanceStatus)
        if let distanceMeters {
            try values.encode(distanceMeters, forKey: .distanceMeters)
        } else {
            try values.encodeNil(forKey: .distanceMeters)
        }
    }
}

private struct SpeechLogRecord: Encodable {
    let recordType: String
    let sessionID: String
    let frameID: Int
    let alertID: String
    let speechStartLatencyMilliseconds: Double
    let spokenText: String
    let spokenDistanceMeters: Double?
    let distanceStatus: String
    let riskSource: String
    let level: String

    enum CodingKeys: String, CodingKey {
        case recordType = "record_type"
        case sessionID = "session_id"
        case frameID = "frame_id"
        case alertID = "alert_id"
        case speechStartLatencyMilliseconds = "speech_start_latency_ms"
        case spokenText = "spoken_text"
        case spokenDistanceMeters = "distance_meters"
        case distanceStatus = "distance_status"
        case riskSource = "risk_source"
        case level
    }
}

private extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}
