import CoreGraphics
import Foundation

struct FrameResponse: Decodable, Sendable {
    let sessionID: String
    let frameID: Int
    let modelName: String?
    let detections: [DetectionPayload]
    let assessments: [AssessmentPayload]
    let alert: AlertPayload?
    let timing: TimingPayload?

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
        case frameID = "frame_id"
        case modelName = "model_name"
        case detections
        case assessments
        case alert
        case timing
    }
}

struct DetectionPayload: Decodable, Sendable, Identifiable {
    let label: String
    let confidence: Double
    let bbox: [Double]
    let frameWidth: Int
    let frameHeight: Int

    var id: String { "\(label):\(bbox.map(String.init(describing:)).joined(separator: ","))" }

    var rect: CGRect? {
        guard bbox.count == 4, bbox.allSatisfy(\.isFinite), bbox[2] >= bbox[0], bbox[3] >= bbox[1] else {
            return nil
        }
        return CGRect(x: bbox[0], y: bbox[1], width: bbox[2] - bbox[0], height: bbox[3] - bbox[1])
    }

    enum CodingKeys: String, CodingKey {
        case label
        case confidence
        case bbox
        case frameWidth = "frame_width"
        case frameHeight = "frame_height"
    }
}

struct DistancePayload: Decodable, Sendable {
    let meters: Double?
    let usable: Bool
    let isStale: Bool
    let sampleCount: Int
    let coverageRatio: Double
    let status: String?
    let depthSource: String?
    let quality: String?
    let freshnessFrames: Int?
    let candidateStatus: String?
    let candidateP20Meters: Double?
    let candidateMedianMeters: Double?
    let candidateCenterMedianMeters: Double?

    var derivedStatus: String {
        if let status { return status }
        if isStale { return "stale" }
        return usable ? "usable" : "unavailable"
    }

    enum CodingKeys: String, CodingKey {
        case meters
        case usable
        case isStale = "is_stale"
        case sampleCount = "sample_count"
        case coverageRatio = "coverage_ratio"
        case status
        case depthSource = "depth_source"
        case quality
        case freshnessFrames = "freshness_frames"
        case candidateStatus = "candidate_status"
        case candidateP20Meters = "candidate_p20_meters"
        case candidateMedianMeters = "candidate_median_meters"
        case candidateCenterMedianMeters = "candidate_center_median_meters"
        case legacyPercentile20Meters = "percentile_20_m"
        case legacyCandidateMedianMeters = "candidate_median_m"
        case legacyCandidateCenterMedianMeters = "candidate_center_median_m"
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        meters = try values.decodeIfPresent(Double.self, forKey: .meters)
        usable = try values.decodeIfPresent(Bool.self, forKey: .usable) ?? false
        isStale = try values.decodeIfPresent(Bool.self, forKey: .isStale) ?? false
        sampleCount = try values.decodeIfPresent(Int.self, forKey: .sampleCount) ?? 0
        coverageRatio = try values.decodeIfPresent(Double.self, forKey: .coverageRatio) ?? 0
        status = try values.decodeIfPresent(String.self, forKey: .status)
        depthSource = try values.decodeIfPresent(String.self, forKey: .depthSource)
        quality = try values.decodeIfPresent(String.self, forKey: .quality)
        freshnessFrames = try values.decodeIfPresent(Int.self, forKey: .freshnessFrames)
        candidateStatus = try values.decodeIfPresent(String.self, forKey: .candidateStatus)
        candidateP20Meters = try values.decodeIfPresent(Double.self, forKey: .candidateP20Meters)
            ?? values.decodeIfPresent(Double.self, forKey: .legacyPercentile20Meters)
        candidateMedianMeters = try values.decodeIfPresent(Double.self, forKey: .candidateMedianMeters)
            ?? values.decodeIfPresent(Double.self, forKey: .legacyCandidateMedianMeters)
        candidateCenterMedianMeters = try values.decodeIfPresent(Double.self, forKey: .candidateCenterMedianMeters)
            ?? values.decodeIfPresent(Double.self, forKey: .legacyCandidateCenterMedianMeters)
    }
}

struct AssessmentPayload: Decodable, Sendable, Identifiable {
    let trackKey: String
    let label: String
    let confidence: Double
    let bbox: [Double]
    let frameWidth: Int
    let frameHeight: Int
    let direction: String
    let level: String
    let shouldAlert: Bool
    let message: String
    let riskSource: String?
    let distance: DistancePayload?

    var id: String { trackKey }
    var distanceMeters: Double? {
        guard let distance,
              distance.usable,
              !distance.isStale,
              let meters = distance.meters,
              meters.isFinite,
              meters > 0 else { return nil }
        return meters
    }

    var speechDistanceMeters: Double? {
        guard riskSource == "lidar", distance?.derivedStatus == "valid" else { return nil }
        return distanceMeters
    }

    var rect: CGRect? {
        guard bbox.count == 4, bbox.allSatisfy(\.isFinite), bbox[2] >= bbox[0], bbox[3] >= bbox[1] else {
            return nil
        }
        return CGRect(x: bbox[0], y: bbox[1], width: bbox[2] - bbox[0], height: bbox[3] - bbox[1])
    }

    enum CodingKeys: String, CodingKey {
        case trackKey = "track_key"
        case label
        case confidence
        case bbox
        case frameWidth = "frame_width"
        case frameHeight = "frame_height"
        case direction
        case level
        case shouldAlert = "should_alert"
        case message
        case riskSource = "risk_source"
        case distance
    }
}

struct AlertPayload: Decodable, Sendable {
    let alertID: String?
    let alertFrameID: Int?
    let shouldEmit: Bool
    let reason: String
    let message: String
    let trackKey: String?
    let label: String
    let direction: String
    let level: String
    let distanceMeters: Double?

    enum CodingKeys: String, CodingKey {
        case alertID = "alert_id"
        case alertFrameID = "alert_frame_id"
        case shouldEmit = "should_emit"
        case reason
        case message
        case trackKey = "track_key"
        case label
        case direction
        case level
        case distanceMeters = "distance_meters"
    }
}

struct TimingPayload: Decodable, Sendable {
    let decodeMilliseconds: Double?
    let inferenceMilliseconds: Double?
    let decisionMilliseconds: Double?
    let totalMilliseconds: Double?

    enum CodingKeys: String, CodingKey {
        case decodeMilliseconds = "decode_ms"
        case inferenceMilliseconds = "inference_ms"
        case decisionMilliseconds = "decision_ms"
        case totalMilliseconds = "total_ms"
    }
}

struct OverlayItem: Identifiable, Sendable {
    let id: String
    let label: String
    let direction: String?
    let level: String
    let distanceMeters: Double?
    let rect: CGRect
    let frameSize: CGSize
}

extension FrameResponse {
    var overlayItems: [OverlayItem] {
        if !assessments.isEmpty {
            return assessments.compactMap { assessment in
                guard let rect = assessment.rect else { return nil }
                return OverlayItem(
                    id: assessment.trackKey,
                    label: assessment.label,
                    direction: assessment.direction,
                    level: assessment.level,
                    distanceMeters: assessment.distanceMeters,
                    rect: rect,
                    frameSize: CGSize(width: assessment.frameWidth, height: assessment.frameHeight)
                )
            }
        }
        return detections.compactMap { detection in
            guard let rect = detection.rect else { return nil }
            return OverlayItem(
                id: detection.id,
                label: detection.label,
                direction: nil,
                level: "low",
                distanceMeters: nil,
                rect: rect,
                frameSize: CGSize(width: detection.frameWidth, height: detection.frameHeight)
            )
        }
    }
}
