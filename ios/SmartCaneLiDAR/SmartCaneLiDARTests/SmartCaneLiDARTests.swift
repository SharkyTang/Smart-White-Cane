import CoreGraphics
import XCTest
@testable import SmartCaneLiDAR

final class SmartCaneLiDARTests: XCTestCase {
    func testSCF1RoundTripAndHeaderLengths() throws {
        let metadata = SCF1Metadata(
            sessionID: "session-1",
            frameID: 7,
            rgbWidth: 480,
            rgbHeight: 640,
            depthWidth: 192,
            depthHeight: 256,
            timestampSeconds: 12.5,
            source: "arkit_scene_depth",
            orientation: "portrait_up"
        )
        let jpeg = Data([0xFF, 0xD8, 0xFF, 0xD9])
        let depth = Data(repeating: 0x12, count: 192 * 256 * 2)
        let confidence = Data(repeating: 2, count: 192 * 256)
        let packet = SCF1Packet(
            metadata: metadata,
            jpegData: jpeg,
            depthData: depth,
            confidenceData: confidence
        )

        let encoded = try packet.encoded()
        XCTAssertEqual(Data(encoded.prefix(4)), Data("SCF1".utf8))
        XCTAssertEqual(readBigEndianUInt32(encoded, at: 8), UInt32(jpeg.count))
        XCTAssertEqual(readBigEndianUInt32(encoded, at: 12), UInt32(depth.count))
        XCTAssertEqual(readBigEndianUInt32(encoded, at: 16), UInt32(confidence.count))
        let metadataLength = Int(readBigEndianUInt32(encoded, at: 4))
        let metadataJSON = try XCTUnwrap(
            JSONSerialization.jsonObject(with: encoded.subdata(in: 20..<(20 + metadataLength)))
                as? [String: Any]
        )
        XCTAssertEqual(metadataJSON["schema_version"] as? Int, 1)
        XCTAssertEqual(metadataJSON["source"] as? String, "arkit_scene_depth")
        XCTAssertEqual(metadataJSON["depth_source"] as? String, "raw_scene_depth")
        XCTAssertEqual(metadataJSON["orientation"] as? String, "portrait_up")
        XCTAssertTrue(metadataJSON["scenario_id"] is NSNull)
        XCTAssertEqual(try SCF1Packet.decode(encoded), packet)
    }

    func testSCF1FixedVectorUsesBigEndianLengthsAndLittleEndianDepth() throws {
        let metadata = SCF1Metadata(
            sessionID: "s",
            frameID: 0,
            rgbWidth: 1,
            rgbHeight: 1,
            depthWidth: 1,
            depthHeight: 1,
            timestampSeconds: 1.25
        )
        let metadataJSON = Data(#"{"depth_height":1,"depth_source":"raw_scene_depth","depth_width":1,"frame_id":0,"orientation":"portrait_up","rgb_height":1,"rgb_width":1,"scenario_id":null,"schema_version":1,"session_id":"s","source":"arkit_scene_depth","timestamp_seconds":1.25}"#.utf8)
        let jpeg = Data([0xFF, 0xD8, 0xFF, 0xD9])
        let depth = Data([0x00, 0x3E]) // Float16(1.5), little-endian
        let confidence = Data([2])
        let encoded = try SCF1Packet(
            metadata: metadata,
            jpegData: jpeg,
            depthData: depth,
            confidenceData: confidence
        ).encoded()

        var expected = Data("SCF1".utf8)
        appendBigEndianUInt32(UInt32(metadataJSON.count), to: &expected)
        appendBigEndianUInt32(4, to: &expected)
        appendBigEndianUInt32(2, to: &expected)
        appendBigEndianUInt32(1, to: &expected)
        expected.append(metadataJSON)
        expected.append(jpeg)
        expected.append(depth)
        expected.append(confidence)
        XCTAssertEqual(encoded, expected)
    }

    func testSCF1RejectsWrongDepthLength() {
        let metadata = SCF1Metadata(
            sessionID: "session-1",
            frameID: 0,
            rgbWidth: 480,
            rgbHeight: 640,
            depthWidth: 2,
            depthHeight: 2
        )
        let packet = SCF1Packet(
            metadata: metadata,
            jpegData: Data([1]),
            depthData: Data(repeating: 0, count: 7),
            confidenceData: Data(repeating: 0, count: 4)
        )
        XCTAssertThrowsError(try packet.encoded()) { error in
            XCTAssertEqual(error as? SCF1PacketError, .invalidLength)
        }
    }

    func testSCF1RejectsPacketOverTwoMegabytes() {
        let metadata = SCF1Metadata(
            sessionID: "session-1",
            frameID: 0,
            rgbWidth: 1,
            rgbHeight: 1,
            depthWidth: 1,
            depthHeight: 1
        )
        let packet = SCF1Packet(
            metadata: metadata,
            jpegData: Data(repeating: 0xAA, count: 2_000_000),
            depthData: Data(repeating: 0, count: 2),
            confidenceData: Data([0])
        )
        XCTAssertThrowsError(try packet.encoded()) { error in
            XCTAssertEqual(error as? SCF1PacketError, .payloadTooLarge)
        }
    }

    func testFrameResponseParsesNestedDistanceAndAlertFrameID() throws {
        let json = #"{"session_id":"s","frame_id":9,"model_name":"yolo26n.pt","detections":[],"assessments":[{"track_key":"person:1","label":"person","confidence":0.9,"bbox":[10,20,110,220],"frame_width":480,"frame_height":640,"direction":"center","level":"high","should_alert":true,"message":"Danger","risk_source":"lidar","distance":{"meters":0.82,"usable":true,"status":"valid","depth_source":"raw_scene_depth","quality":"good","freshness_frames":0,"candidate_status":"valid","is_stale":false,"sample_count":42,"coverage_ratio":0.6,"percentile_20_m":0.75,"candidate_median_m":0.9,"candidate_center_median_m":0.85}}],"alert":{"alert_id":"a1","alert_frame_id":8,"should_emit":true,"reason":"first_alert","message":"Danger","track_key":"person:1","label":"person","direction":"center","level":"high","distance_meters":0.82},"timing":{"total_ms":24.5}}"#.data(using: .utf8)!
        let response = try JSONDecoder().decode(FrameResponse.self, from: json)
        XCTAssertEqual(response.frameID, 9)
        XCTAssertEqual(response.modelName, "yolo26n.pt")
        XCTAssertEqual(try XCTUnwrap(response.assessments.first?.distanceMeters), 0.82, accuracy: 0.0001)
        XCTAssertEqual(try XCTUnwrap(response.assessments.first?.speechDistanceMeters), 0.82, accuracy: 0.0001)
        XCTAssertEqual(response.assessments.first?.distance?.candidateP20Meters, 0.75)
        XCTAssertEqual(response.assessments.first?.distance?.depthSource, "raw_scene_depth")
        XCTAssertEqual(response.assessments.first?.distance?.quality, "good")
        XCTAssertEqual(response.assessments.first?.distance?.freshnessFrames, 0)
        XCTAssertEqual(response.alert?.alertFrameID, 8)
        XCTAssertEqual(response.overlayItems.count, 1)
    }

    @MainActor
    func testAreaFallbackCannotSupplySpokenMeters() throws {
        let json = #"{"session_id":"s","frame_id":4,"detections":[],"assessments":[{"track_key":"chair:1","label":"chair","confidence":0.8,"bbox":[0,0,50,50],"frame_width":100,"frame_height":100,"direction":"left","level":"medium","should_alert":true,"message":"","risk_source":"area_fallback","distance":{"meters":0.7,"usable":true,"status":"valid","is_stale":false,"sample_count":20,"coverage_ratio":0.5}}],"alert":{"alert_id":"a","alert_frame_id":4,"should_emit":true,"reason":"first_alert","message":"","track_key":"chair:1","label":"chair","direction":"left","level":"medium","distance_meters":0.7},"timing":null}"#.data(using: .utf8)!
        let response = try JSONDecoder().decode(FrameResponse.self, from: json)
        XCTAssertNil(response.assessments.first?.speechDistanceMeters)
        let spoken = SpeechCoordinator.spokenMessage(
            alert: try XCTUnwrap(response.alert),
            distanceMeters: response.assessments.first?.speechDistanceMeters
        )
        XCTAssertFalse(spoken.contains("米"))
    }

    func testFrameResponseParsesCoolingAlertWithoutAcknowledgementIdentifiers() throws {
        let json = #"{"session_id":"s","frame_id":10,"detections":[],"assessments":[],"alert":{"should_emit":false,"reason":"cooldown","message":"","track_key":null,"label":"person","direction":"center","level":"high"},"timing":null}"#.data(using: .utf8)!
        let response = try JSONDecoder().decode(FrameResponse.self, from: json)
        XCTAssertFalse(try XCTUnwrap(response.alert).shouldEmit)
        XCTAssertNil(response.alert?.alertID)
        XCTAssertNil(response.alert?.alertFrameID)
    }

    func testDistanceSpeechRounding() {
        XCTAssertEqual(DistanceFormatter.roundedForSpeech(0.84), 0.8)
        XCTAssertEqual(DistanceFormatter.roundedForSpeech(1.96), 2.0)
        XCTAssertEqual(DistanceFormatter.roundedForSpeech(2.26), 2.5)
        XCTAssertNil(DistanceFormatter.roundedForSpeech(5.1))
        XCTAssertEqual(DistanceFormatter.speechText(2.26), "约 2.5 米")
        XCTAssertEqual(DistanceFormatter.displayText(1.236), "1.2 m")
        XCTAssertEqual(
            DistanceFormatter.displayText(1.236, validationPrecision: true),
            "1.24 m"
        )
    }

    func testRGBBoxMapsToPortraitDepthCoordinates() throws {
        let mapped = try XCTUnwrap(
            CoordinateMapper.depthRect(
                forRGBBox: CGRect(x: 120, y: 160, width: 240, height: 320),
                rgbSize: CGSize(width: 480, height: 640),
                depthSize: CGSize(width: 192, height: 256)
            )
        )
        XCTAssertEqual(mapped.origin.x, 48, accuracy: 0.001)
        XCTAssertEqual(mapped.origin.y, 64, accuracy: 0.001)
        XCTAssertEqual(mapped.width, 96, accuracy: 0.001)
        XCTAssertEqual(mapped.height, 128, accuracy: 0.001)
    }

    func testAspectFitMappingAccountsForHorizontalLetterbox() throws {
        let mapped = try XCTUnwrap(
            CoordinateMapper.aspectFitRect(
                forRGBBox: CGRect(x: 0, y: 160, width: 480, height: 320),
                sourceSize: CGSize(width: 480, height: 640),
                viewportSize: CGSize(width: 300, height: 300)
            )
        )
        XCTAssertEqual(mapped.minX, 37.5, accuracy: 0.001)
        XCTAssertEqual(mapped.minY, 75, accuracy: 0.001)
        XCTAssertEqual(mapped.width, 225, accuracy: 0.001)
        XCTAssertEqual(mapped.height, 150, accuracy: 0.001)
    }

    func testFrameSequenceGateRejectsOldAndMismatchedResponses() {
        var gate = FrameSequenceGate()
        gate.markSent(10)
        XCTAssertFalse(gate.accept(responseFrameID: 9))
        XCTAssertTrue(gate.accept(responseFrameID: 10))
        gate.markSent(11)
        XCTAssertFalse(gate.accept(responseFrameID: 10))
        XCTAssertTrue(gate.accept(responseFrameID: 11))
    }

    func testLatestFrameSlotReplacesPendingHistory() {
        var slot = LatestFrameSlot<Int>()
        slot.replace(with: 10)
        slot.replace(with: 11)
        slot.replace(with: 12)
        XCTAssertEqual(slot.take(), 12)
        XCTAssertNil(slot.take())
    }

    func testReconnectBackoffIsBoundedAndNotFrameRateRetry() {
        XCTAssertEqual((1...6).map(ReconnectBackoff.delaySeconds), [1, 2, 4, 8, 8, 8])
    }

    func testHighRiskSpeechOnlyInterruptsLowerRiskSpeech() {
        XCTAssertTrue(
            SpeechInterruptionPolicy.shouldInterrupt(currentLevel: "low", incomingLevel: "high")
        )
        XCTAssertTrue(
            SpeechInterruptionPolicy.shouldInterrupt(currentLevel: "medium", incomingLevel: "high")
        )
        XCTAssertFalse(
            SpeechInterruptionPolicy.shouldInterrupt(currentLevel: "high", incomingLevel: "high")
        )
        XCTAssertFalse(
            SpeechInterruptionPolicy.shouldInterrupt(currentLevel: "low", incomingLevel: "medium")
        )
    }

    private func readBigEndianUInt32(_ data: Data, at offset: Int) -> UInt32 {
        data[offset..<(offset + 4)].reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
    }

    private func appendBigEndianUInt32(_ value: UInt32, to data: inout Data) {
        data.append(UInt8((value >> 24) & 0xFF))
        data.append(UInt8((value >> 16) & 0xFF))
        data.append(UInt8((value >> 8) & 0xFF))
        data.append(UInt8(value & 0xFF))
    }
}
