import Foundation

enum SCF1PacketError: Error, Equatable {
    case invalidMagic
    case truncatedHeader
    case invalidLength
    case invalidMetadata
    case payloadTooLarge
}

struct SCF1Metadata: Codable, Equatable, Sendable {
    let schemaVersion: Int
    let sessionID: String
    let frameID: Int
    let rgbWidth: Int
    let rgbHeight: Int
    let depthWidth: Int
    let depthHeight: Int
    let timestampSeconds: Double
    let source: String
    let depthSource: String
    let orientation: String
    var scenarioID: String?

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version"
        case sessionID = "session_id"
        case frameID = "frame_id"
        case rgbWidth = "rgb_width"
        case rgbHeight = "rgb_height"
        case depthWidth = "depth_width"
        case depthHeight = "depth_height"
        case timestampSeconds = "timestamp_seconds"
        case source
        case depthSource = "depth_source"
        case orientation
        case scenarioID = "scenario_id"
    }

    init(
        sessionID: String,
        frameID: Int,
        rgbWidth: Int,
        rgbHeight: Int,
        depthWidth: Int,
        depthHeight: Int,
        timestampSeconds: Double = 0,
        source: String = "arkit_scene_depth",
        depthSource: String = "raw_scene_depth",
        orientation: String = "portrait_up",
        scenarioID: String? = nil
    ) {
        self.schemaVersion = 1
        self.sessionID = sessionID
        self.frameID = frameID
        self.rgbWidth = rgbWidth
        self.rgbHeight = rgbHeight
        self.depthWidth = depthWidth
        self.depthHeight = depthHeight
        self.timestampSeconds = timestampSeconds
        self.source = source
        self.depthSource = depthSource
        self.orientation = orientation
        self.scenarioID = scenarioID
    }

    func validated() throws -> SCF1Metadata {
        guard schemaVersion == 1,
              !sessionID.isEmpty,
              frameID >= 0,
              rgbWidth > 0,
              rgbHeight > 0,
              depthWidth > 0,
              depthHeight > 0,
              timestampSeconds.isFinite,
              timestampSeconds >= 0,
              source == "arkit_scene_depth",
              ["raw_scene_depth", "smoothed_scene_depth"].contains(depthSource),
              orientation == "portrait_up" else {
            throw SCF1PacketError.invalidMetadata
        }
        return self
    }

    func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(schemaVersion, forKey: .schemaVersion)
        try values.encode(sessionID, forKey: .sessionID)
        try values.encode(frameID, forKey: .frameID)
        try values.encode(rgbWidth, forKey: .rgbWidth)
        try values.encode(rgbHeight, forKey: .rgbHeight)
        try values.encode(depthWidth, forKey: .depthWidth)
        try values.encode(depthHeight, forKey: .depthHeight)
        try values.encode(timestampSeconds, forKey: .timestampSeconds)
        try values.encode(source, forKey: .source)
        try values.encode(depthSource, forKey: .depthSource)
        try values.encode(orientation, forKey: .orientation)
        if let scenarioID {
            try values.encode(scenarioID, forKey: .scenarioID)
        } else {
            try values.encodeNil(forKey: .scenarioID)
        }
    }
}

struct SCF1Packet: Equatable, Sendable {
    static let magic = Data("SCF1".utf8)
    static let headerLength = 20
    static let maximumComponentLength = 4_000_000
    static let maximumMetadataLength = 64 * 1_024
    static let maximumPacketLength = 2_000_000

    let metadata: SCF1Metadata
    let jpegData: Data
    let depthData: Data
    let confidenceData: Data

    func encoded() throws -> Data {
        _ = try metadata.validated()
        guard depthData.count == metadata.depthWidth * metadata.depthHeight * 2,
              confidenceData.count == metadata.depthWidth * metadata.depthHeight else {
            throw SCF1PacketError.invalidLength
        }

        let metadataData = try JSONEncoder.scf1.encode(metadata)
        let lengths = [metadataData.count, jpegData.count, depthData.count, confidenceData.count]
        guard metadataData.count <= Self.maximumMetadataLength,
              lengths.allSatisfy({ $0 > 0 && $0 <= Self.maximumComponentLength }),
              Self.headerLength + lengths.reduce(0, +) <= Self.maximumPacketLength else {
            throw SCF1PacketError.payloadTooLarge
        }

        var output = Data(capacity: Self.headerLength + lengths.reduce(0, +))
        output.append(Self.magic)
        lengths.forEach { output.appendUInt32BigEndian(UInt32($0)) }
        output.append(metadataData)
        output.append(jpegData)
        output.append(depthData)
        output.append(confidenceData)
        return output
    }

    static func decode(_ data: Data) throws -> SCF1Packet {
        guard data.count >= headerLength else { throw SCF1PacketError.truncatedHeader }
        guard data.count <= maximumPacketLength else { throw SCF1PacketError.payloadTooLarge }
        guard data.prefix(4) == magic else { throw SCF1PacketError.invalidMagic }

        let lengths = try stride(from: 4, to: headerLength, by: 4).map {
            Int(try data.uint32BigEndian(at: $0))
        }
        guard lengths[0] <= maximumMetadataLength,
              lengths.allSatisfy({ $0 > 0 && $0 <= maximumComponentLength }) else {
            throw SCF1PacketError.payloadTooLarge
        }
        let payloadLength = lengths.reduce(0, +)
        guard payloadLength == data.count - headerLength else {
            throw SCF1PacketError.invalidLength
        }

        var cursor = headerLength
        func read(_ length: Int) -> Data {
            defer { cursor += length }
            return data.subdata(in: cursor..<(cursor + length))
        }
        let metadataData = read(lengths[0])
        let jpegData = read(lengths[1])
        let depthData = read(lengths[2])
        let confidenceData = read(lengths[3])
        guard let metadata = try? JSONDecoder.scf1.decode(SCF1Metadata.self, from: metadataData),
              (try? metadata.validated()) != nil else {
            throw SCF1PacketError.invalidMetadata
        }
        guard depthData.count == metadata.depthWidth * metadata.depthHeight * 2,
              confidenceData.count == metadata.depthWidth * metadata.depthHeight else {
            throw SCF1PacketError.invalidLength
        }
        return SCF1Packet(
            metadata: metadata,
            jpegData: jpegData,
            depthData: depthData,
            confidenceData: confidenceData
        )
    }
}

private extension JSONEncoder {
    static var scf1: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return encoder
    }
}

private extension JSONDecoder {
    static let scf1 = JSONDecoder()
}

private extension Data {
    mutating func appendUInt32BigEndian(_ value: UInt32) {
        var encoded = value.bigEndian
        Swift.withUnsafeBytes(of: &encoded) { append(contentsOf: $0) }
    }

    func uint32BigEndian(at offset: Int) throws -> UInt32 {
        guard offset >= 0, offset + 4 <= count else { throw SCF1PacketError.truncatedHeader }
        return self[offset..<(offset + 4)].reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
    }
}
