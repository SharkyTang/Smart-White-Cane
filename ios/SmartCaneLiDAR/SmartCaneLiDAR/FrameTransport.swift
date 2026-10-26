import Foundation

enum FrameTransportError: LocalizedError {
    case invalidServerURL
    case invalidResponse
    case httpStatus(Int, String)
    case mismatchedSession

    var errorDescription: String? {
        switch self {
        case .invalidServerURL:
            return "服务器地址必须是有效的 HTTPS 地址"
        case .invalidResponse:
            return "Mac 返回了无法识别的响应"
        case let .httpStatus(code, message):
            return "Mac 服务错误（\(code)）：\(message)"
        case .mismatchedSession:
            return "收到其他会话的响应，已忽略"
        }
    }
}

struct FrameTransportResult: Sendable {
    let response: FrameResponse
    let roundTripMilliseconds: Double
    let packetBytes: Int
    let requestFrameID: Int
    let captureTimestampSeconds: Double
    let depthSource: String
}

private struct SessionResetResponse: Decodable {
    let sessionID: String

    enum CodingKeys: String, CodingKey {
        case sessionID = "session_id"
    }
}

actor FrameTransport {
    static let frameContentType = "application/vnd.smart-cane.rgbd-frame;version=1"

    private let baseURL: URL
    private let session: URLSession

    init(baseURL: URL, session: URLSession? = nil) throws {
        guard baseURL.scheme?.lowercased() == "https", baseURL.host != nil else {
            throw FrameTransportError.invalidServerURL
        }
        self.baseURL = baseURL
        if let session {
            self.session = session
        } else {
            let configuration = URLSessionConfiguration.ephemeral
            configuration.waitsForConnectivity = true
            configuration.timeoutIntervalForRequest = 6
            configuration.timeoutIntervalForResource = 15
            self.session = URLSession(configuration: configuration)
        }
    }

    func reset() async throws -> String {
        let data = try await performJSONRequest(path: "api/session/reset", body: nil)
        let response = try JSONDecoder().decode(SessionResetResponse.self, from: data)
        guard !response.sessionID.isEmpty else { throw FrameTransportError.invalidResponse }
        return response.sessionID
    }

    func send(_ packet: SCF1Packet) async throws -> FrameTransportResult {
        let body = try packet.encoded()
        var request = URLRequest(url: endpoint("api/rgbd-frame"))
        request.httpMethod = "POST"
        request.timeoutInterval = 4.5
        request.setValue(Self.frameContentType, forHTTPHeaderField: "Content-Type")
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = body

        let started = ContinuousClock.now
        let (responseData, response) = try await session.data(for: request)
        let rtt = ContinuousClock.now - started
        try validateHTTP(response, data: responseData)
        let decoded = try JSONDecoder().decode(FrameResponse.self, from: responseData)
        guard decoded.sessionID == packet.metadata.sessionID else {
            throw FrameTransportError.mismatchedSession
        }
        return FrameTransportResult(
            response: decoded,
            roundTripMilliseconds: rtt.milliseconds,
            packetBytes: body.count,
            requestFrameID: packet.metadata.frameID,
            captureTimestampSeconds: packet.metadata.timestampSeconds,
            depthSource: packet.metadata.depthSource
        )
    }

    func acknowledge(sessionID: String, frameID: Int, alertID: String) async throws {
        let body = try JSONSerialization.data(withJSONObject: [
            "session_id": sessionID,
            "frame_id": frameID,
            "alert_id": alertID,
        ])
        _ = try await performJSONRequest(path: "api/alert/ack", body: body)
    }

    private func performJSONRequest(path: String, body: Data?) async throws -> Data {
        var request = URLRequest(url: endpoint(path))
        request.httpMethod = "POST"
        request.timeoutInterval = 4.5
        if body != nil {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.httpBody = body
        let (data, response) = try await session.data(for: request)
        try validateHTTP(response, data: data)
        return data
    }

    private func endpoint(_ path: String) -> URL {
        baseURL.appending(path: path)
    }

    private func validateHTTP(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else {
            throw FrameTransportError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            let message = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])
                .flatMap { $0["error"] as? [String: Any] }?["message"] as? String
            throw FrameTransportError.httpStatus(http.statusCode, message ?? "请求失败")
        }
    }
}

private extension Duration {
    var milliseconds: Double {
        let components = self.components
        return Double(components.seconds) * 1_000 + Double(components.attoseconds) / 1e15
    }
}
