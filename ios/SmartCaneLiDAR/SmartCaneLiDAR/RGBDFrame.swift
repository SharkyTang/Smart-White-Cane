import CoreGraphics
import Foundation

struct RGBDFrame: @unchecked Sendable {
    let sessionID: String
    let frameID: Int
    let timestampSeconds: Double
    let jpegData: Data
    let depthData: Data
    let confidenceData: Data
    let rgbWidth: Int
    let rgbHeight: Int
    let depthWidth: Int
    let depthHeight: Int
    let depthSource: String
    let previewImage: CGImage

    var metadata: SCF1Metadata {
        SCF1Metadata(
            sessionID: sessionID,
            frameID: frameID,
            rgbWidth: rgbWidth,
            rgbHeight: rgbHeight,
            depthWidth: depthWidth,
            depthHeight: depthHeight,
            timestampSeconds: timestampSeconds,
            source: "arkit_scene_depth",
            depthSource: depthSource,
            orientation: "portrait_up"
        )
    }
}
