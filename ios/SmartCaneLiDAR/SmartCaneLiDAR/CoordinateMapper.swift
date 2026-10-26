import CoreGraphics
import Foundation

enum CoordinateMapper {
    static func depthRect(
        forRGBBox box: CGRect,
        rgbSize: CGSize,
        depthSize: CGSize
    ) -> CGRect? {
        guard rgbSize.width > 0, rgbSize.height > 0,
              depthSize.width > 0, depthSize.height > 0 else { return nil }
        let normalized = CGRect(
            x: box.minX / rgbSize.width,
            y: box.minY / rgbSize.height,
            width: box.width / rgbSize.width,
            height: box.height / rgbSize.height
        ).standardized
        let clipped = normalized.intersection(CGRect(x: 0, y: 0, width: 1, height: 1))
        guard !clipped.isNull, !clipped.isEmpty else { return nil }
        return CGRect(
            x: clipped.minX * depthSize.width,
            y: clipped.minY * depthSize.height,
            width: clipped.width * depthSize.width,
            height: clipped.height * depthSize.height
        )
    }

    static func aspectFitRect(
        forRGBBox box: CGRect,
        sourceSize: CGSize,
        viewportSize: CGSize
    ) -> CGRect? {
        guard sourceSize.width > 0, sourceSize.height > 0,
              viewportSize.width > 0, viewportSize.height > 0 else { return nil }
        let scale = min(viewportSize.width / sourceSize.width, viewportSize.height / sourceSize.height)
        let renderedSize = CGSize(width: sourceSize.width * scale, height: sourceSize.height * scale)
        let offset = CGPoint(
            x: (viewportSize.width - renderedSize.width) / 2,
            y: (viewportSize.height - renderedSize.height) / 2
        )
        return CGRect(
            x: box.minX * scale + offset.x,
            y: box.minY * scale + offset.y,
            width: box.width * scale,
            height: box.height * scale
        )
    }
}

struct FrameSequenceGate: Sendable {
    private(set) var inFlightFrameID: Int?
    private(set) var latestAcceptedFrameID: Int = -1

    mutating func markSent(_ frameID: Int) {
        precondition(frameID >= 0)
        inFlightFrameID = frameID
    }

    mutating func accept(responseFrameID: Int) -> Bool {
        guard responseFrameID >= latestAcceptedFrameID,
              responseFrameID == inFlightFrameID else { return false }
        latestAcceptedFrameID = responseFrameID
        inFlightFrameID = nil
        return true
    }

    mutating func clearInFlight(_ frameID: Int) {
        if inFlightFrameID == frameID { inFlightFrameID = nil }
    }

    mutating func reset() {
        inFlightFrameID = nil
        latestAcceptedFrameID = -1
    }
}
