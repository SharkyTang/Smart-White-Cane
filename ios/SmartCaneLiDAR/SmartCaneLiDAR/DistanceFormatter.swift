import Foundation

enum DistanceFormatter {
    static func roundedForSpeech(_ meters: Double) -> Double? {
        guard meters.isFinite, meters > 0, meters <= 5 else { return nil }
        let increment = meters < 2 ? 0.1 : 0.5
        return (meters / increment).rounded() * increment
    }

    static func speechText(_ meters: Double?) -> String? {
        guard let meters, let rounded = roundedForSpeech(meters) else { return nil }
        if rounded < 2 {
            return String(format: "约 %.1f 米", rounded)
        }
        return rounded.rounded() == rounded
            ? String(format: "约 %.0f 米", rounded)
            : String(format: "约 %.1f 米", rounded)
    }

    static func displayText(_ meters: Double?, validationPrecision: Bool = false) -> String {
        guard let meters, meters.isFinite, meters > 0, meters <= 5 else { return "距离不可用" }
        return String(format: validationPrecision ? "%.2f m" : "%.1f m", meters)
    }
}
