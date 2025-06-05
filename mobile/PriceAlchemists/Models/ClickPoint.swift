import Foundation
import SwiftUI

enum SegmentationTool {
    case point
    case rectangle
}

enum SegmentationType: String, Codable {
    case point
    case rectangle
}

struct Point: Codable {
    let x: Int
    let y: Int
    
    init(cgPoint: CGPoint) {
        self.x = Int(round(cgPoint.x))
        self.y = Int(round(cgPoint.y))
    }
}

struct SegmentationClick: Codable {
    let type: SegmentationType
    let points: [Point]
    
    static func point(_ point: CGPoint) -> SegmentationClick {
        SegmentationClick(
            type: .point,
            points: [Point(cgPoint: point)]
        )
    }
    
    static func rectangle(start: CGPoint, end: CGPoint) -> SegmentationClick {
        SegmentationClick(
            type: .rectangle,
            points: [Point(cgPoint: start), Point(cgPoint: end)]
        )
    }
} 