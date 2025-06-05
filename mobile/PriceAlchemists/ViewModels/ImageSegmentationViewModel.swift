import SwiftUI
import Combine

@MainActor
class ImageSegmentationViewModel: ObservableObject {
    @Published var selectedImage: UIImage?
    @Published var segmentationMask: UIImage?  // Visual representation
    @Published var rawSegmentationMask: UIImage?  // Raw mask for sending to server
    @Published var clicks: [SegmentationClick] = []
    @Published var isProcessing = false
    @Published var error: String?
    @Published var selectedTool: SegmentationTool = .point
    @Published var isDragging = false
    @Published var dragStartPoint: CGPoint?
    @Published var currentDragPoint: CGPoint?
    
    private let segmentationService: SegmentationService
    
    init(baseURL: String = "YOUR_API_ENDPOINT") {
        self.segmentationService = SegmentationService(baseURL: baseURL)
    }
    
    private func normalizeImage(_ image: UIImage) -> UIImage {
        // Ensure correct orientation
        let normalizedImage = image.normalizedImage()
        
        // Scale down very large images to prevent memory issues
        let maxDimension: CGFloat = 2048 // Max dimension we'll allow
        let size = normalizedImage.size
        
        if size.width > maxDimension || size.height > maxDimension {
            let scale = maxDimension / max(size.width, size.height)
            let newSize = CGSize(width: size.width * scale, height: size.height * scale)
            
            let renderer = UIGraphicsImageRenderer(size: newSize)
            return renderer.image { context in
                normalizedImage.draw(in: CGRect(origin: .zero, size: newSize))
            }
        }
        
        return normalizedImage
    }
    
    func processImageTap(at point: CGPoint) {
        switch selectedTool {
        case .point:
            clicks.append(.point(point))
            Task {
                await requestSegmentation()
            }
        case .rectangle:
            // Handle in drag gesture
            break
        }
    }
    
    func startDragging(at point: CGPoint) {
        if selectedTool == .rectangle {
            isDragging = true
            dragStartPoint = point
            currentDragPoint = point
        }
    }
    
    func updateDragging(at point: CGPoint) {
        if isDragging {
            currentDragPoint = point
        }
    }
    
    func endDragging(at point: CGPoint) {
        if isDragging, let startPoint = dragStartPoint {
            isDragging = false
            currentDragPoint = nil
            
            // Add rectangle as a single click
            clicks.append(.rectangle(start: startPoint, end: point))
            
            dragStartPoint = nil
            
            // Automatically switch back to point tool
            selectedTool = .point
            
            Task {
                await requestSegmentation()
            }
        }
    }
    
    func resetSegmentation() {
        clicks.removeAll()
        segmentationMask = nil
        rawSegmentationMask = nil
        error = nil
        isDragging = false
        dragStartPoint = nil
        currentDragPoint = nil
    }
    
    private func requestSegmentation() async {
        guard let image = selectedImage else { return }
        
        // Normalize the image before sending
        let normalizedImage = normalizeImage(image)
        
        isProcessing = true
        error = nil
        
        do {
            let mask = try await segmentationService.requestSegmentation(
                image: normalizedImage,
                clicks: clicks
            )
            
            if let darkened = applyMask(mask, to: normalizedImage) {
                segmentationMask = darkened
            }
        } catch {
            self.error = error.localizedDescription
        }
        
        isProcessing = false
    }
    
    private func applyMask(_ mask: UIImage, to originalImage: UIImage) -> UIImage? {
        // Normalize the mask to match original image orientation and size
        let normalizedMask = normalizeImage(mask)
        
        guard let maskCG = normalizedMask.cgImage,
              let originalCG = originalImage.cgImage,
              let colorSpace = originalCG.colorSpace else { return nil }

        // Store the raw mask for sending to server
        self.rawSegmentationMask = normalizedMask

        let width = originalCG.width
        let height = originalCG.height
        let bitsPerComponent = originalCG.bitsPerComponent
        let bytesPerRow = originalCG.bytesPerRow
        let bitmapInfo = originalCG.bitmapInfo

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo.rawValue
        ) else { return nil }

        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        
        // Draw original image
        context.draw(originalCG, in: rect)
        
        // Create darkening effect with inverted mask
        context.saveGState()
        context.clip(to: rect, mask: maskCG)
        context.setFillColor(UIColor(white: 0, alpha: 0.85).cgColor)
        context.fill(rect)
        context.restoreGState()

        guard let resultCG = context.makeImage() else { return nil }
        return UIImage(cgImage: resultCG)
    }
    
    func requestPrediction() async throws -> Double {
        guard let image = selectedImage else {
            throw SegmentationError.invalidImage
        }
        
        return try await segmentationService.requestPrediction(
            image: image,
            mask: rawSegmentationMask  // Use the raw mask instead of the visual one
        )
    }
}

// Extension to handle image orientation
extension UIImage {
    func normalizedImage() -> UIImage {
        if imageOrientation == .up {
            return self
        }
        
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        draw(in: CGRect(origin: .zero, size: size))
        let normalizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return normalizedImage ?? self
    }
} 
