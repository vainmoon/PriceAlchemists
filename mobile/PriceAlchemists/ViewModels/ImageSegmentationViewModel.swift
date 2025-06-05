import SwiftUI
import Combine

@MainActor
class ImageSegmentationViewModel: ObservableObject {
    @Published var selectedImage: UIImage?
    @Published var segmentationMask: UIImage?
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
        error = nil
        isDragging = false
        dragStartPoint = nil
        currentDragPoint = nil
    }
    
    private func requestSegmentation() async {
        guard let image = selectedImage else { return }
        
        isProcessing = true
        error = nil
        
        do {
            let mask = try await segmentationService.requestSegmentation(
                image: image,
                clicks: clicks
            )
            
            // Apply the mask to darken the original image
            if let darkened = applyMask(mask, to: image) {
                segmentationMask = darkened
            }
        } catch {
            self.error = error.localizedDescription
        }
        
        isProcessing = false
    }
    
    private func applyMask(_ mask: UIImage, to originalImage: UIImage) -> UIImage? {
        guard let maskCG = mask.cgImage,
              let originalCG = originalImage.cgImage,
              let colorSpace = originalCG.colorSpace else { return nil }

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
        
        
        // 1. Рисуем оригинальное изображение
        context.draw(originalCG, in: rect)
        
        // 2. Создаём прозрачную чёрную маску
        context.saveGState()
        context.clip(to: rect, mask: maskCG)  // используем maskCG как альфа-маску

        // Цвет — полупрозрачный чёрный (например, alpha = 0.2)
        context.setFillColor(UIColor(white: 0, alpha: 0.85).cgColor)
        context.fill(rect)
        context.restoreGState()

        // 3. Извлекаем финальное изображение
        guard let resultCG = context.makeImage() else { return nil }
        return UIImage(cgImage: resultCG)
    }
    
    func requestPrediction() async throws -> Double {
        guard let image = selectedImage else {
            throw SegmentationError.invalidImage
        }
        
        return try await segmentationService.requestPrediction(
            image: image,
            mask: segmentationMask
        )
    }
} 
