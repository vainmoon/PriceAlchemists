import Foundation
import UIKit

struct PredictionResponse: Codable {
    let price: Double
    let similarProducts: [String]  // Base64 encoded image strings
}

enum SegmentationError: Error {
    case invalidImage
    case networkError(Error)
    case invalidResponse(String)
    case decodingError
    case encodingError
    case serverError(String)
}

extension SegmentationError: LocalizedError {
    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Не удалось обработать изображение"
        case .networkError(let error):
            return "Ошибка сети: \(error.localizedDescription)"
        case .invalidResponse(let message):
            return "Некорректный ответ: \(message)"
        case .decodingError:
            return "Не удалось декодировать ответ"
        case .encodingError:
            return "Не удалось закодировать запрос"
        case .serverError(let message):
            return "Ошибка сервера: \(message)"
        }
    }
}

class SegmentationService {
    private let baseURL: URL
    private let session: URLSession
    
    init(baseURL: String, session: URLSession = .shared) {
        self.baseURL = URL(string: baseURL)!
        self.session = session
    }
    
    func requestSegmentation(image: UIImage, clicks: [SegmentationClick]) async throws -> UIImage {
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
            throw SegmentationError.invalidImage
        }
        
        var request = URLRequest(url: baseURL.appendingPathComponent("segment"))
        request.httpMethod = "POST"
        
        // Create multipart form data
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add image data
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"file.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n".data(using: .utf8)!)
        
        // Convert clicks to JSON string
        let clicksData = try JSONEncoder().encode(clicks)
        guard let clicksString = String(data: clicksData, encoding: .utf8) else {
            throw SegmentationError.encodingError
        }
        
        print("Sending request to:", request.url?.absoluteString ?? "unknown URL")
        print("Sending clicks data:", clicksString)
        print("Image data size:", imageData.count)
        
        // Add clicks as form field
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"prompts\"\r\n\r\n".data(using: .utf8)!)
        body.append(clicksString.data(using: .utf8)!)
        body.append("\r\n".data(using: .utf8)!)
        
        // Add final boundary
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        do {
            let (data, response) = try await session.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw SegmentationError.invalidResponse("Invalid HTTP response")
            }
            
            print("Response Headers:", httpResponse.allHeaderFields)
            print("Content-Type:", httpResponse.allHeaderFields["Content-Type"] ?? "none")
            print("Response Size:", data.count)
            
            // Print raw response data as string
            if let responseString = String(data: data, encoding: .utf8) {
                print("Response Content:", responseString)
            }
            
            // If we got JSON, try to parse it as an error message
            if httpResponse.value(forHTTPHeaderField: "Content-Type")?.contains("application/json") == true {
                if let jsonString = String(data: data, encoding: .utf8) {
                    print("Received JSON response:", jsonString)
                    if let json = try? JSONSerialization.jsonObject(with: data) as? [String: String],
                       let errorMessage = json["error"] {
                        throw SegmentationError.serverError(errorMessage)
                    }
                }
                throw SegmentationError.invalidResponse("Received JSON response instead of image")
            }
            
            if !(200...299).contains(httpResponse.statusCode) {
                if let errorMessage = String(data: data, encoding: .utf8) {
                    throw SegmentationError.networkError(NSError(domain: "", code: httpResponse.statusCode, userInfo: [NSLocalizedDescriptionKey: errorMessage]))
                }
                throw SegmentationError.networkError(NSError(domain: "", code: httpResponse.statusCode))
            }
            
            guard let mask = UIImage(data: data) else {
                let dataPreview = String(data: data.prefix(100), encoding: .utf8) ?? "non-text data"
                throw SegmentationError.invalidResponse("Could not create image from response data. First 100 bytes: \(dataPreview)")
            }
            
            return mask
        } catch {
            if let segError = error as? SegmentationError {
                throw segError
            }
            throw SegmentationError.networkError(error)
        }
    }
    
    func requestPrediction(image: UIImage, mask: UIImage?) async throws -> (Double, [UIImage]) {
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
            throw SegmentationError.invalidImage
        }
        
        guard let maskData = mask?.jpegData(compressionQuality: 0.8) else {
            // If no mask, use a blank image
            let renderer = UIGraphicsImageRenderer(size: image.size)
            let blankMask = renderer.image { context in
                UIColor.black.setFill()
                context.fill(CGRect(origin: .zero, size: image.size))
            }
            guard let blankData = blankMask.jpegData(compressionQuality: 0.8) else {
                throw SegmentationError.invalidImage
            }
            return try await sendPredictionRequest(imageData: imageData, maskData: blankData)
        }
        
        return try await sendPredictionRequest(imageData: imageData, maskData: maskData)
    }
    
    private func sendPredictionRequest(imageData: Data, maskData: Data) async throws -> (Double, [UIImage]) {
        var request = URLRequest(url: baseURL.appendingPathComponent("predict"))
        request.httpMethod = "POST"
        
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add image data
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"file.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n".data(using: .utf8)!)
        
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"mask\"; filename=\"mask.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
        body.append(maskData)
        body.append("\r\n".data(using: .utf8)!)
        
        // Add final boundary
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body

        print("Sending prediction request to:", request.url?.absoluteString ?? "unknown URL")
        print("Image data size:", imageData.count)
        print("Mask data size:", maskData.count)
        
        let (data, response) = try await session.data(for: request)
        
        guard let httpResponse = response as? HTTPURLResponse else {
            throw SegmentationError.invalidResponse("Invalid HTTP response")
        }
        
        print("Prediction Response Headers:", httpResponse.allHeaderFields)
        print("Prediction Response Status Code:", httpResponse.statusCode)
        print("Prediction Response Size:", data.count)
        
        // Print raw response data as string if it's small enough
        if data.count < 1000, let responseString = String(data: data, encoding: .utf8) {
            print("Prediction Response Content:", responseString)
        }
        
        let predictionResponse = try JSONDecoder().decode(PredictionResponse.self, from: data)
        print("Predicted price:", predictionResponse.price)
        
        // Convert base64 strings to UIImages
        let similarProducts = try predictionResponse.similarProducts.map { base64String in
            guard let imageData = Data(base64Encoded: base64String),
                  let image = UIImage(data: imageData) else {
                throw SegmentationError.invalidResponse("Could not decode similar product image")
            }
            return image
        }
        
        return (predictionResponse.price, similarProducts)
    }
} 
