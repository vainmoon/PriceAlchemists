//
//  ContentView.swift
//  PriceAlchemists
//
//  Created by Сергей Умнов on 04.06.2025.
//

import SwiftUI
import PhotosUI
import UIKit

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var navigationPath = NavigationPath()
    @State private var shouldResetToRoot = false
    
    func normalizeImage(_ image: UIImage) -> UIImage {
        // Check if the image needs to be normalized
        if image.imageOrientation == .up {
            return image
        }
        
        // Create a new CGContext with the correct orientation
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        image.draw(in: CGRect(origin: .zero, size: image.size))
        let normalizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        
        return normalizedImage ?? image
    }
    
    func processImage(_ imageData: Data) -> UIImage? {
        // Create image source to read EXIF data
        guard let imageSource = CGImageSourceCreateWithData(imageData as CFData, nil) else {
            return nil
        }
        
        // Read the image properties
        let properties = CGImageSourceCopyPropertiesAtIndex(imageSource, 0, nil) as? [String: Any]
        let orientation = properties?[kCGImagePropertyOrientation as String] as? Int
        
        // Create image options
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: 4096, // Max dimension
            kCGImageSourceShouldCacheImmediately: true
        ]
        
        // Create thumbnail with correct orientation
        if let scaledImage = CGImageSourceCreateThumbnailAtIndex(imageSource, 0, options as CFDictionary) {
            let image = UIImage(cgImage: scaledImage)
            return normalizeImage(image)
        }
        
        // If thumbnail creation fails, try to load the original image
        if let image = UIImage(data: imageData) {
            return normalizeImage(image)
        }
        
        return nil
    }
    
    var body: some View {
        NavigationStack(path: $navigationPath) {
            VStack {
                Spacer()
                
                if let image = selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .frame(maxWidth: .infinity, maxHeight: UIScreen.main.bounds.height * 0.6)
                        .cornerRadius(15)
                        .shadow(radius: 8)
                        .padding()
                } else {
                    ContentUnavailableView(
                        "Изображение не выбрано",
                        systemImage: "photo",
                        description: Text("Нажмите кнопку ниже, чтобы выбрать изображение")
                    )
                    .frame(maxHeight: UIScreen.main.bounds.height * 0.6)
                }
                
                Spacer()
                
                VStack(spacing: 16) {
                    if selectedImage != nil {
                        Button(action: {
                            navigationPath.append("segmentation")
                        }) {
                            Text("Узнать стоимость")
                                .font(.headline)
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .cornerRadius(10)
                        }
                    }
                    
                    PhotosPicker(selection: $selectedItem,
                               matching: .images) {
                        Text("Выбрать изображение")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.green)
                            .cornerRadius(10)
                    }
                }
                .padding(.horizontal)
                .padding(.bottom)
            }
            .navigationTitle("Оценка стоимости")
            .onChange(of: selectedItem) { oldItem, newItem in
                Task {
                    if let data = try? await newItem?.loadTransferable(type: Data.self) {
                        // Process image on background thread
                        let processedImage = await Task.detached(priority: .userInitiated) {
                            return processImage(data)
                        }.value
                        
                        // Update UI on main thread
                        await MainActor.run {
                            selectedImage = processedImage
                        }
                    }
                }
            }
            .onChange(of: shouldResetToRoot) { _, newValue in
                if newValue {
                    // Reset everything
                    selectedItem = nil
                    selectedImage = nil
                    navigationPath.removeLast(navigationPath.count)
                    shouldResetToRoot = false
                }
            }
            .navigationDestination(for: String.self) { route in
                switch route {
                case "segmentation":
                    if let image = selectedImage {
                        ImageSegmentationView(
                            image: image,
                            navigationPath: $navigationPath,
                            shouldResetToRoot: $shouldResetToRoot,
                            baseURL: "http://62.84.127.68:8050"
                        )
                        .navigationBarTitleDisplayMode(.inline)
                    }
                default:
                    EmptyView()
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
