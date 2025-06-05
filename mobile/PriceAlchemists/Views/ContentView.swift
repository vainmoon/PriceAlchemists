//
//  ContentView.swift
//  PriceAlchemists
//
//  Created by Сергей Умнов on 04.06.2025.
//

import SwiftUI
import PhotosUI

struct ContentView: View {
    @State private var selectedItem: PhotosPickerItem?
    @State private var selectedImage: UIImage?
    @State private var navigationPath = NavigationPath()
    @State private var shouldResetToRoot = false
    
    var body: some View {
        NavigationStack(path: $navigationPath) {
            VStack {
                if let image = selectedImage {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(height: 300)
                        .cornerRadius(12)
                        .shadow(radius: 5)
                        .padding()
                    
                    Button(action: {
                        navigationPath.append("segmentation")
                    }) {
                        Text("Start Segmentation")
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(Color.blue)
                            .cornerRadius(10)
                    }
                    .padding(.horizontal)
                    .disabled(selectedImage == nil)
                } else {
                    ContentUnavailableView(
                        "No Image Selected",
                        systemImage: "photo",
                        description: Text("Tap the button below to select an image")
                    )
                }
                
                Spacer()
                
                PhotosPicker(selection: $selectedItem,
                           matching: .images) {
                    Text("Select Image")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .cornerRadius(10)
                }
                .padding()
            }
            .navigationTitle("Image Segmentation")
            .onChange(of: selectedItem) { oldItem, newItem in
                Task {
                    if let data = try? await newItem?.loadTransferable(type: Data.self),
                       let image = UIImage(data: data) {
                        selectedImage = image
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
                            baseURL: "http://127.0.0.1:8000"
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
