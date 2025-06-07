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
                    
                    if selectedImage != nil {
                        Button(action: {
                            navigationPath.append("segmentation")
                        }) {
                            Text("Начать сегментацию")
                                .font(.headline)
                                .foregroundColor(.white)
                                .frame(maxWidth: .infinity)
                                .padding()
                                .background(Color.blue)
                                .cornerRadius(10)
                        }
                    }
                }
                .padding(.horizontal)
                .padding(.bottom)
            }
            .navigationTitle("Оценка стоимости")
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
