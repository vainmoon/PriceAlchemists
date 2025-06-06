import SwiftUI

struct ImageSegmentationView: View {
    // Add color constant to match the start segmentation button
    private let accentColor = Color.blue
    private let rectangleFillOpacity = 0.2
    
    @StateObject private var viewModel: ImageSegmentationViewModel
    @State private var showingPredictionResult = false
    @State private var predictedPrice: Double?
    @State private var similarProducts: [UIImage] = []
    @State private var isPredicting = false
    @Binding var navigationPath: NavigationPath
    @Binding var shouldResetToRoot: Bool
    let image: UIImage
    
    init(image: UIImage, navigationPath: Binding<NavigationPath>, shouldResetToRoot: Binding<Bool>, baseURL: String = "YOUR_API_ENDPOINT") {
        _viewModel = StateObject(wrappedValue: ImageSegmentationViewModel(baseURL: baseURL))
        self.image = image
        _navigationPath = navigationPath
        _shouldResetToRoot = shouldResetToRoot
    }
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                Color.black.opacity(0.0) // Invisible background to ensure proper layout
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                
                // Create a container with the same aspect ratio as the image
                let imageAspect = image.size.width / image.size.height
                let containerSize = calculateContainerSize(for: geometry.size, withAspectRatio: imageAspect)
                
                ZStack {
                    // Base image
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                    
                    // Segmentation mask overlay
                    if let mask = viewModel.segmentationMask {
                        Image(uiImage: mask)
                            .resizable()
                            .aspectRatio(contentMode: .fit)
                            .allowsHitTesting(false)
                    }
                    
                    // Rectangle preview while dragging
                    if viewModel.isDragging,
                       let start = viewModel.dragStartPoint,
                       let current = viewModel.currentDragPoint {
                        let startPoint = convertToViewCoordinates(
                            CGPoint(x: start.x, y: start.y),
                            imageSize: image.size,
                            viewSize: containerSize
                        )
                        let currentPoint = convertToViewCoordinates(
                            CGPoint(x: current.x, y: current.y),
                            imageSize: image.size,
                            viewSize: containerSize
                        )
                        
                        Path { path in
                            path.move(to: startPoint)
                            path.addLine(to: CGPoint(x: currentPoint.x, y: startPoint.y))
                            path.addLine(to: currentPoint)
                            path.addLine(to: CGPoint(x: startPoint.x, y: currentPoint.y))
                            path.closeSubpath()
                        }
                        .fill(accentColor.opacity(rectangleFillOpacity))
                        
                        Path { path in
                            path.move(to: startPoint)
                            path.addLine(to: CGPoint(x: currentPoint.x, y: startPoint.y))
                            path.addLine(to: currentPoint)
                            path.addLine(to: CGPoint(x: startPoint.x, y: currentPoint.y))
                            path.closeSubpath()
                        }
                        .stroke(accentColor, lineWidth: 2)
                    }
                    
                    // Click points overlay
                    ForEach(viewModel.clicks.indices, id: \.self) { index in
                        let click = viewModel.clicks[index]
                        
                        if click.type == .point, let point = click.points.first {
                            let scaledPoint = convertToViewCoordinates(
                                CGPoint(x: point.x, y: point.y),
                                imageSize: image.size,
                                viewSize: containerSize
                            )
                            
                            Circle()
                                .fill(accentColor)
                                .frame(width: 10, height: 10)
                                .position(scaledPoint)
                        }
                    }
                }
                .frame(width: containerSize.width, height: containerSize.height)
                .clipped() // Ensure mask doesn't extend beyond image bounds
                .position(x: geometry.size.width / 2, y: geometry.size.height / 2)
                .gesture(
                    viewModel.selectedTool == .rectangle ?
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            let location = value.location
                            if value.translation == .zero {
                                handleTap(at: location, in: geometry, isDragStart: true)
                            } else {
                                handleTap(at: location, in: geometry, isDragging: true)
                            }
                        }
                        .onEnded { value in
                            handleTap(at: value.location, in: geometry, isDragEnd: true)
                        }
                    : nil
                )
                .onTapGesture { location in
                    if viewModel.selectedTool == .point {
                        handleTap(at: location, in: geometry)
                    }
                }
                
                // Loading indicator
                if viewModel.isProcessing {
                    ZStack {
                        Color.black.opacity(0.3)
                        ProgressView()
                            .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            .scaleEffect(2.0)
                    }
                }
                
                // Loading indicator for prediction
                if isPredicting {
                    ZStack {
                        Color.black.opacity(0.3)
                        VStack {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                .scaleEffect(2.0)
                            Text("Расчет стоимости...")
                                .foregroundColor(.white)
                                .padding(.top)
                        }
                    }
                }
                
                // Error overlay
                if let error = viewModel.error {
                    VStack {
                        Text("Ошибка")
                            .font(.headline)
                            .foregroundColor(.white)
                        Text(error)
                            .font(.subheadline)
                            .foregroundColor(.white)
                            .multilineTextAlignment(.center)
                            .padding()
                    }
                    .frame(maxWidth: .infinity, maxHeight: 100)
                    .background(Color.red.opacity(0.8))
                    .cornerRadius(10)
                    .padding()
                    .transition(.move(edge: .top))
                }
            }
        }
        .onAppear {
            viewModel.selectedImage = image
        }
        .navigationBarItems(
            trailing: HStack {
                // Tool toggle button
                Button(action: {
                    viewModel.selectedTool = viewModel.selectedTool == .point ? .rectangle : .point
                }) {
                    Image(systemName: viewModel.selectedTool == .point ? "dot.circle.fill" : "rectangle")
                        .imageScale(.large)
                        .foregroundColor(.blue)
                }
                .padding(.trailing)
                
                // Done button
                Button(action: {
                    Task {
                        await predictPrice()
                    }
                }) {
                    Image(systemName: "dollarsign.circle.fill")
                        .imageScale(.large)
                        .foregroundColor(.blue)
                }
                .padding(.trailing)
                
                Button(action: {
                    viewModel.resetSegmentation()
                }) {
                    Image(systemName: "arrow.counterclockwise")
                        .imageScale(.large)
                }
            }
        )
        .navigationDestination(isPresented: $showingPredictionResult) {
            if let price = predictedPrice {
                PredictionResultView(
                    price: price,
                    similarProducts: similarProducts,
                    navigationPath: $navigationPath,
                    shouldResetToRoot: $shouldResetToRoot
                )
            }
        }
    }
    
    private func calculateContainerSize(for viewSize: CGSize, withAspectRatio aspect: CGFloat) -> CGSize {
        let maxWidth = viewSize.width
        let maxHeight = viewSize.height
        
        if maxWidth / aspect <= maxHeight {
            // Width constrained
            return CGSize(width: maxWidth, height: maxWidth / aspect)
        } else {
            // Height constrained
            return CGSize(width: maxHeight * aspect, height: maxHeight)
        }
    }
    
    private func handleTap(at location: CGPoint, in geometry: GeometryProxy, isDragStart: Bool = false, isDragging: Bool = false, isDragEnd: Bool = false) {
        let imageSize = image.size
        let containerSize = calculateContainerSize(for: geometry.size, withAspectRatio: imageSize.width / imageSize.height)
        let containerOrigin = CGPoint(
            x: (geometry.size.width - containerSize.width) / 2,
            y: (geometry.size.height - containerSize.height) / 2
        )
        
        // Convert tap location to container coordinates
        let containerLocation = CGPoint(
            x: location.x - containerOrigin.x,
            y: location.y - containerOrigin.y
        )
        
        // Convert to absolute image coordinates
        let imageX = (containerLocation.x / containerSize.width) * imageSize.width
        let imageY = (containerLocation.y / containerSize.height) * imageSize.height
        let imagePoint = CGPoint(x: imageX, y: imageY)
        
        // Only process tap if it's within the container bounds
        if containerLocation.x >= 0 && containerLocation.x <= containerSize.width &&
            containerLocation.y >= 0 && containerLocation.y <= containerSize.height {
            if isDragStart {
                viewModel.startDragging(at: imagePoint)
            } else if isDragging {
                viewModel.updateDragging(at: imagePoint)
            } else if isDragEnd {
                viewModel.endDragging(at: imagePoint)
            } else {
                viewModel.processImageTap(at: imagePoint)
            }
        }
    }
    
    private func convertToViewCoordinates(_ point: CGPoint, imageSize: CGSize, viewSize: CGSize) -> CGPoint {
        return CGPoint(
            x: point.x * viewSize.width / imageSize.width,
            y: point.y * viewSize.height / imageSize.height
        )
    }
    
    private func predictPrice() async {
        isPredicting = true
        defer { isPredicting = false }
        
        do {
            let (price, products) = try await viewModel.requestPrediction()
            predictedPrice = price
            similarProducts = products
            showingPredictionResult = true
        } catch {
            // Handle error if needed
            print("Prediction error:", error)
        }
    }
} 