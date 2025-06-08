import SwiftUI

struct PredictionResultView: View {
    let price: Double
    let similarProducts: [UIImage]
    @Binding var navigationPath: NavigationPath
    @Binding var shouldResetToRoot: Bool
    
    // State for managing the current image index
    @State private var currentIndex = 0
    @State private var offset: CGFloat = 0
    @State private var isDragging = false
    
    var body: some View {
        VStack(spacing: 30) {
            Text("Предполагаемая цена")
                .font(.title)
                .foregroundColor(.primary)
            
            HStack(alignment: .center, spacing: 4) {
                Text("\(Int(round(price)))")
                    .font(.system(size: 48, weight: .bold))
                    .foregroundColor(.blue)
                Text("₽")
                    .font(.system(size: 48, weight: .bold))
                    .foregroundColor(.blue)
            }
            
            Spacer()
                .frame(height: 20)
            
            VStack(spacing: 20) {
                Text("Похожие товары на Авито")
                    .font(.system(size: 24, weight: .bold))
                    .foregroundColor(.primary)
                    .frame(maxWidth: .infinity, alignment: .center)
                
                // Carousel View
                GeometryReader { geometry in
                    let imageSize = min(geometry.size.width * 0.9, geometry.size.height)
                    
                    ZStack {
                        ForEach(0..<similarProducts.count, id: \.self) { index in
                            Image(uiImage: similarProducts[index])
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(width: imageSize, height: imageSize)
                                .clipShape(RoundedRectangle(cornerRadius: 15))
                                .shadow(radius: 5)
                                .opacity(currentIndex == index ? 1 : 0)
                                .scaleEffect(currentIndex == index ? 1 : 0.8)
                                .offset(x: offset)
                                .animation(.spring(), value: offset)
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .gesture(
                        DragGesture()
                            .onChanged { value in
                                isDragging = true
                                offset = value.translation.width
                            }
                            .onEnded { value in
                                isDragging = false
                                let threshold: CGFloat = 50
                                withAnimation(.spring()) {
                                    if value.translation.width > threshold {
                                        // Swipe right
                                        currentIndex = (currentIndex - 1 + similarProducts.count) % similarProducts.count
                                    } else if value.translation.width < -threshold {
                                        // Swipe left
                                        currentIndex = (currentIndex + 1) % similarProducts.count
                                    }
                                    offset = 0
                                }
                            }
                    )
                }
                
                // Page Indicator
                HStack {
                    ForEach(0..<similarProducts.count, id: \.self) { index in
                        Circle()
                            .fill(currentIndex == index ? Color.blue : Color.gray.opacity(0.5))
                            .frame(width: 8, height: 8)
                    }
                }
                .frame(maxWidth: .infinity)
            }
            
            Spacer()
            
            Button(action: {
                shouldResetToRoot = true
                navigationPath.removeLast(navigationPath.count)
            }) {
                Text("Вернуться к началу")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(10)
            }
            .padding(.horizontal)
        }
        .padding()
        .navigationBarBackButtonHidden(true)
    }
} 
