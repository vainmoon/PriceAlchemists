import SwiftUI

struct PredictionResultView: View {
    let price: Double
    @Binding var navigationPath: NavigationPath
    @Binding var shouldResetToRoot: Bool
    
    var body: some View {
        VStack(spacing: 30) {
            Text("Estimated Price")
                .font(.title)
                .foregroundColor(.primary)
            
            Text("$\(String(format: "%.2f", price))")
                .font(.system(size: 48, weight: .bold))
                .foregroundColor(.blue)
            
            Button(action: {
                shouldResetToRoot = true
                navigationPath.removeLast(navigationPath.count)
            }) {
                Text("Return to Start")
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(10)
            }
            .padding(.horizontal)
        }
        .navigationBarBackButtonHidden(true)
    }
} 