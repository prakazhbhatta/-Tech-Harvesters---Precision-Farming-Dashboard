import torch
from PIL import Image
import torchvision.transforms as transforms
import os

# Import the same model class from backend
IMG_SIZE = 128

class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 128), torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def test_model(model_path, num_classes, model_name):
    print(f"\n=== Testing {model_name} Model ===")
    try:
        # Load model
        model = SimpleCNN(num_classes)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        print(f"✓ {model_name} model loaded successfully")
        
        # Test with dummy input
        dummy_input = torch.randn(1, 3, 128, 128)
        with torch.no_grad():
            output = model(dummy_input)
            pred = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, pred].item()
        
        print(f"✓ Model prediction test passed")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Predicted class: {pred}")
        print(f"  - Confidence: {confidence:.3f}")
        
        return True
    except Exception as e:
        print(f"✗ Error testing {model_name} model: {e}")
        return False

if __name__ == "__main__":
    print("Testing newly trained models...")
    
    # Test potato model
    potato_success = test_model('models/potato_leaf_model.pt', 3, 'Potato')
    
    # Test tomato model  
    tomato_success = test_model('models/tomato_leaf_model.pt', 10, 'Tomato')
    
    if potato_success and tomato_success:
        print("\n🎉 All models are working correctly!")
        print("Your backend should now work with these models.")
    else:
        print("\n❌ Some models failed to load. Check the errors above.") 