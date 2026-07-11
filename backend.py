from flask import Flask, request, jsonify
import pandas as pd
import os
from PIL import Image
import torch
from torchvision import transforms
import io
import threading

app = Flask(__name__)

# Check if network is available
def is_online():
    try:
        requests.get('https://www.google.com', timeout=3)
        return True
    except requests.RequestException:
        return False

# CSV Search
def csv_search(query):
    if not os.path.exists('answers.csv'):
        return "No offline answers available."
    df = pd.read_csv('answers.csv')
    for _, row in df.iterrows():
        if row['Question'].strip().lower() == query.strip().lower():
            return row['Answer']
    return "No answer found offline."

# --- Model and Advice Setup ---
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

def load_model_threadsafe(path, num_classes):
    model = SimpleCNN(num_classes)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Potato model
POTATO_MODEL_PATH = 'models/potato_leaf_model.pt'
POTATO_CLASSES = ['Healthy', 'Early Blight', 'Late Blight']
POTATO_ADVICE = {
    'Healthy': 'No action needed.',
    'Early Blight': 'Remove affected leaves and apply fungicide.',
    'Late Blight': 'Destroy infected plants and use recommended fungicide.'
}
potato_model = load_model_threadsafe(POTATO_MODEL_PATH, num_classes=3)

# Tomato model
TOMATO_MODEL_PATH = 'models/tomato_leaf_model.pt'
TOMATO_CLASSES = [
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]
TOMATO_ADVICE = {c: 'Consult agricultural expert for advice.' for c in TOMATO_CLASSES}
tomato_model = load_model_threadsafe(TOMATO_MODEL_PATH, num_classes=10)

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Prediction Endpoints ---
def predict_with_model(model, class_labels, advice_dict, file):
    img = Image.open(file.stream).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        pred_label = class_labels[pred_idx] if pred_idx < len(class_labels) else str(pred_idx)
        confidence = torch.softmax(output, dim=1)[0, pred_idx].item()
        suggestion = advice_dict.get(pred_label, '')
    return pred_label, confidence, suggestion

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('message', '')
    if not question:
        return jsonify({'answer': 'Please ask a question.'})
    answer = csv_search(question)
    return jsonify({'answer': answer})

@app.route('/predict_potato', methods=['POST'])
def predict_potato():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        pred_label, confidence, suggestion = predict_with_model(potato_model, POTATO_CLASSES, POTATO_ADVICE, file)
        return jsonify({'class': pred_label, 'confidence': confidence, 'suggestion': suggestion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_tomato', methods=['POST'])
def predict_tomato():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        pred_label, confidence, suggestion = predict_with_model(tomato_model, TOMATO_CLASSES, TOMATO_ADVICE, file)
        return jsonify({'class': pred_label, 'confidence': confidence, 'suggestion': suggestion})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 