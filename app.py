from flask import Flask, render_template, request, jsonify
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load the trained model
model = models.densenet121(pretrained=False)
num_classes = 5  # Assuming you have 5 classes in your dataset
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, num_classes)

# Load the trained model's weights
model.load_state_dict(torch.load('densenet121_cotton_leaf_model.pth'))
model.eval()

# Define the transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class labels with their respective indices
class_labels = {
    0: 'bacterial_blight',
    1: 'curl_virus',
    2: 'healthy',
    3: 'target_spot',
    4: 'fusarium_wilt'
}

# Function to predict the disease
def predict_disease(image_path):
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)  # Calculate softmax probabilities
        confidence, predicted_idx = torch.max(probs, 1)
    predicted_label = class_labels[predicted_idx.item()]
    return predicted_label, confidence.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image_path = 'temp_image.jpg'
        file.save(image_path)
        predicted_disease, confidence = predict_disease(image_path)
        return jsonify({'predicted_disease': predicted_disease, 'confidence': confidence})
    else:
        return jsonify({'error': 'File not supported'})
if __name__ == '__main__':
    app.run(debug=True)
