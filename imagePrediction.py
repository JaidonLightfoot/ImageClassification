import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import sys
import os
import json

MODEL_PATH = "resnet50_model.pth"
NUM_CLASSES = 5
CLASS_NAMES = ["Bathroom", "Bedroom", "Exterior", "Kitchen", "Living Room"]  


# Load the model
def load_model():
    model = models.resnet50()
    num_features = model.fc.in_features
    model.fc = torch.nn.Linear(num_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model


# Preprocess the image
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension


# Convert to JPG if needed
def convert_to_jpg(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        new_path = image_path.rsplit('.', 1)[0] + ".jpg"
        image.save(new_path, "JPEG")
        return new_path
    except Exception as e:
        raise RuntimeError(f"Failed to convert image to JPG: {e}")


# Run prediction and return JSON output
def predict(image_path):
    original_path = image_path
    if not image_path.lower().endswith(".jpg"):
        try:
            image_path = convert_to_jpg(image_path)
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            return

    model = load_model()
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    predicted_label = CLASS_NAMES[predicted_class.item()]
    confidence_score = int(confidence.item() * 100)  # Convert to int 

    # JSON response
    result = {
        "filename": os.path.basename(original_path),
        "roomType": predicted_label,
        "score": confidence_score
    }

    print(json.dumps(result))

    # Clean up temp jpg if created
    if original_path != image_path and os.path.exists(image_path):
        os.remove(image_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(json.dumps({"error": f"Image file '{image_path}' not found."}))
        sys.exit(1)

    predict(image_path)
