import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import io
import base64

# 1. DEVICE
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2. CHARGER LE MODELE
def load_model():
    model = models.resnet50(weights=None)
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load('model/resnet50_best.pth',
                          map_location=device))
    model.to(device)
    return model

model = load_model()
print("✅ Modèle chargé !")

# 3. TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 4. PREDICTION + GRAD-CAM
def predict(image: Image.Image):
    # Preprocessing
    orig_array   = np.array(image.resize((224, 224))) / 255.0
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred    = torch.argmax(probs).item()

    label      = 'Tuberculosis' if pred == 1 else 'Normal'
    confidence = round(probs[pred].item() * 100, 2)

    # Grad-CAM
    model.train()  # ← important pour les gradients !
    target_layer  = [model.layer4[-1]]
    cam           = GradCAM(model=model, target_layers=target_layer)
    targets       = [ClassifierOutputTarget(pred)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    model.eval()  
    
    visualization = show_cam_on_image(
        orig_array.astype(np.float32),
        grayscale_cam, use_rgb=True
    )

    # Convertir heatmap en base64
    heatmap_img = Image.fromarray(visualization)
    buffer      = io.BytesIO()
    heatmap_img.save(buffer, format='PNG')
    heatmap_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {
        'label'      : label,
        'confidence' : confidence,
        'heatmap'    : heatmap_b64
    }