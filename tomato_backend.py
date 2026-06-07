import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from google import genai
from google.colab import userdata

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformation = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )
])

dataset_classes = [
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]


class ImageClassificationBase(nn.Module):
    def validation_step(self, batch):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        acc = Accuracy(outputs, labels)
        return {f"val_loss": loss.detach()}

class ResNet(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        try:
            self.network = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except AttributeError:
            self.network = models.resnet18(pretrained = True)

        num_fltrs = self.network.fc.in_features

        # Adding the Dropouts
        self.network.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_fltrs, len(dataset_classes))
        )

    def forward(self, xb):
        return self.network(xb)

def load_prediction_engine():
    model_instance = ResNet().to(device)

    checkpoint_path = "best_model.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location = device)
        model_instance.load_state_dict(checkpoint['model_state_dict'])
        print(f"🎯 [Backend] State dict parameters parsed successfully from '{checkpoint_path}'")
    else:
        print(f"⚠️ [Backend] Warning: '{checkpoint_path}' not discovered. Initializing raw baseline model layers.")

    model_instance.eval()
    return model_instance

def get_gemni_client():
    GOOGLE_API_KEY = userdata.get('API_KEY')
    return genai.Client(api_key=GOOGLE_API_KEY)
