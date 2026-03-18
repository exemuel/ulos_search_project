import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class FeatureExtractor:
    def __init__(self):
        # 1. Load the pre-trained ResNet50 model
        weights = models.ResNet50_Weights.DEFAULT
        base_model = models.resnet50(weights=weights)
        
        # 2. Remove the final classification layer to get the embeddings (vectors)
        self.model = torch.nn.Sequential(*list(base_model.children())[:-1])
        self.model.eval() # Set model to evaluation mode (freezes weights)
        
        # 3. Define the image transformations required by ResNet
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, img):
        """Converts a PIL Image into a 1D feature vector."""
        # Ensure image is in RGB format (removes alpha channels if present)
        img = img.convert('RGB')
        
        # Apply transformations and add a batch dimension
        img_tensor = self.preprocess(img).unsqueeze(0)
        
        # Extract features without calculating gradients (saves memory)
        with torch.no_grad():
            feature = self.model(img_tensor)
            
        # Flatten the output to a 1D numpy array
        return feature.numpy().flatten()