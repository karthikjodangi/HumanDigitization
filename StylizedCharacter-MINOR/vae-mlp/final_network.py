import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import torch.nn as nn
from RES_VAE_Dynamic import VAE
import os
import glob

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),  # Ensure the image is resized to 64x64
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def load_vae_model(model_path, device):
    vae_net = VAE().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    vae_net.load_state_dict(checkpoint['model_state_dict'])
    vae_net.eval()
    return vae_net

def reshape_to_matrices(flat_encodings, output_shape):
    return flat_encodings.view(flat_encodings.size(0), *output_shape)

# Paths
input_dir = "data/input"
fer_model_path = 'Models/fer_64.pt'
character_model_path = 'Models/mery_128.pt'
mlp_model_path = 'Models/mlp_model.pth'

# Load models
human_vae = load_vae_model(fer_model_path, device)
encoder = human_vae.encoder
char_vae = load_vae_model(character_model_path, device)
decoder = char_vae.decoder

# Load MLP model
mlp_model = None  
mlp_model_loaded = False  

for image_path in glob.glob(os.path.join(input_dir, '*.jpg')):
 
    image_tensor = preprocess_image(image_path).to(device)

    # Encode image
    with torch.no_grad():
        encoding, _, _ = encoder(image_tensor)
        print(f"Encoding shape after VAE encoder: {encoding.shape}")

    # Flatten encoding
    encoding = encoding.view(encoding.size(0), -1)
    print(f"Flattened encoding shape: {encoding.shape}")

    # Load MLP model if not already loaded
    if not mlp_model_loaded:
        mlp_input_size = encoding.size(1)
        mlp_output_size = 128 * 8 * 8  
        mlp_model = MLP(input_size=mlp_input_size, output_size=mlp_output_size).to(device)
        mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device))
        mlp_model.eval()
        mlp_model_loaded = True

    # MLP forward pass
    with torch.no_grad():
        encoded_vector2 = mlp_model(encoding)
        print(f"MLP output shape: {encoded_vector2.shape}")

    # Reshape MLP output
    encoded_vector2 = reshape_to_matrices(encoded_vector2, (128, 8, 8))
    print(f"Reshaped MLP output shape: {encoded_vector2.shape}")

    # Decode image
    with torch.no_grad():
        recon_img = decoder(encoded_vector2)
        print(f"Reconstructed image shape: {recon_img.shape}")

    # Create output directory if it doesn't exist
    output_dir = "mery2-reconstructed"
    os.makedirs(output_dir, exist_ok=True)

    # Save reconstructed image
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    vutils.save_image(recon_img, output_image_path, normalize=True)

    print("Reconstructed image saved at:", output_image_path)