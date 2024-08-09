import os
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import torch.nn as nn
from RES_VAE_Dynamic import VAE

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

def preprocess_image(image_path, size=(64, 64)):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(size),
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
input_folder = "data5-tomnjerry/human_expression"
ground_truth_folder = "data5-tomnjerry/character_expression"
fer_model_path = 'Models/fer_64.pt'
character_model_path = 'Models/tom_128.pt'
mlp_model_path = 'Models/mlp_model.pth'
output_folder = "tom_reconstructed_images"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load models
human_vae = load_vae_model(fer_model_path, device)
encoder = human_vae.encoder
char_vae = load_vae_model(character_model_path, device)
decoder = char_vae.decoder

# Load MLP model
mlp_input_size = 128 * 4 * 4  # Assuming encoding is flattened from this shape
mlp_output_size = 128 * 8 * 8  # Adjust according to the expected decoder input
mlp_model = MLP(input_size=mlp_input_size, output_size=mlp_output_size).to(device)
mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device))
mlp_model.eval()

# Process each image in the input folder and its subfolders
for root, _, files in os.walk(input_folder):
    for image_name in files:
        image_path = os.path.join(root, image_name)
        
        # Create corresponding subfolder structure in the output folder
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)
        
        # Path for the ground truth image
        gt_image_path = os.path.join(ground_truth_folder, relative_path, image_name)
        
        # Preprocess input image
        input_image_tensor = preprocess_image(image_path, size=(64, 64)).to(device)
        
        # Preprocess ground truth image
        gt_image_tensor = preprocess_image(gt_image_path, size=(128, 128)).to(device)
        
        # Encode input image
        with torch.no_grad():
            encoding, _, _ = encoder(input_image_tensor)
            encoding = encoding.view(encoding.size(0), -1)
        
        # MLP forward pass
        with torch.no_grad():
            encoded_vector2 = mlp_model(encoding)
        
        # Reshape MLP output
        encoded_vector2 = reshape_to_matrices(encoded_vector2, (128, 8, 8))
        
        # Decode image
        with torch.no_grad():
            recon_img = decoder(encoded_vector2)
        
        # Resize input and ground truth images to match the reconstructed image size
        recon_img_size = recon_img.size(2), recon_img.size(3)  # (Height, Width)
        resize_transform = transforms.Resize(recon_img_size)
        input_image_resized = resize_transform(input_image_tensor.squeeze(0)).unsqueeze(0)
        gt_image_resized = resize_transform(gt_image_tensor.squeeze(0)).unsqueeze(0)
        
        # Concatenate input, ground truth, and reconstructed images
        concatenated_img = torch.cat((input_image_resized, gt_image_resized, recon_img), dim=3)  # Concatenate along width
        
        # Save concatenated image
        output_image_path = os.path.join(output_subfolder, image_name)
        vutils.save_image(concatenated_img, output_image_path, normalize=True)
        print(f"Concatenated image saved at: {output_image_path}")
