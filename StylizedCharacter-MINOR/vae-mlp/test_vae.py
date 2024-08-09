import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
import torch.nn as nn

from RES_VAE_Dynamic import VAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, device):
    vae_net = VAE().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    vae_net.load_state_dict(checkpoint['model_state_dict'])

    vae_net.eval()

    return vae_net

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  
    transform = transforms.Compose([
        transforms.Resize(128), 
        # transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0) 
    return image_tensor

image_path = "data3-malcolm/character_expression/angry/4.jpg"  

image_tensor = preprocess_image(image_path).to(device)

fer_model_path = 'Models/malcolm_128.pt'

vae = load_model(fer_model_path, device)
encoder = vae.encoder
vae.eval()

with torch.no_grad():
    encoding, mu, log_var = encoder(image_tensor)

encoding = encoding.squeeze() 

decoder = vae.decoder

with torch.no_grad():
    recon_img = decoder(encoding.unsqueeze(0))

output_image_path = "reconstructed_image.jpg"  
vutils.save_image(recon_img, output_image_path, normalize=True)