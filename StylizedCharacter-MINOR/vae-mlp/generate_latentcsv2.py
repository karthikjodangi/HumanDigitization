import torch
from torchvision import transforms
from PIL import Image
import csv
import os
import re
from RES_VAE_Dynamic import VAE 

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
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0)  
    return image_tensor

def zero_pad_numbers(s):
    return re.sub(r'(\d+)', lambda x: x.group().zfill(3), s)

def extract_last_two_components(image_path):
    parts = image_path.split(os.path.sep)
    return os.path.sep.join(parts[-3:])

def save_encodings_to_csv(encodings, image_paths, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['image_path'] + ['encoding_{}'.format(i) for i in range(encodings.shape[1])]
        writer.writerow(header)
        for encoding, image_path in zip(encodings, image_paths):
            modified_image_path = extract_last_two_components(image_path)
            modified_image_path = zero_pad_numbers(modified_image_path)
            writer.writerow([modified_image_path] + encoding.cpu().numpy().tolist())

model_path = 'Models/malcolm_128.pt'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loaded_model = load_model(model_path, device)
encoder = loaded_model.encoder

root_image_dir = 'data3-malcolm/character_expression'

output_csv = 'latents/malcolm_latent128.csv'

encodings = []
image_paths = []

for subdir, _, files in os.walk(root_image_dir):
    for file in files:
        image_path = os.path.join(subdir, file)
        if os.path.isfile(image_path) and image_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            image_tensor = preprocess_image(image_path).to(device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():

                    encoded_image = encoder(image_tensor)[0]  
                    
                    encodings.append(encoded_image.squeeze())
                    image_paths.append(image_path)

encodings_tensor = torch.stack(encodings)

save_encodings_to_csv(encodings_tensor, image_paths, output_csv)

print(f'Encodings saved to {output_csv}')
