import torch
from torchvision import transforms
from PIL import Image
import csv
import os
import re
from RES_VAE_Dynamic import VAE  # Assuming this is where your VAE class is defined

def load_model(model_path, device):
    # Initialize the VAE model
    vae_net = VAE().to(device)

    # Load the state dictionary
    checkpoint = torch.load(model_path, map_location=device)
    vae_net.load_state_dict(checkpoint['model_state_dict'])

    # Set the model to evaluation mode
    vae_net.eval()

    return vae_net

def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB if not already

    # Example preprocessing using torchvision transforms:
    transform = transforms.Compose([
        transforms.Resize(64),
        # transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def zero_pad_numbers(s):
    # Add leading zeros to any numbers in the string
    return re.sub(r'(\d+)', lambda x: x.group().zfill(3), s)

def extract_last_two_components(image_path):
    # Extract the last two components of the path
    parts = image_path.split(os.path.sep)
    return os.path.sep.join(parts[-3:])

def save_encodings_to_csv(encodings, image_paths, output_file):
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header (assuming encodings are 1D vectors)
        header = ['image_path'] + ['encoding_{}'.format(i) for i in range(encodings.shape[1])]
        writer.writerow(header)
        # Write encodings with their corresponding image paths
        for encoding, image_path in zip(encodings, image_paths):
            # Extract the last two components and zero-pad numbers
            modified_image_path = extract_last_two_components(image_path)
            modified_image_path = zero_pad_numbers(modified_image_path)
            writer.writerow([modified_image_path] + encoding.cpu().numpy().tolist())

# Set the path to your saved model
model_path = 'Models/fer_64.pt'
# Set the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
loaded_model = load_model(model_path, device)
encoder = loaded_model.encoder

# Directory containing images to process
root_image_dir = 'data5-tomnjerry/human_expression'
# Output CSV file to store encodings
output_csv = 'latents/fer-tom_latent64.csv'

# Iterate over images in the directory and its subdirectories
encodings = []
image_paths = []

for subdir, _, files in os.walk(root_image_dir):
    for file in files:
        image_path = os.path.join(subdir, file)
        if os.path.isfile(image_path) and image_path.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            image_tensor = preprocess_image(image_path).to(device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # Assuming the encoder outputs the latent vector
                    encoded_image = encoder(image_tensor)[0]  # Get the first element of the tuple if needed
                    
                    # Append the encoding and image path to the list
                    encodings.append(encoded_image.squeeze())
                    image_paths.append(image_path)

# Convert the list of encodings to a tensor
encodings_tensor = torch.stack(encodings)

# Save encodings to CSV
save_encodings_to_csv(encodings_tensor, image_paths, output_csv)

print(f'Encodings saved to {output_csv}')
