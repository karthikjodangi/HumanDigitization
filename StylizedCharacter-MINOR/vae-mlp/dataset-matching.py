import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import sys
from tqdm import tqdm

class ExpressionMatchingModel(nn.Module):
    def __init__(self):
        super(ExpressionMatchingModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 512)
        self.fc2 = nn.Linear(512, 7)  # 7 output classes

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define your custom dataset class
class ExpressionDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.classes = os.listdir(data_dir)

        self.data = []
        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(data_dir, class_name)
            for filename in os.listdir(class_dir):
                img_path = os.path.join(class_dir, filename)
                self.data.append((img_path, i))  # (image path, class index)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        if self.transform:
            image = self.transform(image)
        return image, label
    
    
# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((48, 48)),  # Resize images to the desired size
    transforms.ToTensor(),         # Convert PIL image to tensor
])

# Load datasets
human_dataset = ExpressionDataset(data_dir='data/human_expression-tomnjerry', transform=transform)
character_dataset = ExpressionDataset(data_dir='data/tom', transform=transform)

# Define data loaders
human_loader = DataLoader(human_dataset, batch_size=64, shuffle=True)
character_loader = DataLoader(character_dataset, batch_size=64, shuffle=True)

# Initialize your model and move it to CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ExpressionMatchingModel().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ExpressionMatchingModel().to(device)
model.load_state_dict(torch.load('Models/tom-expression_matching_model.pth'))
model.eval()


def generate_character_expression(human_expression_path):
    model.eval()
    human_expression = Image.open(human_expression_path).convert('L')
    human_expression = transform(human_expression).unsqueeze(0).to(device)  # Move data to CUDA and add batch dimension
    with torch.no_grad():
        character_class = model(human_expression)
    _, predicted_class = torch.max(character_class, 1)
    return human_dataset.classes[predicted_class.item()]  # Return the class label instead of the index

def calculate_similarity(image1, image2):
    array1 = np.array(image1)
    array2 = np.array(image2)
    mse = np.mean((array1 - array2) ** 2)
    return mse

def save_character_image(character_label, human_expression, human_expression_path, output_folder):
    human_expression_name, human_expression_ext = os.path.splitext(os.path.basename(human_expression_path))
    result_image_path = os.path.join(output_folder, human_expression_name + human_expression_ext)

    character_class_dir = os.path.join('data/tom', character_label)
    character_images = os.listdir(character_class_dir)
    best_similarity = float('inf')
    best_image_path = None
    
    for image_name in character_images:
        character_image_path = os.path.join(character_class_dir, image_name)
        character_image = Image.open(character_image_path).convert('L')
        
        # Resize character image to match the size of human expression image
        character_image_resized = character_image.resize(human_expression.size)
        
        similarity = calculate_similarity(human_expression, character_image_resized)
        if similarity < best_similarity:
            best_similarity = similarity
            best_image_path = character_image_path
    
    best_character_image = Image.open(best_image_path)
    best_character_image_rgb = best_character_image.convert('RGB')
    best_character_image_rgb.save(result_image_path)

input_folder = 'data5-tomnjerry/human_expression/surprise'
output_folder = 'data5-tomnjerry/character_expression/surprise'

os.makedirs(output_folder, exist_ok=True)
count = 0
total_files = len([name for name in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, name))])

for filename in tqdm(os.listdir(input_folder), desc="Processing images"):
        image_path = os.path.join(input_folder, filename)
        predicted_class = generate_character_expression(image_path)
        if predicted_class == 'surprise':
            count += 1
        # print("Predicted character expression class:", predicted_class)
        human_expression = Image.open(image_path).convert('L')
        save_character_image(predicted_class, human_expression, image_path, output_folder)
        # print(f"Image saved: {filename} \n")
        
print(f"{count}/{total_files} have been predicted correctly.")