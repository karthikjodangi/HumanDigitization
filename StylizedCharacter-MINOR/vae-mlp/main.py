import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# Define your neural network architecture
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

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    for human_images, human_labels in human_loader:
        human_images, human_labels = human_images.to(device), human_labels.to(device)  # Move data to CUDA
        optimizer.zero_grad()
        human_outputs = model(human_images)
        loss = criterion(human_outputs, human_labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), 'Models/tomnjerry-expression_matching_model.pth')