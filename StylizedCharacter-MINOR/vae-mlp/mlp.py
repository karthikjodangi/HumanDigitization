import ast
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

class EncodingsDataset(Dataset):
    def __init__(self, input_csv, ground_truth_csv):
        self.input_data = pd.read_csv(input_csv)
        self.ground_truth_data = pd.read_csv(ground_truth_csv)

        # Drop the file name column
        self.input_data = self.input_data.drop(columns=['image_path'])
        self.ground_truth_data = self.ground_truth_data.drop(columns=['image_path'])

        # Convert each row of strings to a list of 2x2 matrices and flatten them correctly
        self.input_tensor = torch.tensor(
            [self.parse_encoding(row) for row in self.input_data.values], dtype=torch.float32
        )
        self.ground_truth_tensor = torch.tensor(
            [self.parse_encoding(row) for row in self.ground_truth_data.values], dtype=torch.float32
        )

        # Print the first encoding to verify
        print("First input encoding:", self.input_tensor[0])
        print("First ground truth encoding:", self.ground_truth_tensor[0])

    def parse_encoding(self, row):
        # Convert each stringified 2x2 matrix into a list, then flatten it and concatenate
        matrices = [ast.literal_eval(matrix_str) for matrix_str in row]
        flat_encoding = [item for matrix in matrices for sublist in matrix for item in sublist]
        return flat_encoding

    def __len__(self):
        return len(self.input_tensor)

    def __getitem__(self, idx):
        return self.input_tensor[idx], self.ground_truth_tensor[idx]

# Adjust the input and output sizes accordingly
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

def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    return model

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda() if torch.cuda.is_available() else inputs, targets.cuda() if torch.cuda.is_available() else targets
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in dataloader:
            inputs, targets = inputs.cuda() if torch.cuda.is_available() else inputs, targets.cuda() if torch.cuda.is_available() else targets
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute the loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    print('Training finished.')

    # Save the trained model
    model_save_path = 'Models/mlp_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # Evaluate the model
    evaluation_loss = evaluate(model, dataloader, criterion)
    print(f'Evaluation Loss: {evaluation_loss:.4f}')

if __name__ == "__main__":
    input_csv_path = 'latents/fer-latent64.csv'
    ground_truth_csv_path = 'latents/mery_latent128.csv'

    dataset = EncodingsDataset(input_csv_path, ground_truth_csv_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_size = dataset.input_tensor.shape[1]  
    output_size = dataset.ground_truth_tensor.shape[1]  
    

    print(input_size,output_size)

    model = MLP(input_size, output_size).cuda() if torch.cuda.is_available() else MLP(input_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50

    train(model, dataloader, criterion, optimizer, num_epochs)
