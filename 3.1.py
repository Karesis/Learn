import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
import numpy as np

# 1. Load dataset
mnist = load_dataset('mnist')
print(f"训练集大小: {len(mnist['train'])}, 测试集大小: {len(mnist['test'])}")

# 2. Efficient preprocessing function
def preprocess_images(examples):
    # Convert PIL images to numpy arrays
    images = np.array([np.array(img) for img in examples['image']])
    
    # Convert to PyTorch tensor, add channel dimension, and normalize in one step
    images = torch.tensor(images, dtype=torch.float32).unsqueeze(1) / 255.0
    
    # Convert labels to PyTorch tensor
    labels = torch.tensor(examples['label'], dtype=torch.long)
    
    return {'pixel_values': images, 'labels': labels}

# Apply preprocessing once
mnist_pytorch = mnist.map(preprocess_images, batched=True, batch_size=1000)

# 3. Simplified Dataset class without redundant checks
class EfficientMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Retrieve pre-converted tensors directly
        return self.dataset[idx]['pixel_values'], self.dataset[idx]['labels']

# Create datasets
train_dataset = EfficientMNISTDataset(mnist_pytorch['train'])
test_dataset = EfficientMNISTDataset(mnist_pytorch['test'])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# 4. Simplified training function without redundant checks
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        # Data is already in tensor format, just move to device
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc