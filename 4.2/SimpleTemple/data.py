from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
from torchvision import datasets, transforms

def load_data():
    # Load using HuggingFace datasets
    ds = load_dataset("uoft-cs/cifar10")
    
    # Set PyTorch format
    ds['train'].set_format(type='torch', columns=['img', 'label'])
    ds['test'].set_format(type='torch', columns=['img', 'label'])
    
    return ds['train'], ds['test']

def create_data_loaders(train_dataset, test_dataset, batch_size=64, num_workers=8):

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def prepare_batch(batch, device):
    # Extract inputs and targets
    inputs, targets = batch['img'], batch['label']
    # Move to device
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Convert types if needed
    inputs = inputs.float()
    targets = targets.long()
    
    return inputs, targets

# train_dataset, test_dataset = load_data()
# train_loader, test_loader = create_data_loaders(train_dataset, test_dataset)