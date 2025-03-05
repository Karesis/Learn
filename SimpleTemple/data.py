"""
Data loading utilities for neural network project.
Provides functions to load datasets and create data loaders.
"""

from torch.utils.data import DataLoader
from datasets import load_dataset
import torch
from torchvision import datasets, transforms


def load_fashion_mnist(use_huggingface=True):
    """
    Load the Fashion MNIST dataset
    
    Args:
        use_huggingface: Whether to use HuggingFace datasets or torchvision
        
    Returns:
        train_dataset, test_dataset: Training and test datasets
    """
    if use_huggingface:
        # Load using HuggingFace datasets
        ds = load_dataset("zalando-datasets/fashion_mnist")
        
        # Set PyTorch format
        ds['train'].set_format(type='torch', columns=['image', 'label'])
        ds['test'].set_format(type='torch', columns=['image', 'label'])
        
        return ds['train'], ds['test']
    else:
        # Load using torchvision
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=transform
        )
        
        test_dataset = datasets.FashionMNIST(
            root='./data',
            train=False,
            download=True,
            transform=transform
        )
        
        return train_dataset, test_dataset


def create_data_loaders(train_dataset, test_dataset, batch_size=64, num_workers=8):
    """
    Create PyTorch DataLoaders for training and testing
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        
    Returns:
        train_loader, test_loader: DataLoader objects
    """
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
    """
    Prepare a batch of data for the model
    
    Args:
        batch: Batch from DataLoader
        device: Device to move data to
        
    Returns:
        inputs, targets: Prepared inputs and targets
    """
    # Extract inputs and targets
    if isinstance(batch, dict):
        inputs, targets = batch['image'], batch['label']
    else:
        inputs, targets = batch
    
    # Move to device
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Convert types if needed
    inputs = inputs.float()
    targets = targets.long()
    
    # Add channel dimension if needed
    if len(inputs.shape) == 3:  # [batch, height, width]
        inputs = inputs.unsqueeze(1)  # [batch, 1, height, width]
    
    return inputs, targets


# For direct imports
train_dataset, test_dataset = load_fashion_mnist()
train_loader, test_loader = create_data_loaders(train_dataset, test_dataset)