"""
Configuration for neural network project.
Contains global settings and parameters.
"""

import os
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data configuration
DATA_CONFIG = {
    'batch_size': 64,
    'num_workers': 8,
    'use_huggingface': True
}

# Model configuration
MODEL_CONFIG = {
    'in_channels': 3,
    'num_classes': 10,
    'num_experts': 3
}

# Training configuration
TRAIN_CONFIG = {
    'num_epochs': 10,
    'learning_rate': 0.001,
    'save_dir': 'models'
}

# Paths configuration
PATHS = {
    'data_dir': 'data',
    'models_dir': 'models',
    'results_dir': 'results'
}

# Create directories if they don't exist
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# Fashion MNIST class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]