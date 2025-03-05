"""
Main entry point for the neural network project.
Provides command-line interface for training and evaluation.
"""

import argparse
import os
import torch

from models import create_model
from data import load_fashion_mnist, create_data_loaders
from train import train_model, load_checkpoint
from eval import detailed_evaluation, get_validation_loss
from utils import set_seed, plot_training_history
from config import DEVICE, DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, CLASS_NAMES


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fashion MNIST Neural Network')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'train_eval'],
                      default='train_eval', help='Mode of operation')
    
    # Model parameters
    parser.add_argument('--model', type=str, choices=['expert', 'cnn'],
                      default=MODEL_CONFIG['type'], help='Model type')
    parser.add_argument('--num_experts', type=int, default=MODEL_CONFIG['num_experts'],
                      help='Number of experts (for expert model)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['num_epochs'],
                      help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['learning_rate'],
                      help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=DATA_CONFIG['batch_size'],
                      help='Batch size')
    
    # Paths
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to checkpoint for evaluation or continued training')
    parser.add_argument('--save_dir', type=str, default=TRAIN_CONFIG['save_dir'],
                      help='Directory to save models')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    return parser.parse_args()


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set up paths
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"fashion_mnist_{args.model}")
    
    # Load data
    print("Loading data...")
    train_dataset, test_dataset = load_fashion_mnist(use_huggingface=DATA_CONFIG['use_huggingface'])
    
    train_loader, test_loader = create_data_loaders(
        train_dataset, 
        test_dataset, 
        batch_size=args.batch_size, 
        num_workers=DATA_CONFIG['num_workers']
    )
    
    # Create model
    print(f"Creating {args.model} model...")
    model = create_model(
        model_type=args.model,
        num_classes=MODEL_CONFIG['num_classes'],
        in_channels=MODEL_CONFIG['in_channels'],
        num_experts=args.num_experts
    )
    
    # Training mode
    if args.mode in ['train', 'train_eval']:
        # Load checkpoint if provided
        if args.checkpoint:
            model, start_epoch, _ = load_checkpoint(model, args.checkpoint, device=DEVICE)
            print(f"Resuming training from epoch {start_epoch+1}")
        
        # Define validation loss function (for learning rate scheduler)
        def val_loss_fn(model):
            return get_validation_loss(model, test_loader, device=DEVICE)
        
        # Train model
        print("Training model...")
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=args.epochs,
            lr=args.lr,
            device=DEVICE,
            save_path=save_path
        )
        
        # Plot training history
        plot_training_history(history)
    
    # Evaluation mode
    if args.mode in ['eval', 'train_eval']:
        # Load checkpoint if in eval mode only
        if args.mode == 'eval' and args.checkpoint:
            model, _, _ = load_checkpoint(model, args.checkpoint, device=DEVICE)
        
        # Evaluate model
        print("Evaluating model...")
        results = detailed_evaluation(model, test_loader, CLASS_NAMES, DEVICE)
        print(f"Final accuracy: {results['accuracy']:.2f}%")


if __name__ == "__main__":
    main()