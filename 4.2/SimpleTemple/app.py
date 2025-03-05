"""
User application for the neural network project.
Provides a simple command-line interface for model demo.
"""

import os
import sys
import torch

from models import create_model, MultiFashionExpertWithLSTM
from data import test_loader
from eval import show_random_prediction
from utils import clear_screen
from config import CLASS_NAMES


def load_model(model_path):
    """
    Load a trained model
    
    Args:
        model_path: Path to model checkpoint
        
    Returns:
        model: Loaded model or None if loading fails
    """
    print(f"Loading model: {model_path}")
    
    # Create model instance
    model = MultiFashionExpertWithLSTM(num_classes=10, in_channels=1, num_experts=3)
    
    # Load weights
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Successfully loaded model! Epoch: {checkpoint['epoch']}")
        print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")
        print(f"Training accuracy: {checkpoint.get('train_acc', 'N/A')}")
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def evaluate_model(model):
    """
    Evaluate model on test set
    
    Args:
        model: Model to evaluate
    """
    if model is None:
        print("Please load a model first!")
        return
    
    model.eval()
    device = torch.device('cpu')  # Use CPU for evaluation in the app
    
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for batch in test_loader:
            # Extract data
            if isinstance(batch, dict):
                images, labels = batch['image'], batch['label']
            else:
                images, labels = batch
            
            # Convert types
            images = images.float()
            labels = labels.long()
            
            # Add channel dimension if needed
            if len(images.shape) == 3:
                images = images.unsqueeze(1)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Update statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    # Print overall accuracy
    print(f"\nOverall accuracy: {100 * correct / total:.2f}%")
    
    # Print per-class accuracy
    print("\nClass Accuracy:")
    for i in range(10):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f"{CLASS_NAMES[i]}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    # Find most difficult class
    min_acc_idx = min(range(10), key=lambda i: class_correct[i]/class_total[i] if class_total[i] > 0 else 1.0)
    print("\nModel Analysis:")
    print(f"Most difficult class: {CLASS_NAMES[min_acc_idx]} "
          f"(Accuracy: {100 * class_correct[min_acc_idx] / class_total[min_acc_idx]:.2f}%)")


def display_menu():
    """Display the main menu"""
    clear_screen()
    print("=" * 50)
    print("    Fashion MNIST Model Demo")
    print("=" * 50)
    print("1. Load Model")
    print("2. Show Random Prediction")
    print("3. Show Multiple Predictions (5)")
    print("4. Evaluate Model Performance")
    print("5. Exit")
    print("=" * 50)


def main():
    """Main function"""
    model = None
    default_model_path = "./models/fashion_mnist_expert_final.pt"
    
    # Try to load default model if it exists
    if os.path.exists(default_model_path):
        model = load_model(default_model_path)
    
    while True:
        display_menu()
        
        # Show model status
        if model is not None:
            print(f"Current model: {default_model_path}")
        else:
            print("No model loaded")
        
        # Get user choice
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            # Load model
            model_path = input("Enter model path (press Enter for default): ").strip()
            if not model_path:
                model_path = default_model_path
            
            model = load_model(model_path)
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            # Show random prediction
            if model is not None:
                print("Generating prediction...")
                try:
                    show_random_prediction(model, test_loader, CLASS_NAMES, num_samples=1)
                    input("\nClose the plot window and press Enter to continue...")
                except Exception as e:
                    print(f"Error: {e}")
                    input("\nPress Enter to continue...")
            else:
                print("Please load a model first!")
                input("\nPress Enter to continue...")
                
        elif choice == '3':
            # Show multiple predictions
            if model is not None:
                print("Generating predictions...")
                try:
                    show_random_prediction(model, test_loader, CLASS_NAMES, num_samples=5)
                    input("\nClose the plot window and press Enter to continue...")
                except Exception as e:
                    print(f"Error: {e}")
                    input("\nPress Enter to continue...")
            else:
                print("Please load a model first!")
                input("\nPress Enter to continue...")
                
        elif choice == '4':
            # Evaluate model
            if model is not None:
                print("Evaluating model...")
                try:
                    evaluate_model(model)
                    input("\nPress Enter to continue...")
                except Exception as e:
                    print(f"Error: {e}")
                    input("\nPress Enter to continue...")
            else:
                print("Please load a model first!")
                input("\nPress Enter to continue...")
                
        elif choice == '5':
            # Exit
            print("Goodbye!")
            sys.exit(0)
            
        else:
            print("Invalid choice!")
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()