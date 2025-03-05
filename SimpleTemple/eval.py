"""
Evaluation utilities for neural network project.
Contains functions for evaluating models and visualizing results.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix

from data import prepare_batch


def evaluate_model(model, test_loader, criterion, device=None):
    """
    Evaluate model performance on test set
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader with test data
        criterion: Loss function
        device: Device to use (cuda/cpu)
        
    Returns:
        val_loss: Test loss
        accuracy: Test accuracy
        all_preds: All predictions
        all_targets: All true labels
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    # Create progress bar
    progress_bar = tqdm(test_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Prepare data
            inputs, targets = prepare_batch(batch, device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Update statistics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Collect predictions and targets
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / total,
                'acc': 100. * correct / total
            })
    
    # Calculate final statistics
    val_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'Evaluation Loss: {val_loss:.4f} | Accuracy: {accuracy:.2f}%')
    
    return val_loss, accuracy, np.array(all_preds), np.array(all_targets)


def get_validation_loss(model, test_loader, criterion=None, device=None):
    """
    Get validation loss only (helper function for learning rate scheduler)
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader with test data
        criterion: Loss function (defaults to CrossEntropyLoss)
        device: Device to use (cuda/cpu)
        
    Returns:
        val_loss: Validation loss
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    val_loss, _, _, _ = evaluate_model(model, test_loader, criterion, device)
    return val_loss


def detailed_evaluation(model, test_loader, class_names=None, device=None):
    """
    Perform detailed evaluation of a model
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader with test data
        class_names: List of class names
        device: Device to use (cuda/cpu)
        
    Returns:
        results: Dictionary with evaluation results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    criterion = nn.CrossEntropyLoss()
    
    # Get evaluation results
    val_loss, accuracy, all_preds, all_targets = evaluate_model(
        model, test_loader, criterion, device
    )
    
    # Default class names if not provided
    if class_names is None:
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Calculate per-class accuracy
    class_correct = np.zeros(len(class_names))
    class_total = np.zeros(len(class_names))
    
    for i in range(len(all_targets)):
        label = all_targets[i]
        pred = all_preds[i]
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1
    
    # Print per-class accuracy
    print("\nClass Accuracy:")
    for i in range(len(class_names)):
        if class_total[i] > 0:
            print(f'{class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    # Plot confusion matrix
    plot_confusion_matrix(all_targets, all_preds, class_names)
    
    # Return results
    results = {
        'val_loss': val_loss,
        'accuracy': accuracy,
        'class_correct': class_correct,
        'class_total': class_total,
        'all_preds': all_preds,
        'all_targets': all_targets
    }
    
    return results


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=False, figsize=(10, 8)):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    
    # Set up figure
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    
    # Set up ticks
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def show_random_prediction(model, test_loader, class_names=None, num_samples=1, device=None):
    """
    Show random predictions from the model
    
    Args:
        model: Model to use for predictions
        test_loader: DataLoader with test data
        class_names: List of class names
        num_samples: Number of samples to show
        device: Device to use (cuda/cpu)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model is None:
        print("Please provide a model!")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of data
    dataiter = iter(test_loader)
    batch = next(dataiter)
    
    # Get inputs and targets
    if isinstance(batch, dict):
        images, labels = batch['image'], batch['label']
    else:
        images, labels = batch
    
    # Choose random samples
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    
    # Create figure
    plt.figure(figsize=(12, 4 * num_samples))
    
    for i, idx in enumerate(indices):
        # Get image and label
        image = images[idx]
        label = labels[idx].item()
        
        # Prepare for display
        display_image = image.numpy()
        if len(display_image.shape) == 3:  # [channels, height, width]
            display_image = display_image.squeeze(0)  # Remove channel dim for grayscale
        
        # Prepare model input
        input_tensor = image.float().unsqueeze(0).to(device)  # Add batch dimension
        if len(input_tensor.shape) == 3:  # [batch, height, width]
            input_tensor = input_tensor.unsqueeze(1)  # Add channel dimension
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            pred_idx = predicted.item()
            probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
        
        # Display image
        plt.subplot(num_samples, 2, i*2 + 1)
        plt.imshow(display_image, cmap='gray')
        
        # Set title
        if class_names is not None:
            true_label = class_names[label]
            pred_label = class_names[pred_idx]
        else:
            true_label = f"Class {label}"
            pred_label = f"Class {pred_idx}"
        
        color = 'green' if label == pred_idx else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
        
        # Display probability bar chart
        plt.subplot(num_samples, 2, i*2 + 2)
        bars = plt.barh(range(len(probs)), probs)
        
        # Set ticks
        if class_names is not None:
            plt.yticks(range(len(class_names)), class_names)
        else:
            plt.yticks(range(len(probs)), [f"Class {j}" for j in range(len(probs))])
        
        # Highlight true class and predicted class
        bars[label].set_color('blue')
        if label != pred_idx:
            bars[pred_idx].set_color('red')
        
        plt.xlabel('Probability')
        plt.tight_layout()
    
    plt.tight_layout()
    plt.show()