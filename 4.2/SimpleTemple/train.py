import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from data import prepare_batch


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Create progress bar
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Prepare data
        inputs, targets = prepare_batch(batch, device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss / total,
            'acc': 100. * correct / total
        })
    
    # Calculate final statistics
    epoch_loss = total_loss / total
    accuracy = 100. * correct / total
    
    print(f'Training Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}%')
    
    return epoch_loss, accuracy

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
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
    
    # Calculate final statistics
    val_loss = total_loss / total
    accuracy = 100. * correct / total
    
    print(f'Validation Loss: {val_loss:.4f} | Accuracy: {accuracy:.2f}%')
    
    return val_loss, accuracy

def train_model(model, train_loader, val_loader=None, num_epochs=10, lr=0.001, 
               device=None, save_path=None):
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    # Create save directory if needed
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Train one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Validate
        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Update learning rate scheduler
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    save_checkpoint(model, optimizer, epoch, val_loss, train_acc,
                                  f"{save_path}_best.pt")
                    print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        if save_path:
            if epoch == num_epochs - 1:
                # Save final model
                save_checkpoint(model, optimizer, epoch, val_loss, train_acc,
                              f"{save_path}_final.pt")
            else:
                # Save epoch checkpoint
                save_checkpoint(model, optimizer, epoch, val_loss, train_acc,
                              f"{save_path}_epoch{epoch+1}.pt")
    
    return model, history


def save_checkpoint(model, optimizer, epoch, val_loss, train_acc, path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'train_acc': train_acc
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(model, path, optimizer=None, device=None):
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Get epoch and validation loss
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', float('inf'))
    
    print(f"Loaded checkpoint from {path}")
    print(f"Epoch: {epoch}, Validation Loss: {val_loss:.4f}")
    
    return model, epoch, val_loss