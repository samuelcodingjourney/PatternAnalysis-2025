import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset import HipMRIDataset, get_dataloaders
from modules import ImprovedUNet, DiceLoss, dice_coefficient


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    running_loss = 0.0
    dice_scores_per_class = [[] for _ in range(4)]
    
    pbar = tqdm(train_loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        
        # Calculate dice scores
        with torch.no_grad():
            dice_scores = dice_coefficient(outputs, masks, num_classes=4)
            for i, score in enumerate(dice_scores):
                dice_scores_per_class[i].append(score)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice_prostate': f'{dice_scores[3]:.4f}'
        })
    
    epoch_loss = running_loss / len(train_loader)
    avg_dice_per_class = [np.mean(scores) for scores in dice_scores_per_class]
    
    return epoch_loss, avg_dice_per_class


def validate(model, val_loader, criterion, device):
    """
    Validate the model.
    """
    model.eval()
    running_loss = 0.0
    dice_scores_per_class = [[] for _ in range(4)]
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Track metrics
            running_loss += loss.item()
            
            # Calculate dice scores
            dice_scores = dice_coefficient(outputs, masks, num_classes=4)
            for i, score in enumerate(dice_scores):
                dice_scores_per_class[i].append(score)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice_prostate': f'{dice_scores[3]:.4f}'
            })
    
    epoch_loss = running_loss / len(val_loader)
    avg_dice_per_class = [np.mean(scores) for scores in dice_scores_per_class]
    
    return epoch_loss, avg_dice_per_class


def plot_metrics(train_losses, val_losses, train_dice, val_dice, save_path='training_metrics.png'):
    """
    Plot training metrics.
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    axes[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axes[0].plot(epochs, val_losses, 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot Dice scores for prostate (class 3)
    train_dice_prostate = [dice[3] for dice in train_dice]
    val_dice_prostate = [dice[3] for dice in val_dice]
    
    axes[1].plot(epochs, train_dice_prostate, 'b-', label='Train Dice (Prostate)')
    axes[1].plot(epochs, val_dice_prostate, 'r-', label='Val Dice (Prostate)')
    axes[1].axhline(y=0.75, color='g', linestyle='--', label='Target (0.75)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Dice Score for Prostate (Class 3)')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Metrics plot saved to {save_path}")


def train_model(data_path, num_epochs=100, batch_size=8, learning_rate=1e-4, 
                save_dir='checkpoints'):
    """
    Main training function.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_path, 
        batch_size=batch_size,
        num_workers=4
    )
    
    # Initialize model
    print("Initializing model...")
    model = ImprovedUNet(in_channels=1, num_classes=4, base_features=32)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Training history
    train_losses = []
    val_losses = []
    train_dice_history = []
    val_dice_history = []
    
    best_val_dice = 0.0
    
    print(f"\nStarting training for {num_epochs} epochs...\n")
    
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_dice[3])  # Use prostate dice for scheduling
        
        # Save history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_dice_history.append(train_dice)
        val_dice_history.append(val_dice)
        
        # Print epoch summary
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Train Dice - Background: {train_dice[0]:.4f}, Class1: {train_dice[1]:.4f}, "
              f"Class2: {train_dice[2]:.4f}, Prostate: {train_dice[3]:.4f}")
        print(f"Val Dice - Background: {val_dice[0]:.4f}, Class1: {val_dice[1]:.4f}, "
              f"Class2: {val_dice[2]:.4f}, Prostate: {val_dice[3]:.4f}")
        
        # Save best model
        if val_dice[3] > best_val_dice:
            best_val_dice = val_dice[3]
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_loss': val_loss,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"✓ Saved best model with prostate Dice: {best_val_dice:.4f}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        print()
    
    # Plot final metrics
    plot_metrics(train_losses, val_losses, train_dice_history, val_dice_history,
                 save_path=os.path.join(save_dir, 'training_metrics.png'))
    
    print(f"\nTraining completed!")
    print(f"Best validation Prostate Dice: {best_val_dice:.4f}")
    
    return model, train_losses, val_losses, train_dice_history, val_dice_history


if __name__ == "__main__":
    # Configuration
    data_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    num_epochs = 100
    batch_size = 8
    learning_rate = 1e-4
    
    # Train the model
    model, train_losses, val_losses, train_dice, val_dice = train_model(
        data_path=data_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir='checkpoints'
    )
