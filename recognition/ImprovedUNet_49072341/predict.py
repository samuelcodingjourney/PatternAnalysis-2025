import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm

from dataset import HipMRIDataset, get_dataloaders
from modules import ImprovedUNet, dice_coefficient


def load_model(checkpoint_path, device):
    """
    Load trained model from checkpoint.
    """
    model = ImprovedUNet(in_channels=1, num_classes=6, base_features=32)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Validation Dice scores: {checkpoint['val_dice']}")
    
    return model


def evaluate_test_set(model, test_loader, device):
    """
    Evaluate model on test set.
    """
    model.eval()
    dice_scores_per_class = [[] for _ in range(6)]
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            # Forward pass
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            # Store for visualization
            all_predictions.append(predictions.cpu())
            all_targets.append(masks.cpu())
            # Calculate dice scores
            dice_scores = dice_coefficient(outputs, masks, num_classes=6)
            for i, score in enumerate(dice_scores):
                dice_scores_per_class[i].append(score)
            
            pbar.set_postfix({
                'dice_min': f'{min(dice_scores):.4f}'
            })
    
    # Calculate mean dice scores
    mean_dice_per_class = [np.mean(scores) for scores in dice_scores_per_class]
    std_dice_per_class = [np.std(scores) for scores in dice_scores_per_class]
    
    return mean_dice_per_class, std_dice_per_class, all_predictions, all_targets


def visualize_predictions(model, test_loader, device, num_samples=5, save_path='predictions.png'):
    """
    Visualize predictions on test samples.
    """
    model.eval()
    
    # Get some samples
    images_list = []
    masks_list = []
    preds_list = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            
            images_list.append(images.cpu())
            masks_list.append(masks.cpu())
            preds_list.append(predictions.cpu())
            
            if len(images_list) * images.shape[0] >= num_samples:
                break
    
    # Concatenate batches
    images_all = torch.cat(images_list, dim=0)[:num_samples]
    masks_all = torch.cat(masks_list, dim=0)[:num_samples]
    preds_all = torch.cat(preds_list, dim=0)[:num_samples]
    
    # Plot
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
    
    class_colors = {
        0: [0, 0, 0],        # Class 0 - Black
        1: [1, 0, 0],        # Class 1 - Red
        2: [0, 1, 0],        # Class 2 - Green
        3: [0, 1, 1],        # Class 3 - Cyan
        4: [1, 1, 0],        # Class 4 - Yellow
        5: [0, 0, 1],        # Class 5 - Blue
    }
    
    for i in range(num_samples):
        # Image
        img = images_all[i, 0].numpy()
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        mask = masks_all[i].numpy()
        mask_colored = np.zeros((*mask.shape, 3))
        for class_idx, color in class_colors.items():
            mask_colored[mask == class_idx] = color
        axes[i, 1].imshow(mask_colored)
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        pred = preds_all[i].numpy()
        pred_colored = np.zeros((*pred.shape, 3))
        for class_idx, color in class_colors.items():
            pred_colored[pred == class_idx] = color
        axes[i, 2].imshow(pred_colored)
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {save_path}")


def main():
    # Configuration
    data_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    checkpoint_path = "checkpoints/best_model.pth"
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading model")
    model = load_model(checkpoint_path, device)
    
    # Load test data
    print("\nLoading test dataset")
    _, _, test_loader = get_dataloaders(data_path, batch_size=8, num_workers=4)
    
    # Evaluate on test set
    print("\nEvaluating on test set")
    mean_dice, std_dice, all_preds, all_targets = evaluate_test_set(model, test_loader, device)
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
    for i, name in enumerate(class_names):
        print(f"{name:15s}: Dice = {mean_dice[i]:.4f} ± {std_dice[i]:.4f}")
    print("="*60)
    print(f"\n{'MIN DICE (ALL)':15s}: {min(mean_dice):.4f} (Requirement: ≥ 0.75)")
    
    if min(mean_dice) >= 0.75:
        print("PASSED - All 6 classes meet requirement!")
    else:
        print("FAILED - Some classes below requirement")
    print("="*60)
    
    # Visualize predictions
    print("\nCreating visualizations...")
    visualize_predictions(model, test_loader, device, num_samples=5, 
                         save_path='test_predictions.png')
    
    print("\nPrediction complete!")


if __name__ == "__main__":
    main()
