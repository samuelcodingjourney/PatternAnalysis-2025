import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class HipMRIDataset(Dataset):
    """
    Dataset loader for HipMRI 2D prostate segmentation.
    """
    
    def __init__(self, data_path, split='train', transform=None, normalize=True):
        """
        Args:
            data_path: Base path to keras_slices_data folder
            split: 'train', 'val', or 'test'
            transform: Optional data augmentation transforms
            normalize: Whether to normalize images to [0, 1]
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform
        self.normalize = normalize
        
        # Map split names to folder names
        split_map = {
            'train': 'keras_slices_train',
            'val': 'keras_slices_validate',
            'test': 'keras_slices_test'
        }
        
        split_map_seg = {
            'train': 'keras_slices_seg_train',
            'val': 'keras_slices_seg_validate',
            'test': 'keras_slices_seg_test'
        }
        
        self.img_dir = os.path.join(data_path, split_map[split])
        self.mask_dir = os.path.join(data_path, split_map_seg[split])
        
        # Get all image files
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, '*.nii.gz')))
        
        print(f"Loaded {len(self.img_files)} samples for {split} split")
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.img_files[idx]
        img_nifti = nib.load(img_path)
        img = img_nifti.get_fdata().astype(np.float32)
    
        # Get corresponding mask filename
        img_filename = os.path.basename(img_path)
        mask_filename = img_filename.replace('case_', 'seg_')
        mask_path = os.path.join(self.mask_dir, mask_filename)
    
        # Load mask
        mask_nifti = nib.load(mask_path)
        mask = mask_nifti.get_fdata()
    
        # Round to nearest integer and clip to valid range
        mask = np.round(mask).astype(np.int64)
        mask[mask > 3] = 0  # 6 classes: 0-5
    
        # Normalize image if requested
        if self.normalize:
            # Z-score normalization
            img = (img - img.mean()) / (img.std() + 1e-8)
    
        # Add channel dimension: (H, W) -> (1, H, W)
        img = np.expand_dims(img, axis=0)
    
        # Convert to torch tensors
        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()
    
        # Resize to fixed size (256, 128) if needed
        target_size = (256, 128)
        if img.shape[1:] != target_size:
            img = F.interpolate(img.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=target_size, mode='nearest').squeeze(0).squeeze(0).long()
    
        # Apply transforms if any
        if self.transform:
            img, mask = self.transform(img, mask)
    
        return img, mask


def get_dataloaders(data_path, batch_size=8, num_workers=4):
    """
    Create train, validation, and test dataloaders.
    """
    train_dataset = HipMRIDataset(data_path, split='train')
    val_dataset = HipMRIDataset(data_path, split='val')
    test_dataset = HipMRIDataset(data_path, split='test')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    data_path = "/home/groups/comp3710/HipMRI_Study_open/keras_slices_data"
    
    dataset = HipMRIDataset(data_path, split='train')
    print(f"Dataset size: {len(dataset)}")
    
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}, dtype: {img.dtype}")
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Unique labels: {torch.unique(mask)}")
