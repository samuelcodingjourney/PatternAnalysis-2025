import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


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
        mask = mask_nifti.get_fdata().astype(np.int64)
        
        # Normalize image if requested
        if self.normalize:
            # Z-score normalization
            img = (img - img.mean()) / (img.std() + 1e-8)
        
        # Add channel dimension: (H, W) -> (1, H, W)
        img = np.expand_dims(img, axis=0)
        
        # Convert to torch tensors
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask).long()
        
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
