import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import pandas as pd
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import scipy.ndimage
import statistics

class CellDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform  # Accept a transform parameter

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get the image path
        img_path = self.image_paths[idx]

        # Load the image
        img = Image.open(img_path).convert('RGB')

        # Construct paths for density map and pixel position images
        density_map_path = img_path.replace('images', 'ground_truth_2/density_maps').replace('.tiff', '_density_map.png')
        pixel_pos_path = img_path.replace('images', 'ground_truth_2/pixel_positions').replace('.tiff', '_pixel_positions.png')

        # Load density map and pixel position images
        density_map_img = Image.open(density_map_path).convert('L')
        pixel_pos_img = Image.open(pixel_pos_path).convert('L')

        # Apply the same transform to all images if provided
        if self.transform:
            # Combine images into a single tuple to apply the same transform
            img, density_map_img, pixel_pos_img = self.transform((img, density_map_img, pixel_pos_img))
        else:
            # Convert images to tensors if no transform is provided
            img = F.to_tensor(img)
            density_map_img = F.to_tensor(density_map_img)
            pixel_pos_img = F.to_tensor(pixel_pos_img)

        # Normalize the original image
        img = F.normalize(img, mean=[0.0414, 0.0414, 0.0414], std=[0.0203, 0.0203, 0.0203])

        # Count the number of cells using connected components
        pixel_pos_array = pixel_pos_img.squeeze().numpy()
        binary_image = (pixel_pos_array > 0.5).astype(np.int32)
        labeled_array, num_features = scipy.ndimage.label(binary_image)
        num_cells = num_features

        return img, num_cells, density_map_img, pixel_pos_img

def custom_collate_fn(batch):
    images, num_cells_list, density_maps, pixel_positions = zip(*batch)
    
    # Stack images
    images = torch.stack(images, dim=0)
    density_maps = torch.stack(density_maps, dim=0)
    pixel_positions = torch.stack(pixel_positions, dim=0)

    # Convert num_cells_list to tensor
    num_cells = torch.tensor(num_cells_list, dtype=torch.float32)

    return images, num_cells, density_maps, pixel_positions

def make_data_loaders(image_paths, cell_counts, batch_size=32, num_workers=4, n_splits=5,
                      train_transform=None, val_transform=None, test_transform=None):
    '''
    Create data loaders for training and validation sets using cross-validation.

    Args:
        image_paths (np.array): Numpy array of paths to the images.
        cell_counts (np.array): Numpy array of cell counts corresponding to the images.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loading.
        n_splits (int): Number of splits for cross-validation.
        train_transform: Transformations to apply to the training data.
        val_transform: Transformations to apply to the validation data.
        test_transform: Transformations to apply to the test data.

    Returns:
        loaders (list): A list of data loaders (train, val and test) for each fold.
    '''
    from sklearn.model_selection import StratifiedKFold

    # Initialize StratifiedKFold with the provided number of splits
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    loaders = []
    
    # Perform cross-validation
    for fold, (train_index, val_index) in enumerate(skf.split(image_paths, cell_counts)):
        print(f"Fold {fold + 1}")
        
        # Get train and validation splits
        train_images, val_images = image_paths[train_index], image_paths[val_index]
        train_counts, val_counts = cell_counts[train_index], cell_counts[val_index]
        
        mean = statistics.mean(train_counts)
        std = statistics.stdev(train_counts)
        print(f"training part of fold {fold + 1} mean :", mean)
        print(f"training part of fold {fold + 1} std :", std)
        
        mean = statistics.mean(val_counts)
        std = statistics.stdev(val_counts)
        print(f"validation part of fold {fold + 1} mean :", mean)
        print(f"validation part of fold std {fold + 1}:", std)
        
        # Create datasets with respective transforms
        train_dataset = CellDataset(train_images, transform=train_transform)
        val_dataset = CellDataset(val_images, transform=val_transform)
        test_dataset = CellDataset(val_images, transform=test_transform)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, collate_fn=custom_collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, collate_fn=custom_collate_fn)

        # Append loaders for this fold
        loaders.append((train_loader, val_loader, test_loader))
    
    return loaders

# Define the transforms for training and validation
def train_transform(sample):
    img, density_map_img, pixel_pos_img = sample

    # Random crop parameters
    i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(224, 224))
    img = F.crop(img, i, j, h, w)
    density_map_img = F.crop(density_map_img, i, j, h, w)
    pixel_pos_img = F.crop(pixel_pos_img, i, j, h, w)

    # Resize images
    img = F.resize(img, (256, 256))
    # Resize the density map to match the model output size (downsample by factor of 8)
    output_size = 256 // 8
    density_map_img = F.resize(density_map_img, (output_size, output_size), interpolation=Image.BILINEAR)
    pixel_pos_img = F.resize(pixel_pos_img, (256, 256), interpolation=Image.NEAREST)

    # Convert images to tensors
    img = F.to_tensor(img)
    density_map_img = F.to_tensor(density_map_img)
    pixel_pos_img = F.to_tensor(pixel_pos_img)

    return img, density_map_img, pixel_pos_img

def val_transform(sample):
    img, density_map_img, pixel_pos_img = sample

    # Center crop
    img = F.center_crop(img, output_size=(224, 224))
    density_map_img = F.center_crop(density_map_img, output_size=(224, 224))
    pixel_pos_img = F.center_crop(pixel_pos_img, output_size=(224, 224))

    # Resize images
    img = F.resize(img, (256, 256))
    output_size = 256 // 8
    density_map_img = F.resize(density_map_img, (output_size, output_size), interpolation=Image.BILINEAR)
    pixel_pos_img = F.resize(pixel_pos_img, (256, 256), interpolation=Image.NEAREST)

    # Convert images to tensors
    img = F.to_tensor(img)
    density_map_img = F.to_tensor(density_map_img)
    pixel_pos_img = F.to_tensor(pixel_pos_img)

    return img, density_map_img, pixel_pos_img

def test_transform(sample):
    img, density_map_img, pixel_pos_img = sample

    # # Center crop
    # img = F.center_crop(img, output_size=(224, 224))
    # density_map_img = F.center_crop(density_map_img, output_size=(224, 224))
    # pixel_pos_img = F.center_crop(pixel_pos_img, output_size=(224, 224))

    # Resize images
    # img = F.resize(img, (256, 256))
    output_size = 256 // 8
    density_map_img = F.resize(density_map_img, (output_size, output_size), interpolation=Image.BILINEAR)
    # pixel_pos_img = F.resize(pixel_pos_img, (256, 256), interpolation=Image.NEAREST)

    # Convert images to tensors
    img = F.to_tensor(img)
    density_map_img = F.to_tensor(density_map_img)
    pixel_pos_img = F.to_tensor(pixel_pos_img)

    return img, density_map_img, pixel_pos_img