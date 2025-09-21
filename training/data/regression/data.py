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
from statistics import mean, stdev
import random


class CellDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.transform = transform  # Accept a transform parameter

        # Read image paths from the .txt file
        with open(txt_file, 'r') as f:
            self.image_paths = f.read().splitlines()

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


def custom_collate_fn_training(batch):
    """
    Custom collate function to downsample zero-cell patches dynamically based on batch distribution.

    Args:
        batch (list): List of items returned by Dataset's __getitem__ method.

    Returns:
        tuple: Balanced batch containing images, num_cells, density_maps, and pixel_positions.
    """
    # Separate patches with and without cells
    zero_cell_patches = [item for item in batch if item[1] == 0]  # Zero-cell patches (item[1] is num_cells)
    non_zero_cell_patches = [item for item in batch if item[1] > 0]  # Patches with cells

    # Calculate the dynamic ratio: retain enough zero-cell patches to match the distribution
    if len(non_zero_cell_patches) > 0:
        zero_cell_ratio = len(zero_cell_patches) / (len(zero_cell_patches) + len(non_zero_cell_patches))
        num_zero_to_keep = int(len(non_zero_cell_patches) * zero_cell_ratio)
    else:
        num_zero_to_keep = len(zero_cell_patches)  # If no non-zero patches, keep all zero patches

    # Downsample zero-cell patches
    zero_cell_patches = random.sample(zero_cell_patches, min(len(zero_cell_patches), num_zero_to_keep))

    # Combine the filtered patches
    balanced_batch = zero_cell_patches + non_zero_cell_patches

    # Handle empty batch
    if len(balanced_batch) == 0:
        return None

    # Unpack the balanced batch
    images, num_cells_list, density_maps, pixel_positions = zip(*balanced_batch)
    
    # Stack images
    images = torch.stack(images, dim=0)
    density_maps = torch.stack(density_maps, dim=0)
    pixel_positions = torch.stack(pixel_positions, dim=0)

    # Convert num_cells_list to tensor
    num_cells = torch.tensor(num_cells_list, dtype=torch.float32)

    return images, num_cells, density_maps, pixel_positions


import numpy as np
from torch.utils.data import DataLoader
from statistics import mean, stdev

def make_data_loaders(train_txt_file, val_txt_file, batch_size=32, num_workers=4,
                      train_transform=None, val_transform=None, test_transform=None):
    """
    Create data loaders for training and validation sets using provided txt files.

    Args:
        train_txt_file (str): Path to the txt file containing training image paths.
        val_txt_file (str): Path to the txt file containing validation image paths.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loading.
        train_transform: Transformations to apply to the training data.
        val_transform: Transformations to apply to the validation data.
        test_transform: Transformations to apply to the test data.

    Returns:
        train_loader, val_loader, test_loader: Data loaders for the specified fold.
    """
    # Create datasets
    train_dataset = CellDataset(train_txt_file, transform=train_transform)
    val_dataset = CellDataset(val_txt_file, transform=val_transform)
    test_dataset = CellDataset(val_txt_file, transform=test_transform)  # Using val set as test set in this example

    # Compute statistics for the splits
    # Collect cell counts from datasets
    train_counts = []
    for idx in range(len(train_dataset)):
        _, num_cells, _, _ = train_dataset[idx]
        train_counts.append(num_cells)
    val_counts = []
    for idx in range(len(val_dataset)):
        _, num_cells, _, _ = val_dataset[idx]
        val_counts.append(num_cells)

    # print(f"Training set: mean={mean(train_counts)}, std={stdev(train_counts)}")
    # print(f"Validation set: mean={mean(val_counts)}, std={stdev(val_counts)}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=custom_collate_fn_training)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=custom_collate_fn)

    return train_loader, val_loader, test_loader

# Define the transforms for training and validation

def train_transform(sample):
    """
    Applies data augmentation to the input sample consisting of an image, density map, and pixel position image.

    Args:
        sample (tuple): A tuple containing img, density_map_img, and pixel_pos_img.

    Returns:
        tuple: Transformed img, density_map_img, and pixel_pos_img.
    """
    img, density_map_img, pixel_pos_img = sample

    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(200, 200))
    img = F.crop(img, i, j, h, w)
    density_map_img = F.crop(density_map_img, i, j, h, w)
    pixel_pos_img = F.crop(pixel_pos_img, i, j, h, w)

    # Random horizontal flip
    if random.random() > 0.5:
        img = F.hflip(img)
        density_map_img = F.hflip(density_map_img)
        pixel_pos_img = F.hflip(pixel_pos_img)

    # Random vertical flip
    if random.random() > 0.5:
        img = F.vflip(img)
        density_map_img = F.vflip(density_map_img)
        pixel_pos_img = F.vflip(pixel_pos_img)

    # Random rotation (0, 90, 180, or 270 degrees)
    angles = [0, 90, 180, 270]
    angle = random.choice(angles)
    img = F.rotate(img, angle)
    density_map_img = F.rotate(density_map_img, angle)
    pixel_pos_img = F.rotate(pixel_pos_img, angle)

    # Color jitter (for image only)
    color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    img = color_jitter(img)

    # # Resize images to desired dimensions
    # img = F.resize(img, (256, 256))
    # output_size = 256 // 8
    # density_map_img = F.resize(density_map_img, (output_size, output_size), interpolation=Image.BILINEAR)
    # pixel_pos_img = F.resize(pixel_pos_img, (256, 256), interpolation=Image.NEAREST)

    # Convert images to tensors
    img = F.to_tensor(img)
    density_map_img = F.to_tensor(density_map_img)
    pixel_pos_img = F.to_tensor(pixel_pos_img)

    return img, density_map_img, pixel_pos_img

def val_transform(sample):
    img, density_map_img, pixel_pos_img = sample

    # Center crop
    img = F.center_crop(img, output_size=(200, 200))
    density_map_img = F.center_crop(density_map_img, output_size=(200, 200))
    pixel_pos_img = F.center_crop(pixel_pos_img, output_size=(200, 200))

    # Resize images
    # img = F.resize(img, (256, 256))
    output_size = 256 // 8
    # density_map_img = F.resize(density_map_img, (output_size, output_size), interpolation=Image.BILINEAR)
    # pixel_pos_img = F.resize(pixel_pos_img, (256, 256), interpolation=Image.NEAREST)

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

def cutmix(batch1, batch2, mix_prob=0.5, ratio=2):
    """
    Applies CutMix augmentation to a probabilistic subset of two batches using cutout masks.

    Args:
        batch1 (tuple): A batch containing (images1, num_cells1, density_maps1, pixel_pos_maps1).
        batch2 (tuple): A second batch containing (images2, num_cells2, density_maps2, pixel_pos_maps2).
        mix_prob (float): Probability of applying CutMix to each sample in the batch.
        ratio (float): Ratio for determining the CutMix region size.

    Returns:
        tuple: Augmented batch with CutMix applied to a subset of samples.
    """
    images1, num_cells1, density_maps1, pixel_pos_maps1 = batch1
    images2, num_cells2, density_maps2, pixel_pos_maps2 = batch2

    # Ensure both batches are of the same size
    batch_size = min(images1.size(0), images2.size(0))
    images1, density_maps1, pixel_pos_maps1, num_cells1 = (
        images1[:batch_size],
        density_maps1[:batch_size],
        pixel_pos_maps1[:batch_size],
        num_cells1[:batch_size],
    )
    images2, density_maps2, pixel_pos_maps2, num_cells2 = (
        images2[:batch_size],
        density_maps2[:batch_size],
        pixel_pos_maps2[:batch_size],
        num_cells2[:batch_size],
    )

    indices = torch.arange(batch_size)  # Indices for mixing

    # Apply CutMix to a probabilistic subset of the batch
    for i in range(batch_size):
        if random.random() < mix_prob:
            # Generate a cutout mask
            mask = generate_cutout_mask(images1[i].shape[1:], ratio)

            # Apply mask to mix the two images and their corresponding maps
            images1[i] = images1[i] * mask + images2[indices[i]] * (1 - mask)
            density_maps1[i] = density_maps1[i] * mask + density_maps2[indices[i]] * (1 - mask)
            pixel_pos_maps1[i] = pixel_pos_maps1[i] * mask + pixel_pos_maps2[indices[i]] * (1 - mask)

            # Update num_cells based on the mixed pixel position maps
            pixel_pos_array = pixel_pos_maps1[i].squeeze().cpu().numpy()
            binary_image = (pixel_pos_array > 0.5).astype(np.int32)
            labeled_array, num_features = scipy.ndimage.label(binary_image)
            num_cells1[i] = num_features

    return images1, num_cells1, density_maps1, pixel_pos_maps1


def generate_cutout_mask(img_size, ratio=2):
    """
    Generates a random cutout mask for an image.

    Args:
        img_size (tuple): Tuple containing the height and width of the image.
        ratio (float): Ratio to control the size of the cutout region.

    Returns:
        torch.Tensor: A binary mask for CutMix.
    """
    cutout_area = img_size[0] * img_size[1] / ratio

    w = np.random.randint(img_size[1] // ratio + 1, img_size[1])
    h = int(np.round(cutout_area / w))

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.float()