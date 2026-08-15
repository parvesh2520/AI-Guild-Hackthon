
import os
import csv
import math
import random
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMG_SIZE = 336


def get_train_transforms():
    return transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def normalize_coordinates(lat, lon):
    """Normalize lat/lon to [-1, 1] range for better training."""
    lat_norm = lat / 90.0   # lat in [-90, 90] -> [-1, 1]
    lon_norm = lon / 180.0  # lon in [-180, 180] -> [-1, 1]
    return lat_norm, lon_norm


def denormalize_coordinates(lat_norm, lon_norm):
    """Convert normalized coordinates back to degrees."""
    lat = lat_norm * 90.0
    lon = lon_norm * 180.0
    return lat, lon


class GeoLocationDataset(Dataset):
    """Dataset for geotagged images with latitude/longitude labels."""

    def __init__(self, image_dir, csv_path, transform=None, indices=None, centroids=None):
        """
        Args:
            image_dir: Path to directory containing images
            csv_path: Path to ground_truth_coordinates.csv
            transform: torchvision transforms to apply
            indices: Optional list of indices to use (for train/val split)
        """
        self.image_dir = image_dir
        self.transform = transform or get_val_transforms()
        self.centroids = torch.tensor(centroids, dtype=torch.float32) if centroids is not None else None

        # Load CSV
        self.samples = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                image_id = row['image_id']
                lat = float(row['latitude'])
                lon = float(row['longitude'])
                # Training images are IMG_XXXXX.jpg, CSV has IMG_XXXXX (no .jpg)
                img_filename = image_id + '.jpg' if not image_id.endswith('.jpg') else image_id
                img_path = os.path.join(image_dir, img_filename)
                if os.path.exists(img_path):
                    self.samples.append((img_path, lat, lon))

        # Apply subset indices if provided
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        print(f"[Dataset] Loaded {len(self.samples)} samples from {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lat, lon = self.samples[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[Dataset] Error loading {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        # Apply transforms
        image = self.transform(image)

        # Normalize coordinates
        lat_norm, lon_norm = normalize_coordinates(lat, lon)

        target = torch.tensor([lat_norm, lon_norm], dtype=torch.float32)

        if self.centroids is not None:
            # Find closest centroid using L2 distance on normalized coordinates
            dists = torch.sum((self.centroids - target)**2, dim=1)
            class_idx = torch.argmin(dists)
            return image, target, class_idx
        return image, target


class TestDataset(Dataset):
    """Dataset for test images (no labels)."""

    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform or get_val_transforms()

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        print(f"[TestDataset] Found {len(self.image_files)} test images")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, filename)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"[TestDataset] Error loading {img_path}: {e}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))

        image = self.transform(image)
        return image, filename


def create_train_val_split(csv_path, val_ratio=0.1, seed=42):
    """Create train/val split indices with geographic stratification."""
    random.seed(seed)
    np.random.seed(seed)

    # Load all coordinates
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            lat = float(row['latitude'])
            lon = float(row['longitude'])
            samples.append((i, lat, lon))

    # Stratify by latitude bands (rough geographic stratification)
    # Divide into 18 latitude bands of 10 degrees each
    bands = {}
    for idx, lat, lon in samples:
        band = int((lat + 90) / 10)
        band = min(band, 17)  # Clamp to 0-17
        if band not in bands:
            bands[band] = []
        bands[band].append(idx)

    train_indices = []
    val_indices = []

    for band, indices in bands.items():
        random.shuffle(indices)
        n_val = max(1, int(len(indices) * val_ratio))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])

    random.shuffle(train_indices)
    random.shuffle(val_indices)

    print(f"[Split] Train: {len(train_indices)}, Val: {len(val_indices)}")
    return train_indices, val_indices


def create_dataloaders(data_root, batch_size=16, num_workers=4, val_ratio=0.1, seed=42):
    """Create train and validation dataloaders."""
    image_dir = os.path.join(data_root, 'training_dataset', 'noised_dataset', 'images')
    csv_path = os.path.join(data_root, 'training_dataset', 'noised_dataset', 'ground_truth_coordinates.csv')

    train_indices, val_indices = create_train_val_split(csv_path, val_ratio, seed)

    # Compute KMeans centroids
    print(f"[Dataset] Computing KMeans clusters on {len(train_indices)} training samples...")
    train_coords = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i in train_indices:
                lat, lon = normalize_coordinates(float(row['latitude']), float(row['longitude']))
                train_coords.append([lat, lon])
    train_coords = np.array(train_coords)
    
    num_clusters = min(200, len(train_coords))
    kmeans = KMeans(n_clusters=num_clusters, random_state=seed, n_init='auto').fit(train_coords)
    centroids = kmeans.cluster_centers_
    
    centroids_path = os.path.join(data_root, 'centroids.npy')
    np.save(centroids_path, centroids)
    print(f"[Dataset] Saved {num_clusters} cluster centroids to {centroids_path}")

    train_dataset = GeoLocationDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        transform=get_train_transforms(),
        indices=train_indices,
        centroids=centroids
    )

    val_dataset = GeoLocationDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        transform=get_val_transforms(),
        indices=val_indices,
        centroids=centroids
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader


def create_test_dataloader(data_root, batch_size=32, num_workers=4):
    """Create test dataloader."""
    test_dir = os.path.join(data_root, 'test_images_sampled')

    test_dataset = TestDataset(
        image_dir=test_dir,
        transform=get_val_transforms()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return test_loader


if __name__ == '__main__':
    # Quick test
    DATA_ROOT = 'geolocation-prediction'
    train_loader, val_loader = create_dataloaders(DATA_ROOT, batch_size=4, num_workers=0)
    print(f"\nTrain batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Test a batch
    images, targets, class_idx = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Target batch shape: {targets.shape}")
    print(f"Class idx batch shape: {class_idx.shape}")
    print(f"Target sample (normalized): {targets[0]}")
    lat, lon = denormalize_coordinates(targets[0][0].item(), targets[0][1].item())
    print(f"Target sample (degrees): lat={lat:.4f}, lon={lon:.4f}")
    print(f"Class index sample: {class_idx[0].item()}")
