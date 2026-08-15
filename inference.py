"""
GeoGuessr Hackathon - Inference Pipeline
Generate predictions for test images and create submission CSV.
"""

import os
import csv
import math
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.amp import autocast

from dataset import create_test_dataloader, denormalize_coordinates
from model import GeoLocationModel


def load_model(checkpoint_dir, device):
    """Load the best model from checkpoints."""
    model = GeoLocationModel(hidden_dim=256, dropout=0.0)

    # Try loading in order of preference
    checkpoint_files = [
        'best_stage3_full.pt',
        'best_stage2_partial.pt',
        'best_stage1_heads.pt',
        'final_model.pt',
    ]

    for ckpt_name in checkpoint_files:
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
        if os.path.exists(ckpt_path):
            print(f"[Inference] Loading checkpoint: {ckpt_name}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'val_metrics' in checkpoint:
                print(f"[Inference] Checkpoint val metrics: {checkpoint['val_metrics']}")
            break
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate_predictions(model, test_loader, device, centroids=None):
    """Generate predictions for all test images."""
    predictions = []

    for batch_idx, (images, filenames) in enumerate(test_loader):
        images = images.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=False):
            coords, log_radius, logits = model(images)

        # Denormalize regressed coordinates
        pred_lat = coords[:, 0].cpu().numpy() * 90.0
        pred_lon = coords[:, 1].cpu().numpy() * 180.0

        if centroids is not None:
            # The exact configuration that scored 2.8!
            probs = torch.softmax(logits, dim=1)
            centroids_lat = torch.tensor(centroids[:, 0], device=probs.device, dtype=torch.float32) * 90.0
            centroids_lon = torch.tensor(centroids[:, 1], device=probs.device, dtype=torch.float32) * 180.0
            
            expected_lat = torch.sum(probs * centroids_lat, dim=1).cpu().numpy()
            expected_lon = torch.sum(probs * centroids_lon, dim=1).cpu().numpy()
            
            # 80/20 Ensemble
            pred_lat = 0.2 * pred_lat + 0.8 * expected_lat
            pred_lon = 0.2 * pred_lon + 0.8 * expected_lon

        # Convert log_radius to km
        pred_radius = np.exp(log_radius.cpu().numpy().squeeze(-1))

        # Clamp radius to reasonable range [5, 5000] km
        pred_radius = np.clip(pred_radius, 5.0, 5000.0)

        # Clamp coordinates to valid ranges
        pred_lat = np.clip(pred_lat, -90.0, 90.0)
        pred_lon = np.clip(pred_lon, -180.0, 180.0)

        for i in range(len(filenames)):
            predictions.append({
                'image_id': filenames[i],
                'pred_lat': float(pred_lat[i]),
                'pred_lon': float(pred_lon[i]),
                'pred_radius_km': float(pred_radius[i]),
            })

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
            print(f"  Processed {(batch_idx+1)*test_loader.batch_size}/{len(test_loader.dataset)} images")

    return predictions


def optimize_radius(predictions, default_radius=500.0, scale_factor=1.2):
    """
    Post-process radius predictions for better calibration.
    """
    for pred in predictions:
        pred['pred_radius_km'] = pred['pred_radius_km'] * scale_factor
        pred['pred_radius_km'] = max(pred['pred_radius_km'], 10.0)
        pred['pred_radius_km'] = min(pred['pred_radius_km'], 5000.0)

    return predictions


def save_submission(predictions, output_path, sample_path):
    """Save predictions as submission CSV in the exact order of sample_submission."""
    # Load expected order
    with open(sample_path, 'r') as f:
        reader = csv.DictReader(f)
        expected_ids = [row['image_id'] for row in reader]
        
    # Map predictions by image_id
    pred_map = {p['image_id']: p for p in predictions}
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id', 'pred_lat', 'pred_lon', 'pred_radius_km'])
        writer.writeheader()
        for img_id in expected_ids:
            if img_id in pred_map:
                writer.writerow(pred_map[img_id])
            else:
                # Fallback if missing
                writer.writerow({'image_id': img_id, 'pred_lat': 0.0, 'pred_lon': 0.0, 'pred_radius_km': 1000.0})

    print(f"\n[Inference] Saved {len(predictions)} predictions to {output_path}")

    # Print summary statistics
    lats = [p['pred_lat'] for p in predictions]
    lons = [p['pred_lon'] for p in predictions]
    radii = [p['pred_radius_km'] for p in predictions]

    print(f"  Lat range: [{min(lats):.2f}, {max(lats):.2f}]")
    print(f"  Lon range: [{min(lons):.2f}, {max(lons):.2f}]")
    print(f"  Radius range: [{min(radii):.1f}, {max(radii):.1f}] km")
    print(f"  Median radius: {np.median(radii):.1f} km")


def validate_submission(submission_path, sample_submission_path):
    """Validate that submission matches the expected format."""
    # Load sample submission
    with open(sample_submission_path, 'r') as f:
        reader = csv.DictReader(f)
        expected_ids = set(row['image_id'] for row in reader)

    # Load our submission
    with open(submission_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    submitted_ids = set(row['image_id'] for row in rows)

    # Check
    assert len(rows) == len(expected_ids), \
        f"Row count mismatch: {len(rows)} vs {len(expected_ids)}"

    assert submitted_ids == expected_ids, \
        f"Image ID mismatch. Missing: {expected_ids - submitted_ids}, Extra: {submitted_ids - expected_ids}"

    # Check columns
    expected_cols = {'image_id', 'pred_lat', 'pred_lon', 'pred_radius_km'}
    actual_cols = set(rows[0].keys())
    assert actual_cols == expected_cols, \
        f"Column mismatch: expected {expected_cols}, got {actual_cols}"

    # Check value ranges
    for row in rows:
        lat = float(row['pred_lat'])
        lon = float(row['pred_lon'])
        radius = float(row['pred_radius_km'])
        assert -90 <= lat <= 90, f"Invalid lat: {lat}"
        assert -180 <= lon <= 180, f"Invalid lon: {lon}"
        assert radius > 0, f"Invalid radius: {radius}"

    print(f"\n[Inference] [SUCCESS] Submission validation PASSED ({len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(description='GeoGuessr Inference')
    parser.add_argument('--data_root', type=str, default='geolocation-prediction')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--output', type=str, default='submission.csv')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--radius_scale', type=float, default=1.2,
                        help='Scale factor for radius (>1 = safer, <1 = riskier)')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Inference] Using device: {device}")

    # Load model
    model = load_model(args.checkpoint_dir, device)

    # Create test dataloader
    test_loader = create_test_dataloader(
        args.data_root,
        batch_size=4,
        num_workers=args.num_workers
    )

    # Load centroids if available
    centroids_path = os.path.join(args.data_root, 'centroids.npy')
    centroids = None
    if os.path.exists(centroids_path):
        print(f"[Inference] Loading classification centroids from {centroids_path}")
        centroids = np.load(centroids_path)

    # Generate predictions
    print("\n[Inference] Generating predictions...")
    predictions = generate_predictions(model, test_loader, device, centroids=centroids)

    # Optimize radius
    predictions = optimize_radius(predictions, scale_factor=args.radius_scale)

    # Validate paths
    sample_sub_path = os.path.join(args.data_root, 'sample_submission.csv')

    # Save submission
    save_submission(predictions, args.output, sample_sub_path)

    # Validate
    validate_submission(args.output, sample_sub_path)


if __name__ == '__main__':
    main()
