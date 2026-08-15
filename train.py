"""
GeoGuessr Hackathon - Training Pipeline
3-stage training: head-only -> partial fine-tune -> full fine-tune
"""

import os
import sys
import time
import json
import random
import argparse
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

from dataset import create_dataloaders, denormalize_coordinates
from model import GeoLocationModel, GeoLocationLoss, haversine_distance


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"[Train] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Train] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("[Train] Using CPU (training will be slow)")
    return device


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch,
                    grad_accum_steps=4):
    """Train for one epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0
    total_haversine = 0
    total_dist = 0
    num_batches = 0
    optimizer.zero_grad()

    for batch_idx, (images, targets, target_classes) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_classes = target_classes.to(device, non_blocking=True)

        # Mixed precision forward pass (disabled for float32 stability)
        with autocast(device_type=device.type, enabled=False):
            coords, log_radius, logits = model(images)
            loss, loss_dict = criterion(coords, log_radius, logits, targets, target_classes)
            loss = loss / grad_accum_steps

        # Backward pass with gradient scaling
        if device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation step
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(loader):
            if device.type == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss_dict['total']
        total_haversine += loss_dict['haversine']
        total_dist += loss_dict['median_dist_km']
        num_batches += 1

        # Log progress
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            avg_loss = total_loss / num_batches
            avg_dist = total_dist / num_batches
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {avg_loss:.4f} | "
                  f"Median Dist: {avg_dist:.1f} km")

    return {
        'loss': total_loss / num_batches,
        'haversine': total_haversine / num_batches,
        'median_dist_km': total_dist / num_batches,
    }


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    all_distances = []
    total_loss = 0
    num_batches = 0

    for images, targets, target_classes in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_classes = target_classes.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=False):
            coords, log_radius, logits = model(images)
            loss, loss_dict = criterion(coords, log_radius, logits, targets, target_classes)

        # Calculate actual distances
        pred_lat = coords[:, 0] * 90.0
        pred_lon = coords[:, 1] * 180.0
        true_lat = targets[:, 0] * 90.0
        true_lon = targets[:, 1] * 180.0
        distances = haversine_distance(pred_lat, pred_lon, true_lat, true_lon)
        all_distances.extend(distances.cpu().tolist())

        total_loss += loss_dict['total']
        num_batches += 1

    all_distances = np.array(all_distances)
    return {
        'loss': total_loss / num_batches,
        'median_dist_km': np.median(all_distances),
        'mean_dist_km': np.mean(all_distances),
        'p25_dist_km': np.percentile(all_distances, 25),
        'p75_dist_km': np.percentile(all_distances, 75),
        'within_100km': np.mean(all_distances < 100) * 100,
        'within_500km': np.mean(all_distances < 500) * 100,
        'within_1000km': np.mean(all_distances < 1000) * 100,
    }


def train_stage(model, train_loader, val_loader, criterion, optimizer, scheduler,
                scaler, device, stage_name, num_epochs, checkpoint_dir,
                grad_accum_steps=4, patience=3):
    """Train for a given stage with early stopping."""
    print(f"\n{'='*60}")
    print(f"  STAGE: {stage_name}")
    print(f"  Epochs: {num_epochs}, Patience: {patience}")
    trainable, total = model.count_parameters()
    print(f"  Parameters: {trainable:,} trainable / {total:,} total")
    print(f"{'='*60}")

    best_val_dist = float('inf')
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, epoch, grad_accum_steps
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log
        print(f"\n  [{stage_name}] Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
        print(f"    Train Loss: {train_metrics['loss']:.4f} | "
              f"Train Median Dist: {train_metrics['median_dist_km']:.1f} km")
        print(f"    Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Median Dist: {val_metrics['median_dist_km']:.1f} km")
        print(f"    Val Mean Dist: {val_metrics['mean_dist_km']:.1f} km | "
              f"Within 100km: {val_metrics['within_100km']:.1f}% | "
              f"Within 500km: {val_metrics['within_500km']:.1f}%")

        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'time': epoch_time,
        })

        # Check for improvement
        if val_metrics['median_dist_km'] < best_val_dist:
            best_val_dist = val_metrics['median_dist_km']
            epochs_without_improvement = 0

            # Save best checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'best_{stage_name}.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'epoch': epoch,
                'stage': stage_name,
            }, checkpoint_path)
            print(f"    [Saved] Saved best checkpoint (median dist: {best_val_dist:.1f} km)")
        else:
            epochs_without_improvement += 1
            print(f"    No improvement for {epochs_without_improvement}/{patience} epochs")

        if epochs_without_improvement >= patience:
            print(f"    Early stopping triggered!")
            break

    # Load best checkpoint
    best_path = os.path.join(checkpoint_dir, f'best_{stage_name}.pt')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\n  Loaded best {stage_name} checkpoint (median dist: {best_val_dist:.1f} km)")

    return history, best_val_dist


def main():
    parser = argparse.ArgumentParser(description='GeoGuessr Model Training')
    parser.add_argument('--data_root', type=str, default='geolocation-prediction',
                        help='Path to extracted dataset')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (4 for ViT-Large on 6GB VRAM)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--grad_accum', type=int, default=16,
                        help='Gradient accumulation steps (effective batch = batch_size * grad_accum)')
    # Stage-specific
    parser.add_argument('--stage1_epochs', type=int, default=8,
                        help='Epochs for head-only training')
    parser.add_argument('--stage2_epochs', type=int, default=12,
                        help='Epochs for partial fine-tuning')
    parser.add_argument('--stage3_epochs', type=int, default=6,
                        help='Epochs for full fine-tuning')
    parser.add_argument('--stage1_lr', type=float, default=1e-3)
    parser.add_argument('--stage2_lr_backbone', type=float, default=1e-5)
    parser.add_argument('--stage2_lr_heads', type=float, default=5e-4)
    parser.add_argument('--stage3_lr', type=float, default=1e-6)
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = get_device()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Data
    print("\n[Train] Loading data...")
    train_loader, val_loader = create_dataloaders(
        args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=0.1,
        seed=args.seed
    )

    # Model
    print("\n[Train] Building model...")
    model = GeoLocationModel(hidden_dim=256, dropout=0.3)
    model = model.to(device)

    # Loss
    criterion = GeoLocationLoss(lambda_coord=1.0, lambda_radius=0.1, lambda_mse=0.5, lambda_cls=1.0)

    # Mixed precision scaler (disabled for float32 stability)
    scaler = GradScaler(device.type, enabled=False)

    all_history = {}

    # ===== STAGE 1: Head-only training =====
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.stage1_lr,
        weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage1_epochs, eta_min=1e-5
    )

    history, best_dist = train_stage(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        scaler, device, 'stage1_heads', args.stage1_epochs, args.checkpoint_dir,
        grad_accum_steps=args.grad_accum, patience=4
    )
    all_history['stage1'] = history

    # ===== STAGE 2: Unfreeze last 2 blocks =====
    print("\n[Train] Unfreezing last 2 backbone blocks...")
    model.unfreeze_backbone_last_n(n=2)

    param_groups = model.get_param_groups(
        lr_backbone=args.stage2_lr_backbone,
        lr_heads=args.stage2_lr_heads
    )
    optimizer = optim.AdamW(param_groups, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage2_epochs, eta_min=1e-6
    )

    history, best_dist = train_stage(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        scaler, device, 'stage2_partial', args.stage2_epochs, args.checkpoint_dir,
        grad_accum_steps=args.grad_accum, patience=4
    )
    all_history['stage2'] = history

    # ===== STAGE 3: Full fine-tune (only if stage 2 improved) =====
    print("\n[Train] Unfreezing full backbone...")
    model.unfreeze_all()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.stage3_lr,
        weight_decay=0.001
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage3_epochs, eta_min=1e-7
    )

    history, best_dist = train_stage(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        scaler, device, 'stage3_full', args.stage3_epochs, args.checkpoint_dir,
        grad_accum_steps=args.grad_accum, patience=3
    )
    all_history['stage3'] = history

    # Save final model
    final_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
    }, final_path)
    print(f"\n[Train] Saved final model to {final_path}")

    # Save training history
    history_path = os.path.join(args.checkpoint_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(all_history, f, indent=2)
    print(f"[Train] Saved training history to {history_path}")

    # Final validation
    print("\n" + "="*60)
    print("  FINAL VALIDATION RESULTS")
    print("="*60)
    val_metrics = validate(model, val_loader, criterion, device)
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
    print("="*60)


if __name__ == '__main__':
    main()
