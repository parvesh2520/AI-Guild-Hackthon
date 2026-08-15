"""
GeoGuessr Hackathon - Run Everything
Single script to train the model and generate submission.
Usage: python run.py
"""

import os
import sys
import time


def main():
    print("=" * 70)
    print("  GEOGUESSR HACKATHON - GEOLOCATION PREDICTION MODEL")
    print("  Training + Inference Pipeline")
    print("=" * 70)

    start_time = time.time()

    # Check data exists
    data_root = 'geolocation-prediction'
    assert os.path.exists(data_root), f"Data directory not found: {data_root}"
    assert os.path.exists(os.path.join(data_root, 'training_dataset', 'noised_dataset', 'images')), \
        "Training images not found"
    assert os.path.exists(os.path.join(data_root, 'test_images_sampled')), \
        "Test images not found"
    print("\n✓ Data directories verified\n")

    # Check CUDA
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    else:
        print("⚠ No CUDA GPU detected. Training will be slow on CPU.")

    # Step 1: Train
    print("\n" + "=" * 70)
    print("  STEP 1: TRAINING MODEL")
    print("=" * 70)

    train_cmd = (
        f'python train.py '
        f'--data_root {data_root} '
        f'--batch_size 16 '
        f'--num_workers 4 '
        f'--grad_accum 4 '
        f'--stage1_epochs 8 '
        f'--stage2_epochs 12 '
        f'--stage3_epochs 6 '
        f'--checkpoint_dir checkpoints '
        f'--seed 42'
    )
    print(f"\nRunning: {train_cmd}\n")
    ret = os.system(train_cmd)
    if ret != 0:
        print(f"\n✗ Training failed with return code {ret}")
        sys.exit(1)

    train_time = time.time() - start_time
    print(f"\n✓ Training completed in {train_time/60:.1f} minutes")

    # Step 2: Generate submission
    print("\n" + "=" * 70)
    print("  STEP 2: GENERATING SUBMISSION")
    print("=" * 70)

    inference_cmd = (
        f'python inference.py '
        f'--data_root {data_root} '
        f'--checkpoint_dir checkpoints '
        f'--output submission.csv '
        f'--batch_size 32 '
        f'--num_workers 4 '
        f'--radius_scale 1.2'
    )
    print(f"\nRunning: {inference_cmd}\n")
    ret = os.system(inference_cmd)
    if ret != 0:
        print(f"\n✗ Inference failed with return code {ret}")
        sys.exit(1)

    total_time = time.time() - start_time

    print("\n" + "=" * 70)
    print("  ✓ ALL DONE!")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Submission file: submission.csv")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Upload submission.csv to the Kaggle competition")
    print("  2. Check your leaderboard score")
    print("  3. Iterate and improve!")


if __name__ == '__main__':
    main()
