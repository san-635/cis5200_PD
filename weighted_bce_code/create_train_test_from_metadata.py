"""
Create Train/Test Split from Existing Metadata
Maintains consistency with original split used in Focal Loss experiments
"""

import os
import json
import shutil
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


def create_train_test_from_metadata(
    preprocessed_dir='preprocessed_images',
    train_metadata_path='train_metadata.csv',
    test_metadata_path='test_metadata.csv',
    output_dir='kfold_data_with_test',
    n_folds=10
):
    """
    Create train/test split with k-fold structure from existing metadata

    Args:
        preprocessed_dir: Directory with preprocessed images
        train_metadata_path: Path to train_metadata.csv
        test_metadata_path: Path to test_metadata.csv
        output_dir: Output directory for structured data
        n_folds: Number of folds for cross-validation
    """

    print("="*80)
    print("CREATING TRAIN/TEST SPLIT FROM EXISTING METADATA")
    print("="*80)

    # Load metadata
    print("\n1. Loading metadata files...")
    train_df = pd.read_csv(train_metadata_path)
    test_df = pd.read_csv(test_metadata_path)

    print(f"   Train metadata: {len(train_df)} subjects")
    print(f"   Test metadata: {len(test_df)} subjects")

    # Create output directories
    print("\n2. Creating directory structure...")
    os.makedirs(output_dir, exist_ok=True)

    # Create test set directory
    test_dir = Path(output_dir) / 'test_set'
    for label in ['PD', 'Control']:
        (test_dir / label).mkdir(parents=True, exist_ok=True)

    # Copy test set images
    print("\n3. Copying test set images...")
    test_stats = {'total': 0, 'pd': 0, 'control': 0}

    for _, row in test_df.iterrows():
        src = Path(preprocessed_dir) / row['filename']
        label = 'PD' if row['label'] == 1 else 'Control'
        dst = test_dir / label / row['filename']

        if src.exists():
            shutil.copy2(src, dst)
            test_stats['total'] += 1
            if label == 'PD':
                test_stats['pd'] += 1
            else:
                test_stats['control'] += 1

    print(f"   Copied {test_stats['total']} test images (PD: {test_stats['pd']}, Control: {test_stats['control']})")

    # Create k-fold splits for training data
    print(f"\n4. Creating {n_folds}-fold cross-validation splits...")

    X = train_df['filename'].values
    y = train_df['label'].values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_stats = {}

    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X, y)):
        print(f"   Processing fold {fold_idx}...")

        fold_dir = Path(output_dir) / f'fold_{fold_idx}'

        # Create train and val directories
        for split in ['train', 'val']:
            for label in ['PD', 'Control']:
                (fold_dir / split / label).mkdir(parents=True, exist_ok=True)

        # Copy training images
        train_stats = {'total': 0, 'pd': 0, 'control': 0}
        for idx in train_indices:
            row = train_df.iloc[idx]
            src = Path(preprocessed_dir) / row['filename']
            label = 'PD' if row['label'] == 1 else 'Control'
            dst = fold_dir / 'train' / label / row['filename']

            if src.exists():
                shutil.copy2(src, dst)
                train_stats['total'] += 1
                if label == 'PD':
                    train_stats['pd'] += 1
                else:
                    train_stats['control'] += 1

        # Copy validation images
        val_stats = {'total': 0, 'pd': 0, 'control': 0}
        for idx in val_indices:
            row = train_df.iloc[idx]
            src = Path(preprocessed_dir) / row['filename']
            label = 'PD' if row['label'] == 1 else 'Control'
            dst = fold_dir / 'val' / label / row['filename']

            if src.exists():
                shutil.copy2(src, dst)
                val_stats['total'] += 1
                if label == 'PD':
                    val_stats['pd'] += 1
                else:
                    val_stats['control'] += 1

        # Store fold statistics
        fold_stats[f'fold_{fold_idx}'] = {
            'train_total': train_stats['total'],
            'train_pd': train_stats['pd'],
            'train_control': train_stats['control'],
            'val_total': val_stats['total'],
            'val_pd': val_stats['pd'],
            'val_control': val_stats['control']
        }

    # Save split information
    print("\n5. Saving split information...")
    split_info = {
        'test_set': test_stats,
        'folds': fold_stats
    }

    with open(Path(output_dir) / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)

    print("\n" + "="*80)
    print("SPLIT SUMMARY")
    print("="*80)
    print(f"\nTest Set (HELD OUT):")
    print(f"  Total: {test_stats['total']}")
    print(f"  PD: {test_stats['pd']}")
    print(f"  Control: {test_stats['control']}")

    print(f"\nTraining Folds: {n_folds}")
    print(f"\nFirst fold example:")
    fold_0 = fold_stats['fold_0']
    print(f"  Train: {fold_0['train_total']} (PD: {fold_0['train_pd']}, Control: {fold_0['train_control']})")
    print(f"  Val: {fold_0['val_total']} (PD: {fold_0['val_pd']}, Control: {fold_0['val_control']})")

    print("\n" + "="*80)
    print("Train/test split created successfully!")
    print("="*80)


if __name__ == '__main__':
    create_train_test_from_metadata()
