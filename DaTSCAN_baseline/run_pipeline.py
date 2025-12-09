"""
Complete pipeline for DaTSCAN classification with Weighted BCE loss
"""

import os
import sys
import json
import argparse
import subprocess
import pandas as pd
from pathlib import Path


def run_command(cmd, description):
    """Run a shell command"""
    print(f"\n{description}...")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error running: {description}")
        return False
    return True


def create_data_split(base_dir):
    """Create train/test split from metadata"""
    print("\n" + "="*80)
    print("STEP 1: Creating train/test split")
    print("="*80)

    os.chdir(base_dir.parent)

    cmd = [sys.executable, str(base_dir / 'create_train_test_from_metadata.py')]

    if not run_command(cmd, "Creating train/test split"):
        return False

    print("\nTrain/test split created")
    return True


def train_model(base_dir, model_name, output_dir, device='cuda'):
    """Train a single model"""
    print(f"\nTraining {model_name}...")

    kfold_data_dir = base_dir.parent / 'kfold_data_with_test'

    cmd = [
        sys.executable,
        str(base_dir / 'train_kfold_weighted_bce.py'),
        '--model', model_name,
        '--kfold_data_dir', str(kfold_data_dir),
        '--output_dir', str(output_dir),
        '--n_folds', '10',
        '--patience', '50',
        '--device', device
    ]

    if not run_command(cmd, f"Training {model_name}"):
        return False

    print(f"{model_name} training complete")
    return True


def find_best_checkpoint(output_dir):
    """Find best model checkpoint based on validation accuracy"""
    summary_path = output_dir / 'training_summary.csv'

    if not summary_path.exists():
        print(f"Training summary not found: {summary_path}")
        return None

    df = pd.read_csv(summary_path)
    best_idx = df['best_val_acc'].idxmax()
    best_fold = df.loc[best_idx]

    fold_num = int(best_fold['fold'])
    checkpoint_path = output_dir / f'model_fold_{fold_num}_best.pth'

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return None

    return {
        'fold': fold_num,
        'path': str(checkpoint_path),
        'val_acc': float(best_fold['best_val_acc']),
        'val_loss': float(best_fold['best_val_loss'])
    }


def evaluate_model(base_dir, model_name, checkpoint_path, device='cuda'):
    """Evaluate model on test set"""
    print(f"\nEvaluating {model_name} on test set...")

    test_data_dir = base_dir.parent / 'kfold_data_with_test' / 'test_set'
    output_dir = base_dir / f'{model_name}_test_results'

    cmd = [
        sys.executable,
        str(base_dir / 'evaluate_on_test_set.py'),
        '--test_data_dir', str(test_data_dir),
        '--best_model_path', checkpoint_path,
        '--output_dir', str(output_dir),
        '--device', device
    ]

    if not run_command(cmd, f"Evaluating {model_name}"):
        return False

    results_path = output_dir / 'test_set_results.json'
    if results_path.exists():
        with open(results_path, 'r') as f:
            results = json.load(f)

        print(f"\n{model_name} Test Results:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Sensitivity: {results['sensitivity']:.4f}")
        print(f"  Specificity: {results['specificity']:.4f}")
        print(f"  ROC AUC: {results['roc_auc']:.4f}")

    return True


def compare_models(base_dir, model_checkpoints, device='cuda'):
    """Compare all trained models"""
    print("\n" + "="*80)
    print("Comparing all models")
    print("="*80)

    test_data_dir = base_dir.parent / 'kfold_data_with_test' / 'test_set'
    output_dir = base_dir / 'model_comparison'

    cmd = [
        sys.executable,
        str(base_dir / 'compare_all_models.py'),
        '--test_data_dir', str(test_data_dir),
        '--output_dir', str(output_dir),
        '--device', device
    ]

    for model_name, checkpoint in model_checkpoints.items():
        model_key = model_name.lower().replace('-', '')
        cmd.extend([f'--{model_key}_path', checkpoint])

    if not run_command(cmd, "Comparing models"):
        return False

    print(f"\nComparison results saved to: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description='DaTSCAN Weighted BCE Pipeline')

    parser.add_argument('--mode', type=str, default='full',
                       choices=['split', 'train', 'eval', 'compare', 'full'],
                       help='Pipeline mode')

    parser.add_argument('--models', type=str, nargs='+',
                       default=['inceptionv3', 'resnet18', 'resnet34', 'resnet50'],
                       help='Models to train')

    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])

    args = parser.parse_args()

    base_dir = Path(__file__).parent

    print("="*80)
    print("DaTSCAN Classification - Weighted BCE Loss")
    print("="*80)

    if args.mode in ['split', 'full']:
        if not create_data_split(base_dir):
            print("Failed to create data split")
            return 1

    model_checkpoints = {}

    if args.mode in ['train', 'full']:
        for model_name in args.models:
            output_dir = base_dir / model_name
            output_dir.mkdir(exist_ok=True)

            if train_model(base_dir, model_name, output_dir, args.device):
                best_model = find_best_checkpoint(output_dir)
                if best_model:
                    model_checkpoints[model_name] = best_model['path']
                    print(f"\nBest {model_name} model:")
                    print(f"  Fold: {best_model['fold']}")
                    print(f"  Val Accuracy: {best_model['val_acc']:.4f}")
    else:
        for model_name in args.models:
            output_dir = base_dir / model_name
            best_model = find_best_checkpoint(output_dir)
            if best_model:
                model_checkpoints[model_name] = best_model['path']

    if args.mode in ['eval', 'full']:
        for model_name, checkpoint in model_checkpoints.items():
            evaluate_model(base_dir, model_name, checkpoint, args.device)

    if args.mode in ['compare', 'full'] and len(model_checkpoints) >= 2:
        compare_models(base_dir, model_checkpoints, args.device)

    print("\n" + "="*80)
    print("Pipeline complete")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
