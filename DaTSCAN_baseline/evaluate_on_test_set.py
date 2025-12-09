"""
Final Evaluation on Held-Out Test Set

Evaluates the best model (selected from k-fold CV) on the held-out test set.
This gives the true generalization performance.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

from train_kfold_weighted_bce import DaTscanDataset, get_transforms
from model import create_model


class TestSetEvaluator:
    """
    Evaluates best model on held-out test set
    """

    def __init__(self, test_data_dir, best_model_path, output_dir, device='cuda'):
        self.test_data_dir = Path(test_data_dir)
        self.best_model_path = Path(best_model_path)
        self.output_dir = Path(output_dir)
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.img_size = 299
        self.batch_size = 16

        print(f"Using device: {self.device}")

    def create_test_loader(self):
        """Create test data loader"""
        dataset = DaTscanDataset(
            self.test_data_dir,
            transform=get_transforms(augment=False, img_size=self.img_size)
        )

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )

        return loader, dataset

    def get_predictions(self, model, data_loader):
        """Get predictions from model"""
        model.eval()

        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc='Testing'):
                images = images.to(self.device)

                outputs = model(images)
                probabilities = torch.sigmoid(outputs)

                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy().flatten())
                all_predictions.extend((probabilities > 0.5).cpu().numpy().flatten())

        return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)

    def evaluate(self):
        """Evaluate on test set"""
        print("\n" + "="*80)
        print("FINAL EVALUATION ON HELD-OUT TEST SET")
        print("="*80 + "\n")

        # Load model
        print(f"Loading best model from: {self.best_model_path}")
        model = create_model(pretrained=True, freeze_backbone=True, device=self.device)
        checkpoint = torch.load(self.best_model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Model from epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"Validation accuracy during training: {checkpoint.get('val_acc', 'unknown'):.4f}\n")

        # Load test data
        test_loader, test_dataset = self.create_test_loader()

        print(f"Test set size: {len(test_dataset)}")
        print(f"Classes: {test_dataset.classes}\n")

        # Get predictions
        y_true, y_pred, y_pred_proba = self.get_predictions(model, test_loader)

        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value

        # F1 Score = 2 * (precision * recall) / (precision + recall)
        # where recall = sensitivity
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)

        # Store results
        results = {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'f1_score': float(f1_score),
            'npv': float(npv),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'test_size': int(len(y_true)),
            'test_pd': int(np.sum(y_true)),
            'test_control': int(len(y_true) - np.sum(y_true))
        }

        # Save results
        with open(self.output_dir / "test_set_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        # Print results
        print("\n" + "="*80)
        print("TEST SET RESULTS (HELD-OUT DATA)")
        print("="*80 + "\n")

        # Convert to int for printing
        n_total = int(len(y_true))
        n_pd = int(np.sum(y_true))
        n_control = n_total - n_pd

        print(f"Test Set Size: {n_total} subjects")
        print(f"  PD: {n_pd} ({n_pd/n_total*100:.1f}%)")
        print(f"  Control: {n_control} ({n_control/n_total*100:.1f}%)")
        print()
        print(f"Performance Metrics:")
        print(f"  Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Sensitivity:     {sensitivity:.4f} ({sensitivity*100:.2f}%)")
        print(f"  Specificity:     {specificity:.4f} ({specificity*100:.2f}%)")
        print(f"  Precision (PPV): {precision:.4f} ({precision*100:.2f}%)")
        print(f"  F1 Score:        {f1_score:.4f} ({f1_score*100:.2f}%)")
        print(f"  NPV:             {npv:.4f} ({npv*100:.2f}%)")
        print(f"  ROC AUC:         {roc_auc:.4f}")
        print(f"  PR AUC:          {pr_auc:.4f}")
        print()
        print(f"Confusion Matrix:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
        print("\n" + "="*80)

        # Plot confusion matrix
        self.plot_confusion_matrix(y_true, y_pred)

        # Plot ROC curve
        self.plot_roc_curve(fpr, tpr, roc_auc)

        # Plot PR curve
        self.plot_pr_curve(precision_curve, recall_curve, pr_auc)

        return results

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Control', 'PD'],
                    yticklabels=['Control', 'PD'])
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix_test.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrix saved")

    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """Plot ROC curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Test Set', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curve_test.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"ROC curve saved")

    def plot_pr_curve(self, precision, recall, pr_auc):
        """Plot Precision-Recall curve"""
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkgreen', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve - Test Set', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "pr_curve_test.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"PR curve saved")


def main():
    parser = argparse.ArgumentParser(description="Evaluate best model on held-out test set")

    parser.add_argument(
        '--test_data_dir',
        type=str,
        default='kfold_data_with_test/test_set',
        help='Directory containing test set'
    )

    parser.add_argument(
        '--best_model_path',
        type=str,
        required=True,
        help='Path to best model checkpoint (.pth file)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='test_set_results',
        help='Output directory for results'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = TestSetEvaluator(
        test_data_dir=args.test_data_dir,
        best_model_path=args.best_model_path,
        output_dir=args.output_dir,
        device=args.device
    )

    # Evaluate
    results = evaluator.evaluate()

    print(f"\nTest set evaluation complete!")
    print(f"   Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
