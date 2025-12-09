"""
Compare All Models on Test Set

Generates comprehensive visualizations comparing InceptionV3, ResNet18, ResNet34, ResNet50
on the held-out test set including:
- AUROC curves
- Accuracy comparison
- F1 scores
- Sensitivity/Specificity
- Confusion matrices
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
    precision_recall_curve, average_precision_score, f1_score
)

from train_kfold_weighted_bce import DaTscanDataset, get_transforms
from model import create_model as create_inception_model


class ModelComparator:
    """
    Compare multiple models on test set
    """

    def __init__(self, test_data_dir, model_configs, output_dir, device='cuda'):
        """
        Args:
            test_data_dir: Path to test data
            model_configs: List of dicts with 'name', 'type', 'path'
            output_dir: Output directory for results
            device: cuda or cpu
        """
        self.test_data_dir = Path(test_data_dir)
        self.model_configs = model_configs
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

    def load_model(self, model_type, model_path):
        """Load a model based on type"""
        if model_type == 'inceptionv3':
            # Load InceptionV3
            model = create_inception_model(pretrained=True, freeze_backbone=True, device=self.device)
        else:
            # Load ResNet
            
            from models_resnet import create_model as create_resnet_model
            model = create_resnet_model(
                model_name=model_type,
                pretrained=True,
                freeze_backbone=True,
                device=self.device
            )

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model

    def get_predictions(self, model, data_loader):
        """Get predictions from model"""
        model.eval()

        all_labels = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc='Testing', leave=False):
                images = images.to(self.device)

                outputs = model(images)
                probabilities = torch.sigmoid(outputs)

                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy().flatten())
                all_predictions.extend((probabilities > 0.5).cpu().numpy().flatten())

        return np.array(all_labels), np.array(all_predictions), np.array(all_probabilities)

    def compute_metrics(self, y_true, y_pred, y_pred_proba):
        """Compute all metrics"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = f1_score(y_true, y_pred)

        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)

        return {
            'accuracy': float(accuracy),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'npv': float(npv),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'precision_curve': precision_curve.tolist(),
            'recall_curve': recall_curve.tolist()
        }

    def compare_all_models(self):
        """Compare all models on test set"""
        print("\n" + "="*80)
        print("COMPARING ALL MODELS ON HELD-OUT TEST SET")
        print("="*80 + "\n")

        # Load test data
        test_loader, test_dataset = self.create_test_loader()
        print(f"Test set size: {len(test_dataset)}\n")

        # Evaluate each model
        all_results = {}

        for config in self.model_configs:
            model_name = config['name']
            model_type = config['type']
            model_path = config['path']

            print(f"\nEvaluating {model_name}...")
            print(f"  Loading from: {model_path}")

            # Load model
            model = self.load_model(model_type, model_path)

            # Get predictions
            y_true, y_pred, y_pred_proba = self.get_predictions(model, test_loader)

            # Compute metrics
            metrics = self.compute_metrics(y_true, y_pred, y_pred_proba)

            all_results[model_name] = {
                'metrics': metrics,
                'y_true': y_true,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }

            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  ROC AUC:  {metrics['roc_auc']:.4f}")

        # Save results
        self.save_results(all_results)

        # Generate visualizations
        self.plot_comparison_metrics(all_results)
        self.plot_roc_curves(all_results)
        self.plot_confusion_matrices(all_results)
        self.plot_detailed_comparison(all_results)

        return all_results

    def save_results(self, all_results):
        """Save comparison results"""
        # Create comparison table
        comparison_data = []

        for model_name, data in all_results.items():
            metrics = data['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'F1 Score': metrics['f1_score'],
                'Sensitivity': metrics['sensitivity'],
                'Specificity': metrics['specificity'],
                'Precision': metrics['precision'],
                'ROC AUC': metrics['roc_auc'],
                'PR AUC': metrics['pr_auc']
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('ROC AUC', ascending=False)
        df.to_csv(self.output_dir / "model_comparison.csv", index=False)

        # Save detailed results
        detailed_results = {}
        for model_name, data in all_results.items():
            detailed_results[model_name] = {k: v for k, v in data['metrics'].items()
                                           if k not in ['fpr', 'tpr', 'precision_curve', 'recall_curve']}

        with open(self.output_dir / "detailed_results.json", 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"\nResults saved to {self.output_dir}")

    def plot_comparison_metrics(self, all_results):
        """Plot bar chart comparing all metrics"""
        metrics_to_plot = ['accuracy', 'f1_score', 'sensitivity', 'specificity', 'roc_auc']
        metric_labels = ['Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'ROC AUC']

        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(20, 4))

        for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            models = list(all_results.keys())
            values = [all_results[m]['metrics'][metric] for m in models]

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(models)]
            axes[idx].bar(models, values, color=colors, alpha=0.8)
            axes[idx].set_ylabel(label, fontsize=12)
            axes[idx].set_ylim([0, 1.0])
            axes[idx].set_title(f'{label} Comparison', fontsize=13, fontweight='bold')
            axes[idx].grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom', fontsize=10)

            # Rotate x-axis labels
            axes[idx].set_xticklabels(models, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig(self.output_dir / "metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Metrics comparison saved")

    def plot_roc_curves(self, all_results):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        linestyles = ['-', '--', '-.', ':']

        for idx, (model_name, data) in enumerate(all_results.items()):
            metrics = data['metrics']
            fpr = np.array(metrics['fpr'])
            tpr = np.array(metrics['tpr'])
            roc_auc = metrics['roc_auc']

            plt.plot(fpr, tpr, color=colors[idx % len(colors)],
                    linestyle=linestyles[idx % len(linestyles)],
                    lw=2.5, label=f'{model_name} (AUC = {roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curves - All Models (Test Set)', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "roc_curves_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"ROC curves comparison saved")

    def plot_confusion_matrices(self, all_results):
        """Plot confusion matrices for all models"""
        n_models = len(all_results)
        cols = 2
        rows = (n_models + 1) // 2

        fig, axes = plt.subplots(rows, cols, figsize=(12, 6*rows))
        axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, data) in enumerate(all_results.items()):
            y_true = data['y_true']
            y_pred = data['y_pred']
            metrics = data['metrics']

            cm = confusion_matrix(y_true, y_pred)

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Control', 'PD'],
                       yticklabels=['Control', 'PD'],
                       cbar=True, square=True)

            axes[idx].set_ylabel('True Label', fontsize=11)
            axes[idx].set_xlabel('Predicted Label', fontsize=11)
            axes[idx].set_title(f'{model_name}\nAcc: {metrics["accuracy"]:.3f}, F1: {metrics["f1_score"]:.3f}',
                               fontsize=12, fontweight='bold')

        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrices_all.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Confusion matrices saved")

    def plot_detailed_comparison(self, all_results):
        """Plot detailed comparison with multiple subplots"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.35, top=0.95, bottom=0.05, left=0.08, right=0.95)

        # 1. Accuracy, F1, ROC AUC comparison
        ax1 = fig.add_subplot(gs[0, 0])
        models = list(all_results.keys())
        metrics_data = {
            'Accuracy': [all_results[m]['metrics']['accuracy'] for m in models],
            'F1 Score': [all_results[m]['metrics']['f1_score'] for m in models],
            'ROC AUC': [all_results[m]['metrics']['roc_auc'] for m in models]
        }

        x = np.arange(len(models))
        width = 0.25
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for idx, (metric, values) in enumerate(metrics_data.items()):
            ax1.bar(x + idx*width, values, width, label=metric, color=colors[idx], alpha=0.8)

        ax1.set_ylabel('Score', fontsize=12)
        ax1.set_title('Primary Metrics Comparison', fontsize=13, fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim([0, 1.0])

        # 2. Sensitivity vs Specificity
        ax2 = fig.add_subplot(gs[0, 1])
        sensitivity = [all_results[m]['metrics']['sensitivity'] for m in models]
        specificity = [all_results[m]['metrics']['specificity'] for m in models]

        x = np.arange(len(models))
        ax2.bar(x - 0.2, sensitivity, 0.4, label='Sensitivity', color='#2ca02c', alpha=0.8)
        ax2.bar(x + 0.2, specificity, 0.4, label='Specificity', color='#d62728', alpha=0.8)

        ax2.set_ylabel('Score', fontsize=12)
        ax2.set_title('Sensitivity vs Specificity', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim([0, 1.0])

        # 3. ROC Curves
        ax3 = fig.add_subplot(gs[1, :])
        colors_roc = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for idx, (model_name, data) in enumerate(all_results.items()):
            metrics = data['metrics']
            fpr = np.array(metrics['fpr'])
            tpr = np.array(metrics['tpr'])
            roc_auc = metrics['roc_auc']
            ax3.plot(fpr, tpr, color=colors_roc[idx % len(colors_roc)], lw=2.5,
                    label=f'{model_name} (AUC = {roc_auc:.4f})')

        ax3.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.5)
        ax3.set_xlabel('False Positive Rate', fontsize=12)
        ax3.set_ylabel('True Positive Rate', fontsize=12)
        ax3.set_title('ROC Curves - All Models', fontsize=13, fontweight='bold')
        ax3.legend(loc="lower right")
        ax3.grid(True, alpha=0.3)

        # 4. Performance Table
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('tight')
        ax4.axis('off')

        table_data = []
        for model_name in models:
            metrics = all_results[model_name]['metrics']
            table_data.append([
                model_name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['f1_score']:.4f}",
                f"{metrics['sensitivity']:.4f}",
                f"{metrics['specificity']:.4f}",
                f"{metrics['precision']:.4f}",
                f"{metrics['roc_auc']:.4f}"
            ])

        table = ax4.table(cellText=table_data,
                         colLabels=['Model', 'Accuracy', 'F1 Score', 'Sensitivity', 'Specificity', 'Precision', 'ROC AUC'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Style header
        for i in range(7):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Alternate row colors
        for i in range(1, len(table_data) + 1):
            for j in range(7):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#E7E6E6')

        plt.savefig(self.output_dir / "detailed_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Detailed comparison saved")

    def print_summary(self, all_results):
        """Print summary of results"""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80 + "\n")

        # Sort by ROC AUC
        sorted_models = sorted(all_results.items(),
                             key=lambda x: x[1]['metrics']['roc_auc'],
                             reverse=True)

        print(f"{'Model':<20} {'Accuracy':<12} {'F1 Score':<12} {'Sensitivity':<12} {'Specificity':<12} {'ROC AUC':<12}")
        print("-" * 92)

        for model_name, data in sorted_models:
            metrics = data['metrics']
            print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['f1_score']:<12.4f} "
                  f"{metrics['sensitivity']:<12.4f} {metrics['specificity']:<12.4f} {metrics['roc_auc']:<12.4f}")

        print("\n" + "="*80)

        # Best model
        best_model = sorted_models[0]
        print(f"\n🏆 BEST MODEL (by ROC AUC): {best_model[0]}")
        print(f"   ROC AUC: {best_model[1]['metrics']['roc_auc']:.4f}")
        print(f"   Accuracy: {best_model[1]['metrics']['accuracy']:.4f}")
        print(f"   F1 Score: {best_model[1]['metrics']['f1_score']:.4f}")
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare all models on test set")

    parser.add_argument(
        '--test_data_dir',
        type=str,
        default='kfold_data_with_test/test_set',
        help='Directory containing test set'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        default='model_comparison_results',
        help='Output directory for comparison results'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use'
    )

    # Model paths
    parser.add_argument('--inceptionv3_path', type=str, help='Path to InceptionV3 best model')
    parser.add_argument('--resnet18_path', type=str, help='Path to ResNet18 best model')
    parser.add_argument('--resnet34_path', type=str, help='Path to ResNet34 best model')
    parser.add_argument('--resnet50_path', type=str, help='Path to ResNet50 best model')

    args = parser.parse_args()

    # Build model configurations
    model_configs = []

    if args.inceptionv3_path and os.path.exists(args.inceptionv3_path):
        model_configs.append({
            'name': 'InceptionV3',
            'type': 'inceptionv3',
            'path': args.inceptionv3_path
        })

    if args.resnet18_path and os.path.exists(args.resnet18_path):
        model_configs.append({
            'name': 'ResNet18',
            'type': 'resnet18',
            'path': args.resnet18_path
        })

    if args.resnet34_path and os.path.exists(args.resnet34_path):
        model_configs.append({
            'name': 'ResNet34',
            'type': 'resnet34',
            'path': args.resnet34_path
        })

    if args.resnet50_path and os.path.exists(args.resnet50_path):
        model_configs.append({
            'name': 'ResNet50',
            'type': 'resnet50',
            'path': args.resnet50_path
        })

    if len(model_configs) == 0:
        print("ERROR: No valid model paths provided!")
        print("\nPlease provide at least one model path:")
        print("  --inceptionv3_path PATH")
        print("  --resnet18_path PATH")
        print("  --resnet34_path PATH")
        print("  --resnet50_path PATH")
        return

    print(f"\nFound {len(model_configs)} models to compare:")
    for config in model_configs:
        print(f"  - {config['name']}")

    # Create comparator
    comparator = ModelComparator(
        test_data_dir=args.test_data_dir,
        model_configs=model_configs,
        output_dir=args.output_dir,
        device=args.device
    )

    # Run comparison
    all_results = comparator.compare_all_models()

    # Print summary
    comparator.print_summary(all_results)

    print(f"\nModel comparison complete!")
    print(f"   Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
