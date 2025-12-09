"""
K-Fold Cross-Validation Training with Weighted BCE Loss and Early Stopping
This version uses class-weighted Binary Cross-Entropy instead of Focal Loss
Matches the sMRI baseline methodology from CIS 5200 report
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from model import create_model as create_inception_model, count_parameters
from models_resnet import create_model as create_resnet_model


class DaTscanDataset(Dataset):
    """Custom Dataset for DaTscan images"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        self.labels = []

        # Class to idx mapping (Control=0, PD=1)
        self.class_to_idx = {'Control': 0, 'PD': 1}
        self.classes = list(self.class_to_idx.keys())

        # Load all samples
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            for img_path in class_dir.glob("*.png"):
                self.samples.append(str(img_path))
                self.labels.append(class_idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32).unsqueeze(0)


def get_transforms(augment=True, img_size=299):
    """Data augmentation as per paper"""
    if augment:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),  # Width and height shifts ±10%
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2),  # Brightness 0.8-1.2
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # Validation transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


class StepDecayLR:
    """Learning rate scheduler with step decay from 1e-3 to 1e-6 over 500 epochs"""

    def __init__(self, optimizer, initial_lr=1e-3, final_lr=1e-6, total_epochs=500):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def step(self):
        """Update learning rate using exponential decay"""
        lr = self.initial_lr * (self.final_lr / self.initial_lr) ** (self.current_epoch / self.total_epochs)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr

    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for handling class imbalance

    Unlike Focal Loss, this applies a simple class weight to each sample
    matching the sMRI baseline methodology in CIS 5200 report
    """

    def __init__(self, pos_weight=1.0):
        """
        Args:
            pos_weight: Weight for positive class (PD)
                       Calculated as ratio of negative to positive samples
        """
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs: raw logits from model (before sigmoid)
            targets: ground truth labels (0 or 1)

        Returns:
            Weighted BCE loss
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Weighted BCE: -[w_p * y * log(p) + (1-y) * log(1-p)]
        loss = -(self.pos_weight * targets * torch.log(probs + 1e-7) +
                 (1 - targets) * torch.log(1 - probs + 1e-7))

        return loss.mean()


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """

    def __init__(self, patience=50, min_delta=0.001, mode='min'):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, epoch, score):
        """
        Check if should stop

        Args:
            epoch: Current epoch number
            score: Validation metric (loss or accuracy)

        Returns:
            True if should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False

        # Check if improved
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class KFoldTrainerWeightedBCE:
    """
    K-Fold trainer with Weighted BCE Loss and early stopping
    Matches sMRI baseline methodology from CIS 5200 report
    """

    def __init__(self, kfold_data_dir, output_dir, n_folds=10, device='cuda',
                 early_stopping_patience=50, early_stopping_metric='val_loss', model_arch='inceptionv3'):
        self.kfold_data_dir = Path(kfold_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_folds = n_folds
        self.device = device
        self.model_arch = model_arch.lower()

        # Training hyperparameters
        self.max_epochs = 500  # Maximum epochs (early stopping may stop earlier)
        self.batch_size = 16
        self.img_size = 299
        self.initial_lr = 1e-3
        self.final_lr = 1e-6

        # Loss function: Weighted BCE instead of Focal Loss
        self.use_focal_loss = False  # Changed from True

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_metric = early_stopping_metric

    def calculate_pos_weight(self, train_dataset):
        """
        Calculate positive class weight as ratio of negative to positive samples

        Args:
            train_dataset: Training dataset

        Returns:
            pos_weight: Weight for positive class
        """
        labels = []
        for _, label in train_dataset:
            labels.append(label)

        labels = torch.tensor(labels)
        n_pos = labels.sum().item()
        n_neg = len(labels) - n_pos

        # w_p = n_neg / n_pos
        pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        print(f"  Training samples: {len(labels)} (PD: {int(n_pos)}, Control: {int(n_neg)})")
        print(f"  Positive class weight: {pos_weight:.4f}")

        return pos_weight

    def create_data_loaders(self, train_dir, val_dir):
        """Create training and validation data loaders"""

        train_dataset = DaTscanDataset(
            train_dir,
            transform=get_transforms(augment=True, img_size=self.img_size)
        )

        val_dataset = DaTscanDataset(
            val_dir,
            transform=get_transforms(augment=False, img_size=self.img_size)
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device == 'cuda' else False
        )

        return train_loader, val_loader, train_dataset, val_dataset

    def train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch"""
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc='Training', leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device).float().unsqueeze(1)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self, model, val_loader, criterion):
        """Validate the model"""
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).float().unsqueeze(1)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()

                # Calculate accuracy
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = running_loss / len(val_loader)
        val_acc = correct / total

        return val_loss, val_acc

    def train_fold(self, fold_idx):
        """Train a single fold"""
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx + 1}/{self.n_folds}")
        print(f"{'='*60}")

        # Setup data
        train_dir = self.kfold_data_dir / f'fold_{fold_idx}' / 'train'
        val_dir = self.kfold_data_dir / f'fold_{fold_idx}' / 'val'

        train_loader, val_loader, train_dataset, val_dataset = self.create_data_loaders(train_dir, val_dir)

        # Calculate positive class weight
        pos_weight = self.calculate_pos_weight(train_dataset)

        # Create model based on architecture
        if self.model_arch == 'inceptionv3':
            model = create_inception_model(pretrained=True, freeze_backbone=True, device=self.device)
        elif self.model_arch in ['resnet18', 'resnet34', 'resnet50']:
            model = create_resnet_model(model_name=self.model_arch, pretrained=True, freeze_backbone=True, device=self.device)
        else:
            raise ValueError(f"Unknown model architecture: {self.model_arch}")

        # Loss function: Weighted BCE
        criterion = WeightedBCELoss(pos_weight=pos_weight)
        print(f"Using Weighted BCE Loss (pos_weight={pos_weight:.4f})")

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.initial_lr)

        # Learning rate scheduler
        lr_scheduler = StepDecayLR(
            optimizer,
            initial_lr=self.initial_lr,
            final_lr=self.final_lr,
            total_epochs=self.max_epochs
        )

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.early_stopping_patience,
            min_delta=0.001,
            mode='min' if self.early_stopping_metric == 'val_loss' else 'max'
        )

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }

        best_val_acc = 0.0
        best_val_loss = float('inf')

        # Training loop
        for epoch in range(self.max_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)

            # Validate
            val_loss, val_acc = self.validate(model, val_loader, criterion)

            # Update learning rate
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)

            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.max_epochs}] "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
                      f"LR: {current_lr:.6f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss

                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'train_acc': train_acc,
                    'train_loss': train_loss
                }

                torch.save(checkpoint, self.output_dir / f'model_fold_{fold_idx}_best.pth')

            # Early stopping
            metric_value = val_loss if self.early_stopping_metric == 'val_loss' else val_acc
            if early_stopping(epoch, metric_value):
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                print(f"Best epoch: {early_stopping.best_epoch + 1}")
                break

        # Save training history
        history_df = pd.DataFrame(history)
        history_df.to_csv(self.output_dir / f'history_fold_{fold_idx}.csv', index=False)

        # Plot training curves
        self.plot_training_curves(history, fold_idx)

        return {
            'fold': fold_idx,
            'total_epochs': epoch + 1,
            'best_epoch': early_stopping.best_epoch + 1,
            'best_val_acc': best_val_acc,
            'best_val_loss': best_val_loss,
            'early_stopped': early_stopping.early_stop
        }

    def plot_training_curves(self, history, fold_idx):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Loss
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title(f'Fold {fold_idx} - Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(history['train_acc'], label='Train Acc')
        axes[1].plot(history['val_acc'], label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title(f'Fold {fold_idx} - Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Learning rate
        axes[2].plot(history['lr'])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_title(f'Fold {fold_idx} - Learning Rate')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / f'training_curves_fold_{fold_idx}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def train_all_folds(self):
        """Train all folds"""
        print("="*60)
        print("K-FOLD CROSS-VALIDATION WITH WEIGHTED BCE LOSS")
        print("="*60)
        print(f"Number of folds: {self.n_folds}")
        print(f"Max epochs per fold: {self.max_epochs}")
        print(f"Early stopping patience: {self.early_stopping_patience}")
        print(f"Early stopping metric: {self.early_stopping_metric}")
        print(f"Loss function: Weighted Binary Cross-Entropy")
        print("="*60)

        results = []

        for fold_idx in range(self.n_folds):
            fold_result = self.train_fold(fold_idx)
            results.append(fold_result)

        # Save summary
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'training_summary.csv', index=False)

        # Print summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(results_df.to_string(index=False))
        print("\n" + "="*60)
        print(f"Mean validation accuracy: {results_df['best_val_acc'].mean():.4f} ± {results_df['best_val_acc'].std():.4f}")
        print(f"Mean epochs trained: {results_df['total_epochs'].mean():.1f}")
        print(f"Folds with early stopping: {results_df['early_stopped'].sum()}/{self.n_folds}")
        print("="*60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train DaTSCAN models with Weighted BCE Loss")
    parser.add_argument('--model', type=str, default='inceptionv3',
                        choices=['inceptionv3', 'resnet18', 'resnet34', 'resnet50'],
                        help='Model architecture to train')
    parser.add_argument('--kfold_data_dir', type=str, default='kfold_data_with_test',
                        help='Directory containing k-fold data')
    parser.add_argument('--output_dir', type=str, default='kfold_results_weighted_bce',
                        help='Output directory for results')
    parser.add_argument('--n_folds', type=int, default=10,
                        help='Number of folds')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--metric', type=str, default='val_loss', choices=['val_loss', 'val_acc'],
                        help='Metric for early stopping')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'],
                        help='Device to use')

    args = parser.parse_args()

    trainer = KFoldTrainerWeightedBCE(
        kfold_data_dir=args.kfold_data_dir,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        device=args.device,
        early_stopping_patience=args.patience,
        early_stopping_metric=args.metric,
        model_arch=args.model
    )

    trainer.train_all_folds()


if __name__ == '__main__':
    main()
