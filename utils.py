""" Utilities for image augmentations and eval metrics. """

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torchvision import transforms

from config import config
from image_aug import RandAug
    
def compute_metrics(labels, preds):
    """ Computes AUROC, accuracy and F1 scores for an epoch. """
    epoch_metrics = {}

    df = pd.DataFrame({'labels': labels, 'preds': preds})
    acc = accuracy_score(df['labels'], (df['preds']>0.5).astype(int))
    auroc = roc_auc_score(df['labels'], df['preds'])
    f1 = f1_score(df['labels'], (df['preds']>0.5).astype(int))

    epoch_metrics['acc'] = acc
    epoch_metrics['auroc'] = auroc
    epoch_metrics['f1'] = f1

    return epoch_metrics

def save_stats(stats, mode, fold_idx=None):
    """ Saves training and validation metrics per epoch OR test results. """
    if mode == "train_val":
        if fold_idx is None:
            # training stats for model training stage
            out_dir = os.path.join(config.RESULTS_PATH, "train_val_stats")
        else:
            # per-fold stats for cross-val stage
            out_dir = os.path.join(config.RESULTS_PATH, f"fold_{fold_idx}", "train_val_stats")
        os.makedirs(out_dir, exist_ok=True)

        path = os.path.join(out_dir, "epoch_stats.txt")
        with open(path, "w") as file:
            file.write(
                f"{'Epoch':<10}{'Train AUROC':<15}{'Train acc':<15}{'Train F1':<15}"
                f"{'Train Loss':<15}{'Val AUROC':<15}{'Val acc':<15}{'Val F1':<15}{'Val Loss':<15}\n"
            )
            for i in range(len(stats["epoch"])):
                file.write(
                    f"{stats['epoch'][i]:<10}"
                    f"{stats['train_auroc'][i]:<15.4g}"
                    f"{stats['train_acc'][i]:<15.4g}"
                    f"{stats['train_f1'][i]:<15.4g}"
                    f"{stats['train_loss'][i]:<15.4g}"
                    f"{stats['val_auroc'][i]:<15.4g}"
                    f"{stats['val_acc'][i]:<15.4g}"
                    f"{stats['val_f1'][i]:<15.4g}"
                    f"{stats['val_loss'][i]:<15.4g}\n"
                )

            file.write("\n")
            file.write(f"Best epoch (early stopping): {stats['best_epoch']}")

    elif mode == "test":
        path = os.path.join(config.RESULTS_PATH, "test_results.txt")
        with open(path, "w") as file:
            file.write(f"AUROC: {stats['auroc']:.4g}\n")
            file.write(f"Accuracy: {stats['acc']:.4g}\n")
            file.write(f"F1: {stats['f1']:.4g}\n")

def plot_train_val_stats(train_val_stats, fold_idx=None):
    """ Plots training and validation metrics per epoch. """
    if fold_idx is None:
        # plots for model training stage
        plots_path = os.path.join(config.RESULTS_PATH, "train_val_stats")
    else:
        # plots for cross-val stage
        plots_path = os.path.join(config.RESULTS_PATH, f"fold_{fold_idx}", "train_val_stats")
    os.makedirs(plots_path, exist_ok=True)

    metric_configs = [
        ("Accuracy", "Accuracy", train_val_stats["train_acc"], train_val_stats["val_acc"], "b"),
        ("AUROC", "AUROC", train_val_stats["train_auroc"], train_val_stats["val_auroc"], "r"),
        ("F1", "F1 Score", train_val_stats["train_f1"], train_val_stats["val_f1"], "m"),
        ("Loss", "Loss", train_val_stats["train_loss"], train_val_stats["val_loss"], "g"),
    ]

    # individual plots
    for metric_name, y_label, train_metric, val_metric, colour in metric_configs:
        plt.figure(figsize=(10, 6))
        plt.plot(train_val_stats["epoch"], train_metric, f"{colour}-", label=f"Train {metric_name}")
        plt.plot(train_val_stats["epoch"], val_metric, f"{colour}--", label=f"Val {metric_name}")
        plt.axvline(train_val_stats["best_epoch"], color="k", linestyle="-", label="Best epoch")
        plt.xlabel("Epoch")
        plt.ylabel(y_label)
        if fold_idx is not None:
            title_suffix = f"(Fold {fold_idx})"
            plt.title(f"Training and Validation {metric_name} {title_suffix}")
        else:
            plt.title(f"Training and Validation {metric_name}")
        plt.legend()
        plt.grid(True)
        fname = f"{metric_name.lower()}.png"
        plt.savefig(os.path.join(plots_path, fname))
        plt.close()

    # combined plot
    plt.figure(figsize=(15, 10))
    # Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(train_val_stats["epoch"], train_val_stats["train_acc"], "b-", label="Train Accuracy")
    plt.plot(train_val_stats["epoch"], train_val_stats["val_acc"], "b--", label="Val Accuracy")
    plt.axvline(train_val_stats["best_epoch"], color="k", linestyle="-", label="Best epoch")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.grid(True)
    # AUROC
    plt.subplot(2, 2, 2)
    plt.plot(train_val_stats["epoch"], train_val_stats["train_auroc"], "r-", label="Train AUROC")
    plt.plot(train_val_stats["epoch"], train_val_stats["val_auroc"], "r--", label="Val AUROC")
    plt.axvline(train_val_stats["best_epoch"], color="k", linestyle="-", label="Best epoch")
    plt.xlabel("Epoch"); plt.ylabel("AUROC"); plt.legend(); plt.grid(True)
    # F1
    plt.subplot(2, 2, 3)
    plt.plot(train_val_stats["epoch"], train_val_stats["train_f1"], "m-", label="Train F1")
    plt.plot(train_val_stats["epoch"], train_val_stats["val_f1"], "m--", label="Val F1")
    plt.axvline(train_val_stats["best_epoch"], color="k", linestyle="-", label="Best epoch")
    plt.xlabel("Epoch"); plt.ylabel("F1 Score"); plt.legend(); plt.grid(True)
    # Loss
    plt.subplot(2, 2, 4)
    plt.plot(train_val_stats["epoch"], train_val_stats["train_loss"], "g-", label="Train Loss")
    plt.plot(train_val_stats["epoch"], train_val_stats["val_loss"], "g--", label="Val Loss")
    plt.axvline(train_val_stats["best_epoch"], color="k", linestyle="-", label="Best epoch")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, "all_metrics.png"))
    plt.close()

def datscan_transforms(model, data_aug=False):
    """ Defines image augmentations for DaTSCAN encoder model. """
    if 'resnet' in model:
        transforms_list = transforms.Compose(
            [
                transforms.Resize((128, 128)),                                              # resize image to (128, 128, 3)
                transforms.ToTensor(),                                                      # convert image to tensor of shape (3, 128, 128)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalise tensor
            ]
        )

    elif 'efficientnet' in model:
        transforms_list = transforms.Compose(
            [
                transforms.Resize((384, 384)),                                              # resize image to (384, 384, 3)
                transforms.ToTensor(),                                                      # convert image to tensor of shape (3, 384, 384)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalise tensor
            ]
        )

    elif 'inception' in model:
        transforms_list = transforms.Compose(
            [
                transforms.Resize((299, 299)),                                              # resize image to (299, 299, 3)
                transforms.ToTensor(),                                                      # convert image to tensor of shape (3, 299, 299)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalise tensor
            ]
        )
    
    if data_aug:
        transforms_list.transforms.insert(0, RandAug(2, 10))                            # additionally apply any two image augmentations before resizing
    return transforms_list

def mri_transforms(model, data_aug=False):
    """ Defines image augmentations for MRI encoder model. """
    if 'resnet' in model:
        transforms_list = transforms.Compose(
            [
                transforms.Resize((224, 224)),                                              # resize image to (224, 224, 3)
                transforms.ToTensor(),                                                      # convert image to tensor of shape (3, 224, 224)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalise tensor
            ]
        )
        
    elif 'efficientnet' in model:
        transforms_list = transforms.Compose(
            [
                transforms.Resize((384, 384)),                                              # resize image to (384, 384, 3)
                transforms.ToTensor(),                                                      # convert image to tensor of shape (3, 384, 384)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalise tensor
            ]
        )

    elif 'inception' in model:
        transforms_list = transforms.Compose(
            [
                transforms.Resize((299, 299)),                                              # resize image to (299, 299, 3)
                transforms.ToTensor(),                                                      # convert image to tensor of shape (3, 299, 299)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalise tensor
            ]
        )
    
    if data_aug:
        transforms_list.transforms.insert(0, RandAug(2, 10))                            # additionally apply any two image augmentations before resizing
    return transforms_list

def save_attn_overlay(input_img_tensor, attn_map, save_path, alpha=0.4):
    """ Creates and saves a heatmap overlay for a single image. """
    if attn_map.dim() == 3 and attn_map.shape[0] == 1:
        attn_map = attn_map[0]  # (H_attn, W_attn)

    img = input_img_tensor.detach().cpu().clone()
    attn = attn_map.detach().cpu().clone()

    # unnormalize input image for visualization
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    img = img * std + mean        # (3, H, W)
    img = img.clamp(0.0, 1.0)

    img_np = img.permute(1, 2, 0).numpy()  # (H, W, 3)

    # resize attention map to image size
    H, W, _ = img_np.shape
    attn_np = torch.nn.functional.interpolate(
        attn.unsqueeze(0).unsqueeze(0),  # (1,1,H_attn,W_attn)
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()

    # normalize attn to [0,1] for colormap
    attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)

    plt.figure(figsize=(4, 4))
    plt.imshow(img_np)
    # plt.imshow(attn_np, cmap="jet", alpha=alpha)
    plt.imshow(attn_np, cmap="Greens", alpha=alpha)
    plt.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()