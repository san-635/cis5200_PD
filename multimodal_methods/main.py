""" Sets up environment, creates dataset, performs 5-fold cross-validation, and performs full model training and evaluation. """

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

from dataset_prep import main as create_dataset
from dataset import PPMI
from models import Encoder, JointModel, JointAttnModel, UnimodalModel
from trainer import Trainer
from utils import save_attn_overlay
from config import config

def build_model():
    """ Initializes multimodal / baseline model based on config settings. """
    if config.BASELINE == 'datscan':
        enc = Encoder(modality='datscan').to(config.DEVICE)
        model = UnimodalModel(enc, modality='datscan').to(config.DEVICE)
    elif config.BASELINE == 'mri':
        enc = Encoder(modality='mri').to(config.DEVICE)
        model = UnimodalModel(enc, modality='mri').to(config.DEVICE)
    else:
        datscan_enc = Encoder(modality='datscan').to(config.DEVICE)
        mri_enc = Encoder(modality='mri').to(config.DEVICE)
        if config.ATTN_FLAG:
            model = JointAttnModel(datscan_enc, mri_enc).to(config.DEVICE)
        else:
            model = JointModel(datscan_enc, mri_enc).to(config.DEVICE)
    return model

def generate_overlays(num_samples=10):
    model = build_model()
    ckpt_path = os.path.join(config.RESULTS_PATH, "model_ckpt.pth")
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()

    out_dir = os.path.join(config.RESULTS_PATH, "attention_overlays")
    os.makedirs(out_dir, exist_ok=True)

    test_dataset = PPMI("test")
    pos_indices, neg_indices = [], []
    for idx in range(len(test_dataset)):
        label = int(test_dataset.pairs[idx]["label"])  # or test_dataset.labels[idx] if you store it separately
        if label == 1 and len(pos_indices) < num_samples // 2:
            pos_indices.append(idx)
        elif label == 0 and len(neg_indices) < num_samples // 2:
            neg_indices.append(idx)
        if len(pos_indices) >= num_samples // 2 and len(neg_indices) >= num_samples // 2:
            break
    selected_indices = pos_indices + neg_indices

    with torch.no_grad():
        for idx in selected_indices:
            sample = test_dataset[idx]
            datscan = sample["datscan"].unsqueeze(0).to(config.DEVICE)  # (1,3,H,W)
            mri = sample["mri"].unsqueeze(0).to(config.DEVICE)          # (1,3,H,W)
            label = int(sample["label"])
            subject_id = test_dataset.pairs[idx]["subject"]

            out = model(datscan, mri)
            ds_attn = out["datscan_attn"][0]  # (1,H,W)
            mri_attn = out["mri_attn"][0]     # (1,H,W)

            label_str = "PD" if label == 1 else "control"
            ds_path = os.path.join(out_dir, f"subject{subject_id}_{label_str}_datscan.png")
            mri_path = os.path.join(out_dir, f"subject{subject_id}_{label_str}_mri.png")

            save_attn_overlay(datscan[0], ds_attn, ds_path, alpha=0.45)
            save_attn_overlay(mri[0], mri_attn, mri_path, alpha=0.45)

def main():
    ## Set up environment ##
    # GPU and CUDA settings
    if config.GPU >= 0:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.GPU)
        torch.cuda.empty_cache()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # random seed settings
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # results directory
    os.makedirs(config.RESULTS_PATH, exist_ok=True)

    ## Dataset preparation ##
    # create dataset folders / metadata (comment out if already created)
    # print(f"\n======= CREATING DATASET =======")
    # create_dataset()

    train_dataset = PPMI("train")
    indices = list(range(len(train_dataset)))
    
    ## Cross-val stage (for hyperparam tuning) ##
    if not config.EVAL:
        kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
        fold_results = {}

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            print(f"\n============ FOLD {fold_idx} ============")
            # save config args for this fold
            fold_dir = os.path.join(config.RESULTS_PATH, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)
            with open(os.path.join(fold_dir, "config_args.txt"), "w") as results_file:
                for arg in vars(config):
                    results_file.write(f"{arg:<30}: {getattr(config, arg)}\n")

            # create train and val dataloaders for this fold
            print(f"Creating dataloaders")
            fold_train_dl = DataLoader(Subset(train_dataset, train_idx), batch_size=config.BATCH_SIZE, shuffle=True)
            fold_val_dl = DataLoader(Subset(train_dataset, val_idx), batch_size=config.BATCH_SIZE, shuffle=False)

            # initialize multimodal model and trainer for this fold
            print(f"Initializing model")
            model = build_model()
            trainer = Trainer(train_dl=fold_train_dl, val_dl=fold_val_dl, test_dl=None, model=model)

            # run training routine and save best val AUROC for this fold
            print(f"Training + val")
            trainer.train(fold_idx=fold_idx)
            fold_results[f"Fold {fold_idx}"] = trainer.best_val_auroc

        # print and save cross-val summary
        print("\n========== CV SUMMARY ==========")
        vals = list(fold_results.values())
        mean_auroc = np.mean(vals)
        std_auroc = np.std(vals)
        for k, v in fold_results.items():
            print(f"{k}: best val AUROC = {v:.4g}")
        print(f"Mean AUROC = {mean_auroc:.4g} +/- {std_auroc:.4g}")

        cv_path = os.path.join(config.RESULTS_PATH, "CV_summary.txt")
        with open(cv_path, "w") as f:
            for k, v in fold_results.items():
                f.write(f"{k}: best val AUROC = {v:.4g}\n")
            f.write(f"\nMean AUROC = {mean_auroc:.4g} +/- {std_auroc:.4g}\n")

    ## Model training and test stage (for reporting) ##
    if config.EVAL:
        # save config args
        with open(os.path.join(config.RESULTS_PATH, "config_args.txt"), "w") as results_file:
            for arg in vars(config):
                results_file.write(f"{arg:<30}: {getattr(config, arg)}\n")
        
        # create train, val (from train dataset) and test (from test dataset) dataloaders
        print("\n===== CREATING DATALOADERS =====")
        split = int(0.875 * len(train_dataset)) # 87.5% train (70% of overall), 12.5% val (10% of overall)
        full_train_idx = indices[:split]
        full_val_idx = indices[split:]

        full_train_dl = DataLoader(Subset(train_dataset, full_train_idx), batch_size=config.BATCH_SIZE, shuffle=True)
        full_val_dl = DataLoader(Subset(train_dataset, full_val_idx), batch_size=config.BATCH_SIZE, shuffle=False)
        test_dl = DataLoader(PPMI("test"), batch_size=config.BATCH_SIZE, shuffle=False)

        # initialize multimodal model and trainer
        print("\n====== INITIALIZING MODEL ======")
        model = build_model()
        full_trainer = Trainer(train_dl=full_train_dl, val_dl=full_val_dl, test_dl=test_dl, model=model)

        print("\n======== TRAINING + VAL ========")
        full_trainer.train()

        print("\n============= TEST =============")
        full_trainer.test()

        # optionally generate interpretable images of attention map overlay on input scans for JointAttnModel
        if config.ATTN_FLAG:
            print("\n===== GENERATING OVERLAYS =====")
            generate_overlays(num_samples=10)
            print(f"Attention overlay images saved")

if __name__ == "__main__":
    main()