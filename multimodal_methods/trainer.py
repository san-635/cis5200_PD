""" Defines training (cross-val and final stages) and testing routines for the joint multimodal model. """

import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR, ReduceLROnPlateau, CosineAnnealingLR

from utils import compute_metrics, plot_train_val_stats, save_stats
from config import config

random.seed(config.SEED)

class Trainer:
    def __init__(self, train_dl, val_dl, test_dl, model):
        self.device = config.DEVICE
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.test_dl = test_dl
        self.model = model

        # loss function (with optional class weighting, where positive class weight = # neg train samples / # pos train samples)
        if config.LOSS_FLAG:
            ds = self.train_dl.dataset
            if hasattr(ds, "pairs"):          # direct PPMI (during model training stage)
                pairs = ds.pairs
                indices = range(len(pairs))
            else:                             # subset of PPMI (during cross-val stage)
                base_ds = ds.dataset
                indices = ds.indices
                pairs = base_ds.pairs

            subset_labels = [pairs[i]["label"] for i in indices]
            subset_labels = torch.tensor(subset_labels, dtype=torch.float32, device=self.device)

            pos = subset_labels.sum()
            neg = len(subset_labels) - pos
            pos_weight = torch.tensor([neg / pos], device=self.device, dtype=torch.float32)
            self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss = nn.BCEWithLogitsLoss()

        # optimizer and LR scheduler
        self.optimiser = optim.AdamW(self.model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
        if config.LR_SCH == "linearW":
            self.scheduler = OneCycleLR(self.optimiser, max_lr=config.LR, epochs=config.EPOCHS, steps_per_epoch=len(self.train_dl), pct_start=0.1, anneal_strategy="linear", final_div_factor=1.0, three_phase=False)
        elif config.LR_SCH == "step":
            self.scheduler = ReduceLROnPlateau(self.optimiser, factor=0.5, patience=config.EPOCHS // 5, mode="min")
        elif config.LR_SCH == "cosine":
            self.scheduler = CosineAnnealingLR(self.optimiser, T_max=config.EPOCHS)

        # tracking
        self.epoch = 0
        self.patience = 0
        self.best_val_auroc = 0.0
        self.train_val_stats = {"epoch": [], "train_acc": [], "train_auroc": [], "train_f1": [], "train_loss": [],
                                "val_acc": [], "val_auroc": [], "val_f1": [], "val_loss": []}
        self.test_results = {}

    def train_epoch(self):
        """ Trains the model for one epoch on train set. """
        train_preds = torch.FloatTensor().to(self.device)
        train_labels = torch.FloatTensor().to(self.device)
        epoch_loss = 0.0

        for batch in self.train_dl:
            datscan = batch["datscan"].to(self.device)
            mri = batch["mri"].to(self.device)
            train_label = batch["label"].to(self.device).unsqueeze(1)

            model_ret = self.model(datscan, mri)
            train_logits = model_ret["logits"]
            train_pred = model_ret["preds"]
            loss = self.loss(train_logits, train_label)
            epoch_loss += loss.item()

            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()

            # update if a per-batch scheduler
            if isinstance(self.scheduler, (OneCycleLR, CosineAnnealingLR)):
                self.scheduler.step()

            train_preds = torch.cat((train_preds, train_pred), dim=0)
            train_labels = torch.cat((train_labels, train_label), dim=0)

        ret = compute_metrics(train_labels.view(-1).data.cpu().numpy(), train_preds.view(-1).data.cpu().numpy())
        for metric in ["acc", "auroc", "f1"]:
            self.train_val_stats[f"train_{metric}"].append(ret[metric])
        self.train_val_stats["train_loss"].append(epoch_loss / len(self.train_dl))
        self.train_val_stats["epoch"].append(self.epoch)

    def val_epoch(self):
        """ Validates the model for one epoch on val set. """
        val_preds = torch.FloatTensor().to(self.device)
        val_labels = torch.FloatTensor().to(self.device)
        epoch_loss = 0.0

        with torch.no_grad():
            for batch in self.val_dl:
                datscan = batch["datscan"].to(self.device)
                mri = batch["mri"].to(self.device)
                val_label = batch["label"].to(self.device).unsqueeze(1)

                model_ret = self.model(datscan, mri)
                val_logits = model_ret["logits"]
                val_pred = model_ret["preds"]
                loss = self.loss(val_logits, val_label)
                epoch_loss += loss.item()

                val_preds = torch.cat((val_preds, val_pred), dim=0)
                val_labels = torch.cat((val_labels, val_label), dim=0)

        ret = compute_metrics(val_labels.view(-1).data.cpu().numpy(), val_preds.view(-1).data.cpu().numpy())
        for metric in ["acc", "auroc", "f1"]:
            self.train_val_stats[f"val_{metric}"].append(ret[metric])
        self.train_val_stats["val_loss"].append(epoch_loss / len(self.val_dl))

    def test_epoch(self):
        """ Tests the model on the test set. """
        test_preds = torch.FloatTensor().to(self.device)
        test_labels = torch.FloatTensor().to(self.device)

        with torch.no_grad():
            for batch in tqdm(self.test_dl, total=len(self.test_dl), desc="Batches"):
                datscan = batch["datscan"].to(self.device)
                mri = batch["mri"].to(self.device)
                test_label = batch["label"].to(self.device).unsqueeze(1)

                model_ret = self.model(datscan, mri)
                test_pred = model_ret["preds"]

                test_preds = torch.cat((test_preds, test_pred), dim=0)
                test_labels = torch.cat((test_labels, test_label), dim=0)

        self.test_results = compute_metrics(test_labels.view(-1).data.cpu().numpy(), test_preds.view(-1).data.cpu().numpy())

    def train(self, fold_idx=None):
        """ Training routine.
            - For cross-val stage, call with fold_idx=0-4 to train on that fold's train/val sets
            - For model training stage, call with fold_idx=None. """
        pbar = tqdm(range(1, config.EPOCHS + 1), desc="Epochs")
        for self.epoch in pbar:
            self.model.train()
            self.train_epoch()

            self.model.eval()
            self.val_epoch()

            # update if a per-epoch scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(self.train_val_stats["val_loss"][-1])

            # early stopping based on val AUROC
            current_val_auroc = self.train_val_stats["val_auroc"][-1]
            if current_val_auroc > self.best_val_auroc:
                self.best_val_auroc = current_val_auroc
                if fold_idx is None:    # only save checkpoint for model training stage; skip saving for cross-val stage
                    self.save_checkpoint()
                self.patience = 0
            else:
                self.patience += 1

            if self.patience >= config.PATIENCE:
                print(f"Early stopping at epoch {self.epoch}")
                self.train_val_stats["best_epoch"] = self.epoch - config.PATIENCE
                break

            pbar.set_postfix({"train_loss": self.train_val_stats["train_loss"][-1], "val_loss": self.train_val_stats["val_loss"][-1]})

        if "best_epoch" not in self.train_val_stats and len(self.train_val_stats["epoch"]) > 0:
            self.train_val_stats["best_epoch"] = self.train_val_stats["epoch"][-1]

        save_stats(self.train_val_stats, mode="train_val", fold_idx=fold_idx)
        plot_train_val_stats(self.train_val_stats, fold_idx=fold_idx)

    def test(self):
        """ Testing routine. """
        self.load_model()
        self.model.eval()
        self.test_epoch()

        save_stats(self.test_results, mode="test")
        print(f"Test AUROC = {self.test_results['auroc']:.4g}")
        print(f"Test accuracy = {self.test_results['acc']:.4g}")
        print(f"Test F1 = {self.test_results['f1']:.4g}")

    def save_checkpoint(self):
        """ Saves model weights to a checkpoint .pth file. """
        ckpt_path = os.path.join(config.RESULTS_PATH, "model_ckpt.pth")
        
        ckpt = {"model": self.model.state_dict()}
        torch.save(ckpt, ckpt_path)

    def load_model(self):
        """ Loads best model weights from checkpoint .pth file. """
        ckpt_path = os.path.join(config.RESULTS_PATH, "model_ckpt.pth")

        if not os.path.exists(ckpt_path):
            print(f"No checkpoint file found at {ckpt_path}. Train model first.")
        else:
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model"])