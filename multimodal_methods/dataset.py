""" Defines the modality-specific data preprocessing pipelines and datamodules. """

import os
import glob
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from utils import datscan_transforms, mri_transforms
from config import config

class PPMI(Dataset):
    """ Dataset class for the PPMI_datscan_mri dataset. """
    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'test']
        self.split = split
        self.metadata = pd.read_csv(os.path.join(config.PPMI_PARENT_PATH, 'PPMI_datscan_mri', f'{split}_metadata.csv'))
        
        # prepare a list of subject-wise DaTSCAN + MRI pairs with corresponding labels (for retrieving in __getitem__)
        grouped = self.metadata.groupby("Subject")
        self.pairs = []
        for subject, group in grouped:
            subject_id = str(subject)

            row_spect = group[group["Modality"] == "SPECT"].iloc[0]
            row_mri = group[group["Modality"] == "MRI"].iloc[0]
            spect_img_id = str(row_spect["Image Data ID"])
            mri_img_id = str(row_mri["Image Data ID"])
            datscan_folder = os.path.join(config.PPMI_PARENT_PATH, "PPMI_datscan_mri", subject_id, f"SPECT_{spect_img_id}")
            mri_folder = os.path.join(config.PPMI_PARENT_PATH, "PPMI_datscan_mri", subject_id, f"MRI_{mri_img_id}")

            label = 1 if row_spect["Group"] == "PD" else 0

            self.pairs.append(
                {
                    "subject": subject_id,
                    "datscan_folder": datscan_folder,
                    "mri_folder": mri_folder,
                    "label": label,
                }
            )

        # define relevant image augmentations
        if self.split == 'train' and config.DATA_AUG_FLAG:
            self.datscan_transform = datscan_transforms(config.DATSCAN_ENC, data_aug=True)
            self.mri_transform = mri_transforms(config.MRI_ENC, data_aug=True)
        else:
            self.datscan_transform = datscan_transforms(config.DATSCAN_ENC, data_aug=False)
            self.mri_transform = mri_transforms(config.MRI_ENC, data_aug=False)
    
    def __getitem__(self, index):
        """ Loads and returns a subject's DaTSCAN + MRI image pair and their corresponding label."""
        pair = self.pairs[index]

        datscan_image_path = glob.glob(os.path.join(pair["datscan_folder"], "*.jpg"))[0]
        mri_image_path = glob.glob(os.path.join(pair["mri_folder"], "*.jpg"))[0]
        datscan_img = Image.open(datscan_image_path).convert("RGB")
        mri_img = Image.open(mri_image_path).convert("RGB")
        datscan_img = self.datscan_transform(datscan_img)
        mri_img = self.mri_transform(mri_img)

        label = torch.tensor(pair["label"], dtype=torch.float32)

        return {
            "datscan": datscan_img,
            "mri": mri_img,
            "label": label,
        }

    def __len__(self):
        """ Number of subject-wise DaTSCAN + MRI pairs. """
        return len(self.pairs)
    
    # default collate function returns a batch dict with items:
        # "datscan": tensor of shape (config.BATCH_SIZE, C, H, W)
        # "mri": tensor of shape (config.BATCH_SIZE, C, H, W)
        # "label": tensor of shape (config.BATCH_SIZE,)