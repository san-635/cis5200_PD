""" Command line arguments for the entire codebase. """

import os
import argparse

parser = argparse.ArgumentParser()

# for main.py
parser.add_argument('--GPU', type=int, default=0, help='ID of GPU to use')
parser.add_argument('--SEED', type=int, default=42, help='Random seed for reproducibility')
parser.add_argument('--BATCH_SIZE', type=int, default=32, help='Batch size')
parser.add_argument('DATSCAN_ENC', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'inception_v3'], help='Encoder architecture for DaTSCAN modality')
parser.add_argument('MRI_ENC', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'efficientnet_b0', 'inception_v3'], help='Encoder architecture for MRI modality')
parser.add_argument('--ATTN_FLAG', action='store_true', help='Use attention-based joint model')
parser.add_argument('--FREEZE_ENCS', action='store_true', help='Freeze encoder backbone weights during training')
parser.add_argument('RESULTS_PATH_SUFFIX', type=str, help='Path to save results')
parser.add_argument('--BASELINE', type=str, choices=['none', 'datscan', 'mri'], default='none', help='DaTSCAN or sMRI baseline, default: none = use joint multimodal model')
parser.add_argument('--EVAL', action='store_true', help='Run final training + test on held-out test set (skip CV if set alone)')

# for trainer.py
parser.add_argument('--LR', type=float, default=0.0001, help='(Max) Learning rate')
parser.add_argument('--LR_SCH', type=str, default='step', choices=['linearW', 'step', 'cosine'], help='Learning rate scheduler, default: step (i.e., ReduceLROnPlateau)')
parser.add_argument('--LOSS_FLAG', action='store_false', help='If not passed, use weighted BCE loss; if passed, use unweighted BCE loss')
parser.add_argument('--WEIGHT_DECAY', type=float, default=0.0, help='Weight decay for AdamW optimizer')
parser.add_argument('--EPOCHS', type=int, default=50, help='Max number of epochs for training routine')
parser.add_argument('--PATIENCE', type=int, default=5, help='Patience for early stopping')

# for dataset_prep.py and dataset.py
parser.add_argument('--PPMI_PARENT_PATH', type=str, default="/shared_data/p_vidalr/sanjanac/", help='Path to the parent directory of the combined PPMI dataset')
parser.add_argument('--PRINT_STATS', action='store_true', default=False, help='Print statistics about the dataset')
parser.add_argument('--DATA_AUG_FLAG', action='store_false', help='If not passed, apply image augmentation techniques during training')

config = parser.parse_args()

config.DEVICE = 'cuda' if config.GPU >= 0 else 'cpu'
config.RESULTS_PATH = os.path.join(os.path.dirname(__file__), 'Results', config.RESULTS_PATH_SUFFIX)