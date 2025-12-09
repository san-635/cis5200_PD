DaTSCAN Classification with Weighted Binary Cross-Entropy Loss

This pipeline implements DaTSCAN-based Parkinson's Disease classification using
class-weighted Binary Cross-Entropy loss instead of Focal Loss.

Requirements:
    pip install -r requirements.txt

Data Structure:
    Parent directory should contain:
    - preprocessed_images/
    - train_metadata.csv
    - test_metadata.csv

Usage:

    Full Pipeline (recommended):
        python run_pipeline.py --mode full --device cuda

    Individual Steps:
        python run_pipeline.py --mode split
        python run_pipeline.py --mode train --models inceptionv3 resnet18
        python run_pipeline.py --mode eval
        python run_pipeline.py --mode compare

    Train Single Model:
        python train_kfold_weighted_bce.py \
            --kfold_data_dir ../kfold_data_with_test \
            --output_dir ./inceptionv3 \
            --device cuda

Training Configuration:
    - Loss: Weighted BCE (w_pos = n_neg / n_pos)
    - Cross-Validation: 10-fold stratified
    - Early Stopping: Patience 50 epochs
    - Optimizer: Adam (lr: 1e-3 to 1e-6)
    - Batch Size: 16
    - Max Epochs: 500

Models:
    - InceptionV3 (ImageNet pretrained)
    - ResNet-18 (ImageNet pretrained)
    - ResNet-34 (ImageNet pretrained)
    - ResNet-50 (ImageNet pretrained)

Output Structure:
    weighted_bce_experiment/
    ├── inceptionv3/
    │   ├── training_summary.csv
    │   ├── model_fold_X_best.pth (10 folds)
    │   └── history_fold_X.csv (10 folds)
    ├── resnet18/
    ├── resnet34/
    ├── resnet50/
    └── model_comparison/
        ├── model_comparison.csv
        ├── roc_curves_comparison.png
        └── confusion_matrices_all.png
