import os
import glob
import copy
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torchvision import transforms as T
import torchio as tio
import torch
import torch.nn as nn

# 2D Model Imports
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms as T

CONFIG = {
    "data_root": "./dataset_root",       
    "labels_csv": "ppmi_labels.csv",     
    "batch_size": 16,
    "lr": 1e-5,
    "num_epochs": 20,                    
    "patience": 5,                       
    "target_shape": (256, 256),          
    "seed": 42,
    "num_workers": 0,                    
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


def get_transforms(mode='train', target_shape=(256, 256)):
    if mode == 'train':
        # all these transforms are to simulate real world variations b/w different MRI machines
        return T.Compose([
            T.ToPILImage(), # converts to python image library image
            T.RandomResizedCrop(target_shape, scale=(0.8, 1.0)), # randomly zooms into image (keeping b/w 80-100%) and then stretches back to 256x256 to avoid memorizing specific things
            T.RandomHorizontalFlip(), # roughly symmetric 
            T.RandomRotation(15), # ignore head tilt
            T.ColorJitter(brightness=0.2, contrast=0.2), # randomly darkens or lightens the image / changes contrast
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]) # normalize
        ])
    else:
        return T.Compose([
            T.ToPILImage(),
            T.Resize(target_shape), # resize
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]) # normalize
        ])

class SMRIDataset(torch.utils.data.Dataset):
    def __init__(self, subject_list, transform=None):
        # stores list of patients and image augmentations to apply
        self.subjects = subject_list
        self.transform = transform

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subject = self.subjects[idx]
        data_3d = subject.mri.data.float()

        # get dimensions of data to 256x256
        if data_3d.shape[3] == 1:
            image_2d = data_3d.squeeze() 
            image_2d = image_2d.unsqueeze(0)
        else:
            depth = data_3d.shape[3]
            image_2d = data_3d[:, :, :, depth // 2]

        if self.transform:
            image_2d = self.transform(image_2d)
        
        label_tensor = torch.tensor(subject.label, dtype=torch.float32)
        return image_2d, label_tensor

def get_subjects_list(data_root, labels_df):
    subjects = []
    print(f"Scanning directory: {data_root}...")

    for patient_id in os.listdir(data_root):
        patient_path = os.path.join(data_root, patient_id)
        if not os.path.isdir(patient_path): continue
        if patient_id not in labels_df.index: continue

        label = labels_df.loc[patient_id, 'Diagnosis']

        subfolders = [d for d in os.listdir(patient_path) if os.path.isdir(os.path.join(patient_path, d))]
        mri_folder_name = next((f for f in subfolders if f.startswith("MRI")), None)

        if mri_folder_name is None: continue
            
        mri_dir = os.path.join(patient_path, mri_folder_name)
        mri_files = glob.glob(os.path.join(mri_dir, "*"))
        if len(mri_files) == 0: continue

        mri_path = mri_files[0] 

        subject = tio.Subject(
            mri=tio.ScalarImage(mri_path),
            label=label,
            patient_id=patient_id
        )
        subjects.append(subject)

    print(f"Found {len(subjects)} valid subjects.")
    return subjects


class SMRI2DResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(SMRI2DResNet, self).__init__()
        weights = ResNe18_Weights.DEFAULT if pretrained else None # start w/ resnet weights
        self.model = resnet18(weights=weights)
        
        # making it so that can look at grayscale images (no RGB like resnet originally)
        old_conv = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias
        )
        
        if pretrained:
            self.model.conv1.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True) # avg together the RGB vals

        # update last layer of model to just output one score
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.model(x)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    # training loop
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(outputs)
        all_preds.extend(probs.detach().cpu().numpy())
        all_targets.extend(labels.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
             print(f"   Batch {batch_idx+1}/{len(loader)}...")

    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc = roc_auc_score(all_targets, all_preds) if len(set(all_targets)) > 1 else 0.5

    return epoch_loss, epoch_auc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)
    binary_preds = [1 if p >= 0.5 else 0 for p in all_preds]
    acc = accuracy_score(all_targets, binary_preds)
    f1 = f1_score(all_targets, binary_preds)
    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5 

    return total_loss, acc, auc, f1

def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    print("Loading labels...")
    labels_df = pd.read_csv(CONFIG["labels_csv"])
    labels_df['Patient_ID'] = labels_df['Patient_ID'].astype(str)
    labels_df.set_index('Patient_ID', inplace=True)

    all_subjects = get_subjects_list(CONFIG["data_root"], labels_df)
    if len(all_subjects) == 0:
        print("No subjects found! Check paths.")
        return

    y_all = [s.label for s in all_subjects]

    train_val_subjects, test_subjects, y_train, y_test = train_test_split(
        all_subjects, y_all, test_size=0.3, stratify=y_all, random_state=CONFIG["seed"]
    )

    print(f"Total: {len(all_subjects)} | Train/Val: {len(train_val_subjects)} | Test: {len(test_subjects)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG["seed"])
    y_train_val = [s.label for s in train_val_subjects]
    
    best_overall_auc = 0.0
    best_model_state = None

    print("\nStarting 5-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_subjects, y_train_val)):
        print(f"\n--- FOLD {fold+1}/5 ---")

        fold_train_subjects = [train_val_subjects[i] for i in train_idx]
        fold_val_subjects = [train_val_subjects[i] for i in val_idx]

        fold_labels = [s.label for s in fold_train_subjects]
        num_neg = fold_labels.count(0)
        num_pos = fold_labels.count(1)
        
        # down-weight the Parkinson's
        pos_weight_value = num_neg / num_pos
        pos_weight = torch.tensor([pos_weight_value]).to(CONFIG['device'])
        
        print(f"   Class Balance: {num_neg} Neg (0), {num_pos} Pos (1)")
        print(f"   Using pos_weight: {pos_weight_value:.4f}")
        
        train_ds = SMRIDataset(fold_train_subjects, transform=get_transforms('train', CONFIG['target_shape']))
        val_ds = SMRIDataset(fold_val_subjects, transform=get_transforms('val', CONFIG['target_shape']))

        train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=CONFIG['num_workers'])
        val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

        model = SMRI2DResNet(pretrained=True).to(CONFIG['device'])
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # weighing penalty of negatives more to account for fact that most data are sick patients
        
        optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

        best_fold_auc = 0.0
        patience_counter = 0

        for epoch in range(CONFIG['num_epochs']):
            train_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimizer, CONFIG['device'])
            val_loss, val_acc, val_auc, val_f1 = evaluate(model, val_loader, criterion, CONFIG['device'])

            scheduler.step(val_loss)

            print(f"Ep {epoch+1}: Loss {train_loss:.3f} | Val AUC {val_auc:.3f} | Val Acc {val_acc:.3f}")

            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                patience_counter = 0
                if val_auc > best_overall_auc:
                    best_overall_auc = val_auc
                    best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1

            if patience_counter >= CONFIG['patience']:
                print("Early stopping.")
                break

    print("\nEvaluated Best Model on Held-out Test Set...")
    if best_model_state is None:
        print("Training failed.")
        return

    final_model = SMRI2DResNet(pretrained=False).to(CONFIG['device'])
    final_model.load_state_dict(best_model_state)

    test_ds = SMRIDataset(test_subjects, transform=get_transforms('val', CONFIG['target_shape']))
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=CONFIG['num_workers'])

    _, test_acc, test_auc, test_f1 = evaluate(final_model, test_loader, criterion, CONFIG['device'])

    print("="*30)
    print("FINAL TEST RESULTS (2D sMRI)")
    print("="*30)
    print(f"Accuracy: {test_acc:.4f}")
    print(f"AUROC:    {test_auc:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print("="*30)

if __name__ == "__main__":
    main()