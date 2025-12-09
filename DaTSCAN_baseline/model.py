"""
Neural Network Model Architecture - PyTorch Implementation
InceptionV3 base + custom binary classifier
Based on the paper's methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DaTscanClassifier(nn.Module):
    """
    DaTscan Parkinson's Disease Classifier
    - InceptionV3 backbone (pre-trained on ImageNet)
    - Custom binary classification head
    """

    def __init__(self, pretrained=True, freeze_backbone=True):
        super(DaTscanClassifier, self).__init__()

        # Load pre-trained InceptionV3
        # Use weights parameter for newer PyTorch versions
        if pretrained:
            from torchvision.models import Inception_V3_Weights
            self.backbone = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.inception_v3(weights=None)

        # Remove the original classifier
        # InceptionV3 has aux_logits, we'll disable it during training
        self.backbone.aux_logits = False
        num_features = self.backbone.fc.in_features

        # Remove the final FC layer
        self.backbone.fc = nn.Identity()

        # Freeze backbone if requested (transfer learning)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier head (following the paper)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling (already done in backbone)
            nn.Flatten(),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Extract features from backbone
        x = self.backbone(x)

        # Apply classifier
        x = self.classifier(x)

        return x

    def unfreeze_backbone(self, num_layers=30):
        """
        Unfreeze the last num_layers of the backbone for fine-tuning
        """
        # Get all parameters
        all_params = list(self.backbone.parameters())

        # Unfreeze last num_layers
        for param in all_params[-num_layers:]:
            param.requires_grad = True

        print(f"Unfroze last {num_layers} layers of backbone")


class DaTscanClassifierV2(nn.Module):
    """
    Alternative implementation with better control over InceptionV3
    """

    def __init__(self, pretrained=True, freeze_backbone=True):
        super(DaTscanClassifierV2, self).__init__()

        # Load InceptionV3 without the final classifier
        # Use weights parameter for newer PyTorch versions
        if pretrained:
            from torchvision.models import Inception_V3_Weights
            inception = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        else:
            inception = models.inception_v3(weights=None)

        # Disable aux_logits
        inception.aux_logits = False

        # Store the backbone (everything except the final FC)
        self.backbone = inception

        # Get the number of features before the FC layer
        num_features = inception.fc.in_features

        # Replace the final FC layer with identity
        self.backbone.fc = nn.Identity()

        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier following the paper
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
            # No sigmoid - BCEWithLogitsLoss includes it internally
        )

    def forward(self, x):
        # Ensure aux_logits is disabled during forward pass
        self.backbone.aux_logits = False
        # Extract features (InceptionV3 already has Global Avg Pool)
        x = self.backbone(x)

        # Classify
        x = self.classifier(x)

        return x


def create_model(pretrained=True, freeze_backbone=True, device='cuda'):
    """
    Create and initialize the model

    Args:
        pretrained: Use ImageNet pre-trained weights
        freeze_backbone: Freeze backbone weights for transfer learning
        device: Device to place model on

    Returns:
        model: PyTorch model
    """
    model = DaTscanClassifierV2(pretrained=pretrained, freeze_backbone=freeze_backbone)

    # Move to device
    if torch.cuda.is_available() and device == 'cuda':
        model = model.cuda()
        print(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
    else:
        model = model.to('cpu')
        print("Model on CPU")

    return model


def count_parameters(model):
    """
    Count model parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    return total_params, trainable_params


class StepDecayLR:
    """
    Learning rate scheduler with step decay
    Gradually decreases LR from initial to final over total_epochs
    """

    def __init__(self, optimizer, initial_lr=1e-3, final_lr=1e-6, total_epochs=500):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def step(self):
        """Update learning rate"""
        # Exponential decay
        lr = self.initial_lr * (self.final_lr / self.initial_lr) ** (self.current_epoch / self.total_epochs)

        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1

        if self.current_epoch % 50 == 0:
            print(f"Epoch {self.current_epoch}: Learning rate = {lr:.6f}")

        return lr

    def get_last_lr(self):
        """Get current learning rate"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


if __name__ == "__main__":
    # Test model creation
    print("Creating model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(pretrained=True, freeze_backbone=True, device=device)

    print("\nModel Architecture:")
    print(model)

    print("\nParameter counts:")
    count_parameters(model)

    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(2, 3, 299, 299)
    if device == 'cuda':
        dummy_input = dummy_input.cuda()

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
