"""
ResNet Model Architectures (ResNet18, ResNet34, ResNet50)
For comparison with InceptionV3 on DaTscan classification
"""

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Classifier(nn.Module):
    """ResNet18-based classifier"""

    def __init__(self, pretrained=True, freeze_backbone=True):
        super(ResNet18Classifier, self).__init__()

        if pretrained:
            from torchvision.models import ResNet18_Weights
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)

        # Remove final FC layer
        num_features = resnet.fc.in_features  # 512
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Custom classifier (same as InceptionV3)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class ResNet34Classifier(nn.Module):
    """ResNet34-based classifier"""

    def __init__(self, pretrained=True, freeze_backbone=True):
        super(ResNet34Classifier, self).__init__()

        if pretrained:
            from torchvision.models import ResNet34_Weights
            resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet34(weights=None)

        num_features = resnet.fc.in_features  # 512
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class ResNet50Classifier(nn.Module):
    """ResNet50-based classifier"""

    def __init__(self, pretrained=True, freeze_backbone=True):
        super(ResNet50Classifier, self).__init__()

        if pretrained:
            from torchvision.models import ResNet50_Weights
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = models.resnet50(weights=None)

        num_features = resnet.fc.in_features  # 2048
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def create_model(model_name='resnet50', pretrained=True, freeze_backbone=True, device='cuda'):
    """
    Create and initialize a ResNet model

    Args:
        model_name: One of ['resnet18', 'resnet34', 'resnet50']
        pretrained: Use ImageNet pre-trained weights
        freeze_backbone: Freeze backbone weights
        device: Device to place model on

    Returns:
        model: PyTorch model
    """
    model_dict = {
        'resnet18': ResNet18Classifier,
        'resnet34': ResNet34Classifier,
        'resnet50': ResNet50Classifier,
    }

    if model_name not in model_dict:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_dict.keys())}")

    model = model_dict[model_name](pretrained=pretrained, freeze_backbone=freeze_backbone)

    # Move to device
    if torch.cuda.is_available() and device == 'cuda':
        model = model.cuda()
        print(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
    else:
        model = model.to('cpu')
        print("Model on CPU")

    return model


def count_parameters(model):
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    return total_params, trainable_params


if __name__ == "__main__":
    # Test all models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models_to_test = ['resnet18', 'resnet34', 'resnet50']

    for model_name in models_to_test:
        print(f"\n{'='*80}")
        print(f"Testing {model_name.upper()}")
        print(f"{'='*80}")

        model = create_model(model_name, pretrained=True, freeze_backbone=True, device=device)
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
