"""
Step 5 (DenseNet-121 variant): DenseNet-121 model, adapted for 5-class DR grading.
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes=5, pretrained=True):
    """
    Loads DenseNet-121 pretrained on ImageNet, and replaces the final
    1000-class layer with a 5-class layer for DR grading (0-4).
    """
    model = models.densenet121(weights="IMAGENET1K_V1" if pretrained else None)

    # model.classifier is the final classification layer — swap it out
    # (DenseNet's equivalent of ResNet's model.fc)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, num_classes)

    return model


def freeze_backbone(model):
    """
    Stage 1 of fine-tuning: lock every layer EXCEPT the new final layer.
    Only the classification head learns at first.
    """
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    return model


def unfreeze_backbone(model, unfreeze_from_layer="denseblock3"):
    """
    Stage 2 of fine-tuning: unlock the deeper layers so they can adapt
    to fundus images too, starting from `unfreeze_from_layer` onward.

    DenseNet-121's feature extractor is, in order: denseblock1 ->
    transition1 -> denseblock2 -> transition2 -> denseblock3 ->
    transition3 -> denseblock4 -> norm5 -> classifier. Default
    "denseblock3" unfreezes denseblock3, transition3, denseblock4, norm5,
    and classifier — leaving denseblock1/denseblock2 (generic low-level
    features) frozen, mirroring ResNet-50's default of "layer3".
    """
    unfreeze = False
    for name, param in model.named_parameters():
        if unfreeze_from_layer in name:
            unfreeze = True
        if unfreeze:
            param.requires_grad = True
    return model


# ---------- Example usage ----------
if __name__ == "__main__":
    model = build_model(num_classes=5, pretrained=True)

    # Stage 1: train only the new head
    model = freeze_backbone(model)
    print("Stage 1: backbone frozen, only classifier layer trainable")

    # ... train for a few epochs here ...

    # Stage 2: unfreeze deeper layers, continue training at a lower LR
    model = unfreeze_backbone(model, unfreeze_from_layer="denseblock3")
    print("Stage 2: denseblock3, denseblock4, and classifier are now trainable")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")