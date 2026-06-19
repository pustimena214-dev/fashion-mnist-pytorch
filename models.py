import torch.nn as nn
from torchvision import models


def create_model(model_name, pretrained=True):

    if model_name == "alexnet":

        weights = (
            models.AlexNet_Weights.DEFAULT
            if pretrained else None
        )

        model = models.alexnet(weights=weights)

        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            10
        )

    elif model_name == "vgg16":

        weights = (
            models.VGG16_Weights.DEFAULT
            if pretrained else None
        )

        model = models.vgg16(weights=weights)

        model.classifier[6] = nn.Linear(
            model.classifier[6].in_features,
            10
        )

    elif model_name == "resnet18":

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained else None
        )

        model = models.resnet18(weights=weights)

        model.fc = nn.Linear(
            model.fc.in_features,
            10
        )

    elif model_name == "efficientnet":

        weights = (
            models.EfficientNet_B0_Weights.DEFAULT
            if pretrained else None
        )

        model = models.efficientnet_b0(
            weights=weights
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            10
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model
