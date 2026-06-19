import torch
import pandas as pd

from dataset import get_dataloaders
from models import create_model
from train import train_model
from evaluate import evaluate_model


device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

train_loader, val_loader, test_loader = \
    get_dataloaders()

experiments = [

    ("alexnet", False),
    ("alexnet", True),

    ("vgg16", False),
    ("vgg16", True),

    ("resnet18", False),
    ("resnet18", True),

    ("efficientnet", False),
    ("efficientnet", True)
]

results = []

for model_name, pretrained in experiments:

    print("=" * 50)

    print(
        model_name,
        "FineTune" if pretrained else "Scratch"
    )

    model = create_model(
        model_name,
        pretrained
    )

    model = train_model(
        model,
        train_loader,
        val_loader,
        device
    )

    accuracy, precision, recall, f1 = \
        evaluate_model(
            model,
            test_loader,
            device
        )

    results.append([
        model_name,
        "FineTune" if pretrained else "Scratch",
        accuracy,
        precision,
        recall,
        f1
    ])

results = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Training",
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]
)

print(results)

results.to_csv(
    "results.csv",
    index=False
)
