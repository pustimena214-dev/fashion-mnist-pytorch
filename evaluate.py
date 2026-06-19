from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import torch


def evaluate_model(
        model,
        test_loader,
        device):

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)

            preds = outputs.argmax(1)

            y_true.extend(labels.numpy())
            y_pred.extend(preds.cpu().numpy())

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="macro"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro"
    )

    return accuracy, precision, recall, f1
