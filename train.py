import torch
import torch.nn as nn


def train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=5):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    best_val_acc = 0

    model.to(device)

    for epoch in range(epochs):

        model.train()

        for batch_idx, (images, labels) in enumerate(train_loader):

            if batch_idx % 100 == 0:

                print(
                    f"Epoch {epoch+1}, "
                    f"Batch {batch_idx}/{len(train_loader)}"
                )

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                preds = outputs.argmax(1)

                correct += (
                    preds == labels
                ).sum().item()

                total += labels.size(0)

        val_acc = correct / total

        print(
            f"Validation Accuracy: "
            f"{val_acc:.4f}"
        )

        if val_acc > best_val_acc:

            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                f"{model.__class__.__name__}_best.pth"
            )

    return model
