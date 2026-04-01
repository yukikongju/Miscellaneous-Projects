"""
Dummy training job to ensure the following:
- reading dataset from GCS
- experiment tracking with mlflow
- pass environment variables

To figure out:
- tensorboard

Arguments:
--data_dir
--epochs
--batch_size
--lr

Usage:
> uv run python playground/dummy_train.py --epochs=5 --batch_size=8
> tensorboard --logdir=runs
"""

import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from tqdm import tqdm


def get_data(data_dir, batch_size):
    transform = transforms.ToTensor()

    train_ds = datasets.FakeData(transform=transform)  # 224x224x3
    val_ds = datasets.FakeData(transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    return train_loader, val_loader


def get_model():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(224 * 224 * 3, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def train(model, train_loader, val_loader, epochs, lr, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter()

    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()

        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

    writer.close()


def main(args):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    train_loader, val_loader = get_data(args.data_dir, args.batch_size)

    model = get_model().to(device)

    train(model, train_loader, val_loader, args.epochs, args.lr, device)

    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "model.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/gcs/data")
    parser.add_argument("--model_dir", type=str, default=os.environ.get("AIP_MODEL_DIR", "./model"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)

    args = parser.parse_args()
    main(args)
