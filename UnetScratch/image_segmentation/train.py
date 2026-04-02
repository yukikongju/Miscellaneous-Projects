"""
Description:
    Training script for a UNet segmentation model on the Kaggle
    'Understanding Clouds from Satellite Images' dataset. Trains a
    smp.Unet with a ResNet50 encoder to predict 4 cloud type masks
    (Fish, Flower, Gravel, Sugar) using BCE+Dice loss.

Usage:
    python train.py

Parameters:
    ENCODER         : str   - Encoder backbone name (default: 'resnet50')
    ENCODER_WEIGHTS : str   - Pretrained weights for encoder (default: 'imagenet')
    ACTIVATION      : None  - Output activation (None = raw logits, handled by BCEDiceLoss)
    NUM_EPOCHS      : int   - Number of training epochs (default: 20)
    DATASET_PATH    : str   - Path to dataset root containing train.csv and train_images/
    TRAIN_BATCH_SIZE: int   - Batch size for training loader (default: 2)
    TEST_BATCH_SIZE : int   - Batch size for validation loader (default: 2)
"""

import segmentation_models_pytorch as smp
import torch
from datasets import CloudDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from metrics import BCEDiceLoss


def load_model(encoder_name: str, encoder_weights_name: str, activation_fn: torch.nn, device: str):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights_name,
        classes=4,
        activation=activation_fn,
    )
    return model.to(device)


def get_loaders(train_batch_size: int, val_batch_size: int):
    DATASET_PATH = "/Users/emulie/Downloads/understanding_cloud_organization/"
    train_dataset = CloudDataset(DATASET_PATH, "train", train_split=0.8)
    val_dataset = CloudDataset(DATASET_PATH, "val", train_split=0.8)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=True)
    loaders = {
        "train": train_loader,
        "val": val_loader,
    }
    return loaders


def train_loop(
    model: torch.nn.Module, optimizer: torch.optim, criterion, train_loader: DataLoader, device: str
):
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader):
        x = batch["image"].float().to(device) / 255.0
        y = batch["mask"].float().to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    return total_train_loss / len(train_loader)


def val_loop(model: torch.nn.Module, criterion, val_loader: DataLoader, device: str):
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader):
            x = batch["image"].float().to(device) / 255.0
            y = batch["mask"].float().to(device)

            logits = model(x)
            loss = criterion(logits, y)
            total_eval_loss += loss.item()
    return total_eval_loss / len(val_loader)


def train(
    model: torch.nn.Module,
    optimizer,
    criterion,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: str,
):
    writer = SummaryWriter()

    for epoch in tqdm(range(epochs)):
        train_loss = train_loop(model, optimizer, criterion, train_loader, device)
        val_loss = val_loop(model, criterion, val_loader, device)

        print(f"Epoch {epoch + 1}: {train_loss=} {val_loss=}")
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Val/Loss", val_loss, epoch)

    writer.close()


def main():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    ENCODER = "resnet50"
    ENCODER_WEIGHTS = "imagenet"
    ACTIVATION = None
    NUM_EPOCHS = 20
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE = 2, 2

    model = load_model(
        encoder_name=ENCODER,
        encoder_weights_name=ENCODER_WEIGHTS,
        activation_fn=ACTIVATION,
        device=device,
    )
    loaders = get_loaders(TRAIN_BATCH_SIZE, VAL_BATCH_SIZE)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.decoder.parameters(), "lr": 1e-3},
            {"params": model.encoder.parameters(), "lr": 1e-3},
        ]
    )
    criterion = BCEDiceLoss()
    #  scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    #  runner = SupervisedRunner()
    train(model, optimizer, criterion, loaders["train"], loaders["val"], NUM_EPOCHS, device)


if __name__ == "__main__":
    main()
