"""
This script trains and evaluate a model end-to-end using the configs defined
in the yaml file passed as a parameter.
"""

import argparse
import lightning as L
import logging
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from configs.load import load_config
from training.runners.lightning import BaseLightningModule
from utils.registry import DATALOADER_REGISTRY, MODEL_REGISTRY, OPTIMIZER_REGISTRY, LOSSES_REGISTRY


def run():
    parser = argparse.ArgumentParser(description="Script to train model from a yaml config file")
    parser.add_argument("yaml_path", help="Path to yaml. Should be inside 'src/configs'")
    parser.add_argument("--log_path", help="Path to log output file")

    args = parser.parse_args()

    # initializing logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=args.log_path if args.log_path else None,
    )

    # loading config file
    logging.info(f"Reading model config from {args.yaml_path}")
    cfg = load_config(args.yaml_path)

    # load model, dataloader, optimizer, loss function
    model_cls = MODEL_REGISTRY[cfg["model"]["name"]]
    model = model_cls(cfg["model"])

    dataloader_cls = DATALOADER_REGISTRY[cfg["dataloader"]["name"]]
    train_loader = dataloader_cls(cfg["dataloader"]["train"])
    val_loader = dataloader_cls(cfg["dataloader"]["val"])
    test_loader = dataloader_cls(cfg["dataloader"]["test"])

    optimizer_fn = OPTIMIZER_REGISTRY[cfg["optimizer"]["name"]](cfg["optimizer"])

    loss_fn = LOSSES_REGISTRY[cfg["loss"]["name"]]

    # initialize lightning module
    lightning_module = BaseLightningModule(model, loss_fn, optimizer_fn)

    # train and test the model
    trainer = L.Trainer(
        min_epochs=cfg["trainer"]["min_epochs"],
        max_epochs=cfg["trainer"]["max_epochs"],
        log_every_n_steps=cfg["trainer"]["log_every_n_steps"],
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
    )
    logging.info("Started model training")
    trainer.fit(model=lightning_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    logging.info("Started model evaluation")
    trainer.test(model=lightning_module, dataloaders=test_loader)
