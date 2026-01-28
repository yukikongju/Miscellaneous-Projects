"""
File which contains PyTorch Trainer
"""

import logging
import torch
from tqdm.auto import tqdm
from schemas.trainers.torch_trainer import TorchTrainerConfig


class TorchTrainerModule:

    def __init__(self, cfg: TorchTrainerConfig):
        self.num_epochs = cfg["num_epochs"]
        self.device = cfg["device"]
        self.log_every_n_steps = cfg["log_every_n_steps"]
        self.logger = None  # TODO

    def _train_step(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
    ):
        model.train()

        metrics = {"loss": 0}

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # compute prediction
            y_pred = model(X)

            # compute loss
            loss = loss_fn(y_pred, y)
            metrics["loss"] += loss.item()

            # back prop
            loss.backward()

            # optimizer
            optimizer.step()
            optimizer.zero_grad()

        metrics["loss"] = metrics["loss"] / len(dataloader)

        return metrics

    def _val_step(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
    ):
        model.eval()

        metrics = {"loss": 0}

        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)

                # compute prediction
                y_pred = model(X)

                # compute loss
                loss = loss_fn(y_pred, y)
                metrics["loss"] += loss.item()

        metrics["loss"] = metrics["loss"] / len(dataloader)

        return metrics

    def train(
        self,
        model: torch.nn.Module,
        train_dataloaders: torch.utils.data.DataLoader,
        val_dataloaders: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
    ):

        model.to(self.device)

        for epoch in tqdm(range(self.num_epochs)):
            logging.info(f"Epoch {epoch} - Training Step")
            train_metrics = self._train_step(
                model=model, dataloader=train_dataloaders, optimizer=optimizer, loss_fn=loss_fn
            )

            logging.info(f"Epoch {epoch} - Validation Step")
            val_metrics = self._val_step(model=model, dataloader=val_dataloaders, loss_fn=loss_fn)

            if epoch % self.log_every_n_steps == 0 and self.logger:  # TODO: log metrics, model, ...
                pass

    def test(self, model: torch.nn.Module, dataloaders: torch.utils.data.DataLoader):
        pass

    def predict(self, model: torch.nn.Module, dataloaders: torch.utils.data.DataLoader):
        pass
