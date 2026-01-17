"""
This script contains the BaseLightning module used to train, validate and test
a model using the lightning library
"""

import lightning as L
import torch
import torch.nn as nn


class BaseLightningModule(L.LightningModule):

    def __init__(self, model: nn.Module, loss_fn, optimizer):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def training_step(self, batch, batch_idx):
        X, y = batch
        X, y = X.to(torch.float32), y.to(torch.float32)

        outputs = self.model(X)
        loss = self.loss_fn(outputs, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return self.optimizer(self.model.parameters())
