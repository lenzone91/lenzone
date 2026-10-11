"""Shared Lightning loss, optimizer and evaluation for memory predictors."""

import torch
from lightning.pytorch import LightningModule
from torch.nn import functional as F


class BasePredictor(LightningModule):
    """Subclasses implement forward(observations, mask, future_actions)."""

    def configure_optimizers(self):
        """Use Adam with a fixed learning rate."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def training_step(self, batch, batch_idx):
        """Return the loss; Lightning handles backward and parameter updates."""
        observations, mask, future, target, _ = batch
        loss = F.binary_cross_entropy_with_logits(self(observations, mask, future), target)
        self.log("train_nll", loss, on_step=False, on_epoch=True, batch_size=len(target))
        return loss

    def on_validation_epoch_start(self):
        self._groups = {}

    def on_test_epoch_start(self):
        self._groups = {}

    def _score_step(self, batch):
        """Accumulate per-example NLL and accuracy overall and by cue delay."""
        observations, mask, future, target, delays = batch
        logits = self(observations, mask, future)
        losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none").tolist()
        correct = ((logits >= 0) == target.bool()).tolist()
        for loss, hit, delay in zip(losses, correct, delays.tolist()):
            for key in ("overall", str(delay)):
                totals = self._groups.setdefault(key, [0, 0., 0])
                totals[0] += 1
                totals[1] += loss
                totals[2] += int(hit)

    def validation_step(self, batch, batch_idx):
        self._score_step(batch)

    def test_step(self, batch, batch_idx):
        self._score_step(batch)

    def _metrics(self):
        metrics = {key: {"examples": n, "nll": loss / n, "accuracy": hits / n}
                   for key, (n, loss, hits) in self._groups.items()}
        return {"overall": metrics["overall"],
                "by_delay": {key: value for key, value in metrics.items() if key != "overall"}}

    def on_validation_epoch_end(self):
        self.validation_metrics = self._metrics()
        self.log("val_nll", self.validation_metrics["overall"]["nll"], prog_bar=True)

    def on_test_epoch_end(self):
        self.test_metrics = self._metrics()
