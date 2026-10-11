"""Evaluate a saved predictor on held-out episodes, overall and by delay."""

import argparse
import json
from pathlib import Path

import torch
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader

from .data import MemoryDataset
from .models import load_predictor


def main():
    """Load a checkpoint and report evaluation metrics to the screen and JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    model = load_predictor(args.checkpoint)
    dataset = MemoryDataset(args.dataset, args.split, model.hparams.window)
    device = torch.device(args.device)
    trainer = Trainer(accelerator="gpu" if device.type == "cuda" else device.type,
                      devices=[device.index or 0] if device.type == "cuda" else 1,
                      logger=False, enable_checkpointing=False)
    trainer.test(model, DataLoader(dataset, batch_size=128), verbose=False)
    report = {"model": model.hparams.model, "split": args.split, "window": model.hparams.window,
              "parameters": sum(p.numel() for p in model.parameters()), **model.test_metrics}
    if model.hparams.model == "lenzone":
        report["slots"] = model.hparams.slots
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
