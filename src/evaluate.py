"""Evaluate a saved predictor on held-out episodes, overall and by delay."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import MemoryDataset
from .models import Predictor


@torch.no_grad()
def score(model, loader, device):
    """Measure binary NLL and accuracy without changing model parameters."""
    model.eval()
    groups = {}
    for observations, mask, future, target, delays in loader:
        logits = model(observations.to(device), mask.to(device), future.to(device))
        target = target.to(device)
        losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none").cpu().tolist()
        correct = ((logits >= 0) == target.bool()).cpu().tolist()
        for loss, hit, delay in zip(losses, correct, delays.tolist()):
            for key in ("overall", str(delay)):
                totals = groups.setdefault(key, [0, 0., 0])
                totals[0] += 1
                totals[1] += loss
                totals[2] += int(hit)
    metrics = {key: {"examples": n, "nll": loss / n, "accuracy": hits / n}
               for key, (n, loss, hits) in groups.items()}
    return {"overall": metrics["overall"],
            "by_delay": {key: value for key, value in metrics.items() if key != "overall"}}


def main():
    """Load a checkpoint and report evaluation metrics to the screen and JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = Predictor(config["window"], config["hidden"]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"])
    dataset = MemoryDataset(args.dataset, args.split, config["window"])
    metrics = score(model, DataLoader(dataset, batch_size=128), args.device)
    report = {"split": args.split, "model": config["model"],
              "parameters": sum(p.numel() for p in model.parameters()), **metrics}
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
