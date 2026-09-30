"""Train either baseline with a shared loop and validation checkpoint selection."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import MemoryDataset
from .evaluate import score
from .models import Predictor


def main():
    """Train on train episodes and save the checkpoint with lowest validation NLL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=["current", "window"], default="current")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    window = 1 if args.model == "current" else 4
    train = DataLoader(MemoryDataset(args.dataset, "train", window),
                       batch_size=args.batch_size, shuffle=True)
    validation = DataLoader(MemoryDataset(args.dataset, "validation", window), batch_size=128)
    model = Predictor(window, args.hidden).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    config = {**vars(args), "dataset": str(args.dataset), "output": str(args.output), "window": window}
    history = []
    best_nll = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.
        for observations, mask, future, target, _ in train:
            optimizer.zero_grad()
            logits = model(observations.to(args.device), mask.to(args.device), future.to(args.device))
            loss = F.binary_cross_entropy_with_logits(logits, target.to(args.device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(target)
        metrics = score(model, validation, args.device)
        record = {"epoch": epoch, "train_nll": total_loss / len(train.dataset), "validation": metrics}
        history.append(record)
        nll = metrics["overall"]["nll"]
        print(f"Epoch {epoch}: train NLL={record['train_nll']:.4f}, validation NLL={nll:.4f}")
        if nll < best_nll:
            best_nll = nll
            torch.save({"config": config, "epoch": epoch, "state_dict": model.state_dict()},
                       args.output / "best.pt")
    report = {"config": config, "parameters": sum(p.numel() for p in model.parameters()), "history": history}
    (args.output / "training.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
