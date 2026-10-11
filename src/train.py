"""Train a predictor with a configurable observation window using Lightning."""

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from lightning.pytorch import Callback, Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from .data import MemoryDataset
from .models import MODELS


class TrainingHistory(Callback):
    """Keep the small JSON epoch report alongside Lightning checkpoints."""

    def __init__(self):
        self.records = []

    def on_train_epoch_end(self, trainer, pl_module):
        record = {"epoch": trainer.current_epoch + 1,
                  "lr": trainer.optimizers[0].param_groups[0]["lr"],
                  "train_nll": float(trainer.callback_metrics["train_nll"]),
                  "validation": pl_module.validation_metrics}
        self.records.append(record)
        print(f"Epoch {record['epoch']}: train NLL={record['train_nll']:.4f}, "
              f"validation NLL={record['validation']['overall']['nll']:.4f}, "
              f"lr={record['lr']:.6g}")


def main():
    """Train on train episodes and save the checkpoint with lowest validation NLL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="delete existing output results before training")
    parser.add_argument("--model", choices=MODELS, default="gru")
    parser.add_argument("--window", type=int, default=1,
                        help="number of recent views to use (positive integer)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--slots", type=int, default=4, help="number of memory slots (lenzone only)")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    if args.window < 1:
        parser.error("window must be positive")
    if args.batch_size < 1 or args.hidden < 1:
        parser.error("batch-size and hidden must be positive")
    if args.slots < 1:
        parser.error("slots must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("lr must be finite and positive")
    try:
        device = torch.device(args.device)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if args.overwrite:
        output = args.output.resolve()
        protected = (Path.cwd().resolve(), Path(__file__).resolve().parents[1],
                     args.dataset.resolve())
        if output == Path(output.anchor) or any(path.is_relative_to(output) for path in protected):
            parser.error("overwrite output must not contain the project, cwd, or dataset")
        if args.output.is_symlink():
            parser.error("overwrite output must not be a symbolic link")
        if args.output.exists():
            if not args.output.is_dir():
                parser.error("overwrite output must be a directory")
            shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed, workers=True)
    window = args.window
    train = DataLoader(MemoryDataset(args.dataset, "train", window),
                       batch_size=args.batch_size, shuffle=True)
    validation = DataLoader(MemoryDataset(args.dataset, "validation", window), batch_size=128)
    options = {"slots": args.slots} if args.model == "lenzone" else {}
    model = MODELS[args.model](window=window, hidden=args.hidden, lr=args.lr, **options)
    config = {**vars(args), "dataset": str(args.dataset), "output": str(args.output)}
    history = TrainingHistory()
    checkpoint = ModelCheckpoint(dirpath=args.output, filename="best", monitor="val_nll",
                                 mode="min", save_top_k=1, enable_version_counter=False)
    trainer = Trainer(max_epochs=args.epochs,
                      accelerator="gpu" if device.type == "cuda" else device.type,
                      devices=[device.index or 0] if device.type == "cuda" else 1,
                      callbacks=[history, checkpoint], logger=False, num_sanity_val_steps=0,
                      default_root_dir=args.output)
    trainer.fit(model, train, validation)
    report = {"config": config, "parameters": sum(p.numel() for p in model.parameters()),
              "history": history.records}
    (args.output / "training.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
