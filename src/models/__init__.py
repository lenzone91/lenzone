"""Model selection and checkpoint loading for the shared benchmark."""

import torch

from .gru import GRUPredictor
from .lenzone import LenzonePredictor

MODELS = {"gru": GRUPredictor, "lenzone": LenzonePredictor}
# Preserve the old import and GRU checkpoint interface.
Predictor = GRUPredictor


def load_predictor(path):
    """Old checkpoints without a model name are GRU checkpoints."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["config"] if path.suffix == ".pt" else checkpoint["hyper_parameters"]
    model_class = MODELS[config.get("model", "gru")]
    if path.suffix == ".pt":
        options = {"slots": config.get("slots", 4)} if config.get("model") == "lenzone" else {}
        model = model_class(window=config["window"], hidden=config["hidden"],
                            lr=config.get("lr", 0.001), **options)
        model.load_state_dict(checkpoint["state_dict"])
        return model
    return model_class.load_from_checkpoint(path, map_location="cpu", weights_only=True)
