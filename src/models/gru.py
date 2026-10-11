"""GRU baseline with a masked, per-example recurrent state."""

import torch
from torch import nn

from ..data import OBSERVATION_SIZE, FUTURE_SIZE
from .base import BasePredictor


class GRUPredictor(BasePredictor):
    """Summarize past views before conditioning on candidate actions."""

    def __init__(self, window=1, hidden=64, lr=0.001, model="gru"):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(nn.Linear(OBSERVATION_SIZE, hidden), nn.ReLU())
        self.memory = nn.GRUCell(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + FUTURE_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, observations, mask, future_actions):
        """Return logits; candidate actions never enter the observation encoder."""
        encoded = self.encoder(observations)
        state = encoded.new_zeros(encoded.size(0), self.hparams.hidden)
        for t in range(encoded.size(1)):
            updated = self.memory(encoded[:, t], state)
            state = torch.where(mask[:, t].bool().unsqueeze(-1), updated, state)
        features = torch.cat([state, future_actions], dim=1)
        return self.head(features).squeeze(-1)

