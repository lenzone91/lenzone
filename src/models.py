"""Small predictors with either the current view or a recent observation window."""

import torch
from torch import nn

from .data import OBSERVATION_SIZE, FUTURE_SIZE


class Predictor(nn.Module):
    """Encode past views, then predict success conditioned on candidate actions.

    window=1 is the no-persistent-memory baseline; window=4 retains four views.
    Both use the same encoder design. Window width changes the head's size.
    """

    def __init__(self, window=1, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(OBSERVATION_SIZE, hidden), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(window * hidden + FUTURE_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, observations, mask, future_actions):
        """Return logits; candidate actions never enter the observation encoder."""
        encoded = self.encoder(observations) * mask.unsqueeze(-1)
        features = torch.cat([encoded.flatten(1), future_actions], dim=1)
        return self.head(features).squeeze(-1)
