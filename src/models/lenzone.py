"""First len(zone) prototype: bounded slots with soft routing and gated writes."""

import math

import torch
from torch import nn

from ..data import OBSERVATION_SIZE, FUTURE_SIZE
from .base import BasePredictor


class LenzonePredictor(BasePredictor):
    """Selective writes, a no-write option and candidate-conditioned reading.

    This prototype does not implement allocation, consolidation or explicit forgetting.
    Each example starts with empty slots; identities distinguish their routing keys.
    """

    def __init__(self, window=1, hidden=64, lr=0.001, slots=4, model="lenzone"):
        super().__init__()
        if slots < 1:
            raise ValueError("slots must be positive")
        self.save_hyperparameters()
        self.encoder = nn.Sequential(nn.Linear(OBSERVATION_SIZE, hidden), nn.ReLU())
        self.slot_ids = nn.Parameter(torch.randn(slots, hidden) / math.sqrt(hidden))
        self.write_query = nn.Linear(hidden, hidden)
        self.write_key = nn.Linear(hidden, hidden)
        self.no_write = nn.Linear(hidden, 1)
        self.update = nn.GRUCell(hidden, hidden)
        self.read_query = nn.Linear(FUTURE_SIZE, hidden)
        self.read_key = nn.Linear(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden + FUTURE_SIZE, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def encode_memory(self, observations, mask):
        """Write past observations/actions only into a [batch, slots, hidden] state."""
        encoded = self.encoder(observations)
        batch, _, hidden = encoded.shape
        state = encoded.new_zeros(batch, self.hparams.slots, hidden)
        for t in range(encoded.size(1)):
            view = encoded[:, t]
            keys = self.write_key(state + self.slot_ids)
            scores = (keys * self.write_query(view).unsqueeze(1)).sum(-1) / math.sqrt(hidden)
            # The final routing destination discards the write without changing any slot.
            routing = torch.cat([scores, self.no_write(view)], dim=1).softmax(dim=1)
            inputs = view.unsqueeze(1).expand_as(state).reshape(-1, hidden)
            proposed = self.update(inputs, state.reshape(-1, hidden)).reshape_as(state)
            weight = routing[:, :-1].unsqueeze(-1)
            updated = state + weight * (proposed - state)
            state = torch.where(mask[:, t].bool().view(batch, 1, 1), updated, state)
        return state

    def forward(self, observations, mask, future_actions):
        """Candidate actions choose which slots to read, never what to write."""
        state = self.encode_memory(observations, mask)
        keys = self.read_key(state + self.slot_ids)
        query = self.read_query(future_actions).unsqueeze(1)
        scores = (keys * query).sum(-1) / math.sqrt(self.hparams.hidden)
        weights = scores.softmax(dim=1).unsqueeze(-1)
        summary = (weights * state).sum(dim=1)
        return self.head(torch.cat([summary, future_actions], dim=1)).squeeze(-1)
