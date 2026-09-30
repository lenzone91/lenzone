"""Read saved episodes and prepare past-only candidate prediction examples."""

import gzip
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

# MiniGrid channels are categories, not continuous pixel intensities.
CHANNEL_SIZES = (11, 6, 3)
ACTION_COUNT = 7
OBSERVATION_SIZE = 7 * 7 * sum(CHANNEL_SIZES) + 4 + ACTION_COUNT
FUTURE_SIZE = 2 * ACTION_COUNT


def encode_observation(observation, preceding_action=None):
    """Encode object/color/state, direction and the action preceding this view."""
    image = torch.tensor(observation["image"], dtype=torch.long)
    cells = torch.cat([
        F.one_hot(image[..., channel], size)
        for channel, size in enumerate(CHANNEL_SIZES)
    ], dim=-1).flatten()
    direction = F.one_hot(torch.tensor(observation["direction"]), 4)
    action = torch.zeros(ACTION_COUNT)
    if preceding_action is not None:
        action[preceding_action] = 1
    return torch.cat([cells, direction, action]).float()


class MemoryDataset(Dataset):
    """One example per branch; labels and visibility metadata stay out of inputs."""

    def __init__(self, directory, split, window):
        self.examples = []
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        with gzip.open(directory / manifest["files"][split], "rt") as handle:
            for line in handle:
                episode = json.loads(line)
                for query in episode["queries"]:
                    t = query["query_t"]
                    indices = range(max(0, t - window + 1), t + 1)
                    history = [encode_observation(
                        episode["observations"][i],
                        episode["actions"][i - 1] if i else None,
                    ) for i in indices]
                    padding = [torch.zeros(OBSERVATION_SIZE)] * (window - len(history))
                    observations = torch.stack(padding + history)
                    mask = torch.tensor([0.] * len(padding) + [1.] * len(history))
                    future = F.one_hot(torch.tensor(query["future_actions"]), ACTION_COUNT).flatten().float()
                    target = torch.tensor(float(query["target_success"]))
                    delay = t - query["cue_t"]
                    self.examples.append((observations, mask, future, target, delay))

    def __len__(self):
        """Return the number of candidate questions."""
        return len(self.examples)

    def __getitem__(self, index):
        """Return history, padding mask, candidate actions, label and delay."""
        return self.examples[index]
