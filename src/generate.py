"""Generate paired, action-conditioned MemoryS11 prediction data."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import sys

import gymnasium as gym
import minigrid
from minigrid.core.actions import Actions


MEMORY_ENV = "MiniGrid-MemoryS11-v0"
RECENT_WINDOW = 4
DWELL_START_T = 8
POLICY_VERSION = "memory-fixed-cue-safe-dwell-v3"


def memory_prefix(seed):
    """Start facing the cue; wait at x=8, beyond its visibility range."""
    dwell_turns = 8 * (seed % 3)
    return (
        [int(Actions.right)]
        + [int(Actions.forward)] * 7
        + [int(Actions.left)] * dwell_turns
        + [int(Actions.forward)]
    )

BRANCHES = {
    "left": [int(Actions.left), int(Actions.forward)],
    "right": [int(Actions.right), int(Actions.forward)],
}


def observed(obs):
    """Only the partial symbolic observation that a model may consume."""
    return {"image": obs["image"].tolist(), "direction": int(obs["direction"])}


def reset(env_id, seed):
    """Create a seeded environment and return its initial partial observation.

    MemoryS11 starts at a fixed position facing the cue; object identities
    and candidate positions still follow the seed."""
    env = gym.make(env_id)
    obs, _ = env.reset(seed=seed)
    if env_id == MEMORY_ENV:
        # Controlled start: the cue is directly north of the agent.
        env.unwrapped.agent_pos = (1, 5)
        env.unwrapped.agent_dir = 3
        obs = env.unwrapped.gen_obs()
    return env, observed(obs)


def step(env, action):
    """Execute one action and return observation, reward, and both end flags.

    Convert NumPy values to plain Python values for JSON storage."""
    obs, reward, terminated, truncated, _ = env.step(action)
    return observed(obs), float(reward), bool(terminated), bool(truncated)


def branch_outcome(seed, prefix, actions):
    """Replay the same prefix and one candidate branch to obtain its label.

    A positive terminal reward yields 1; the unsuccessful branch yields 0."""
    env, _ = reset(MEMORY_ENV, seed)
    try:
        for action in prefix:
            _, _, terminated, truncated = step(env, action)
            if terminated or truncated:
                raise ValueError("memory prefix ended early")
        for action in actions:
            _, reward, terminated, truncated = step(env, action)
        if not terminated or truncated:
            raise ValueError("branch did not reach a terminal choice")
        return int(reward > 0)
    finally:
        env.close()


def cue_visible(env, obs):
    """Track the starting cue itself, not matching objects at the fork."""
    raw = env.unwrapped
    cue_x, cue_y = 1, 4
    x, y = raw.get_view_coords(cue_x, cue_y)
    image = obs["image"]
    return bool(
        0 <= x < len(image) and 0 <= y < len(image[x])
        and image[x][y][0] == raw.grid.get(cue_x, cue_y).encode()[0]
    )


def last_cue_time(visibility):
    """Return the last observation index where the starting cue was visible.

    Require an initially visible cue and no reappearance during waiting."""
    if not visibility[0]:
        raise ValueError("initial cue is absent")
    if any(visibility[DWELL_START_T:]):
        raise ValueError("cue reappears during the delay or at the query")
    return max(t for t, visible in enumerate(visibility) if visible)


def collect_memory(seed, split):
    """Record one prefix trajectory and attach its two candidate questions.

    Store observations and transition records, plus visibility metadata.
    Candidate outcomes come from separate replays of the same seeded world."""
    prefix = memory_prefix(seed)
    env, initial = reset(MEMORY_ENV, seed)
    observations = [initial]
    visibility = [cue_visible(env, initial)]
    rewards, terminated_flags, truncated_flags = [], [], []
    try:
        for action in prefix:
            obs, reward, terminated, truncated = step(env, action)
            if terminated or truncated:
                raise ValueError(f"memory seed {seed} ended before query")
            observations.append(obs)
            visibility.append(cue_visible(env, obs))
            rewards.append(reward)
            terminated_flags.append(terminated)
            truncated_flags.append(truncated)
    finally:
        env.close()
    cue_t = last_cue_time(visibility)
    t = len(prefix)
    if t - cue_t <= RECENT_WINDOW:
        raise ValueError("cue is inside the recent window")
    queries = [
        {
            "query_t": t,
            "cue_t": cue_t,
            "candidate": name,
            "future_actions": actions,
            "target_success": branch_outcome(seed, prefix, actions),
        }
        for name, actions in BRANCHES.items()
    ]
    if sorted(query["target_success"] for query in queries) != [0, 1]:
        raise ValueError(f"memory seed {seed} has unbalanced branches")
    return {
        "id": f"memory-{seed}", "environment": MEMORY_ENV, "seed": seed,
        "split": split, "observations": observations, "actions": prefix,
        "rewards": rewards, "terminated": terminated_flags,
        "truncated": truncated_flags, "cue_visibility": visibility, "queries": queries,
    }


def split_for_seed(index, count):
    """Assign an episode index to train, validation, or test (80/10/10).

    Integer boundaries round down; all records for an episode stay together."""
    train_end = (count * 8) // 10
    validation_end = (count * 9) // 10
    if index < train_end:
        return "train"
    if index < validation_end:
        return "validation"
    return "test"


def save_gif(episode, path):
    """Replay one saved trajectory, then show each candidate branch separately."""
    from PIL import Image, ImageDraw, ImageFont

    frames, durations = [], []
    font = ImageFont.load_default(size=16)

    def add_frame(env, caption, duration=400):
        """Append a global-view frame with local visibility and a caption."""
        grid = Image.fromarray(env.unwrapped.get_frame(highlight=True, tile_size=40))
        frame = Image.new("RGB", (grid.width, grid.height + 80), "#18212b")
        frame.paste(grid, (0, 80))
        draw = ImageDraw.Draw(frame)
        draw.text((10, 8), f"{episode['id']} / {episode['split']}", font=font, fill="white")
        draw.text((10, 32), caption, font=font, fill="white")
        draw.text((10, 56), "Global view; highlighted cells are visible", font=font, fill="white")
        frames.append(frame)
        durations.append(duration)

    env, _ = reset(MEMORY_ENV, episode["seed"])
    try:
        add_frame(env, "Start facing the cue", 1200)
        for t, action in enumerate(episode["actions"], 1):
            step(env, action)
            visibility = "cue visible" if episode["cue_visibility"][t] else "cue out of view"
            add_frame(env, f"Step {t}: {Actions(action).name}; {visibility}")
        durations[-1] = 1200
    finally:
        env.close()
    for query in episode["queries"]:
        env, _ = reset(MEMORY_ENV, episode["seed"])
        try:
            for action in episode["actions"]:
                step(env, action)
            add_frame(env, f"Separate replay: {query['candidate']} branch", 1000)
            for action in query["future_actions"]:
                _, reward, terminated, _ = step(env, action)
                result = "success" if reward > 0 else "failure"
                caption = result if terminated else Actions(action).name
                add_frame(env, f"{query['candidate']} branch: {caption}", 1500 if terminated else 400)
        finally:
            env.close()
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0)


def write_dataset(path, episodes_per_env, seed_start, gifs=0):
    """Generate episodes into a new directory and optionally render random GIFs.

    Write one compressed JSONL file per split and a manifest describing
    the collection. GIF sampling spans all splits without replacement and
    is reproducible from seed_start; it does not alter episode generation."""
    if episodes_per_env < 10:
        raise ValueError("episodes-per-env must be at least 10 for nonempty splits")
    eligible_count = episodes_per_env
    if not 0 <= gifs <= eligible_count:
        raise ValueError(f"gifs must be between 0 and {eligible_count} (all splits)")
    # A dedicated RNG makes previews reproducible without affecting trajectories.
    gif_indices = set(random.Random(seed_start).sample(range(eligible_count), gifs))
    previews = []
    path.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "policy_version": POLICY_VERSION,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "recent_window": RECENT_WINDOW,
        "seed_start": seed_start,
        "episodes_per_env": episodes_per_env,
        "environments": [MEMORY_ENV],
        "versions": {
            "python": sys.version.split()[0],
            "gymnasium": gym.__version__,
            "minigrid": minigrid.__version__,
        },
        "files": {split: f"{split}.jsonl.gz" for split in ("train", "validation", "test")},
    }
    handles = {
        split: gzip.open(path / filename, "wt", encoding="utf-8")
        for split, filename in manifest["files"].items()
    }
    try:
        for index in range(episodes_per_env):
            seed = seed_start + index
            split = split_for_seed(index, episodes_per_env)
            episode = collect_memory(seed, split)
            if index in gif_indices:
                previews.append(episode)
            handles[split].write(json.dumps(episode, separators=(",", ":")) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    manifest["gifs"] = []
    if previews:
        (path / "gifs").mkdir()
        for episode in previews:
            filename = f"gifs/{episode['split']}-{episode['id']}.gif"
            save_gif(episode, path / filename)
            manifest["gifs"].append(filename)
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    """Read command-line options and generate one MemoryS11 dataset."""
    # This script only generates a dataset; there are no subcommands.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-env", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=1000)
    parser.add_argument("--gifs", type=int, default=0,
                        help="number of random train/validation/test trajectories to render as GIFs")
    args = parser.parse_args()
    write_dataset(args.output, args.episodes_per_env, args.seed_start, args.gifs)
    print(f"Saved {args.episodes_per_env} MemoryS11 episodes to {args.output}")


if __name__ == "__main__":
    main()
