# len(zone)

**Learning bounded latent memory states for long-horizon prediction.**

len(zone) is a research project exploring whether a fixed-size, persistent memory can retain the parts of a growing observation history that matter for future predictions. Its proposed memory is a set of latent slots. New evidence is routed to relevant slots, which can be updated or allocated; consolidation and forgetting keep the state within a fixed capacity.

The central question is empirical: under the same memory and compute budgets, does this learned state preserve useful information better than simpler ways of handling long histories?

## Research directions

- **Language models:** retain a short exact context and expose older information through learned latent memory tokens. Compare against full context where feasible, a rolling window, selective retention/pruning, and text summaries.
- **Partially observed world models:** maintain state across observations and actions, then predict future observations or hidden task-relevant variables. Start with controlled navigation tasks and compare against recent-observation windows and recurrent states.

Experiments should use the same data splits, tasks, backbone, recent-context size, and measured memory budget wherever the methods permit. Report predictive quality alongside GPU memory, runtime, and performance as the gap between evidence and query grows. An unconstrained full-history run can serve as a reference, but is not a budget-matched baseline.

## Local environment

The [Conda environment specification](env.yaml) pins the current direct dependencies: Python 3.14, CUDA-enabled PyTorch 2.14.0, MiniGrid 3.1.0, and Pillow 12.3.0 for GIF previews. It targets Linux x86_64 with an NVIDIA driver supporting CUDA 13.2. It does not include model weights or datasets.

On this machine, create the environment with:

```bash
mkdir -p /home/enzo/temp
TMPDIR=/home/enzo/temp conda env create --file env.yaml
conda activate lenzone
```

`TMPDIR` points pip's large temporary downloads to `/home`; `/var/tmp` is on a smaller partition here. The specification lists direct project dependencies, not every transitive package already installed in the existing environment.

## First MiniGrid data milestone

The [trajectory generator](src/generate.py) implements the first step of the [MiniGrid protocol](docs/minigrid_protocol.md). It creates paired delayed-choice queries in MemoryS11. Generated datasets go under `runs/`, which is ignored by Git.

```bash
python -m src.generate --output runs/minigrid/my_first_run --episodes-per-env 30 --gifs 3
```

| Argument | Purpose | Default |
| --- | --- | --- |
| `--output` | New directory for the generated dataset | Required |
| `--episodes-per-env` | Number of MemoryS11 episodes (minimum 10) | 30 |
| `--seed-start` | First environment seed and seed for GIF sampling | 1000 |
| `--gifs` | Number of distinct episode GIFs (0 to episode count) | 0 |

Each run contains `manifest.json`, `train.jsonl.gz`, `validation.jsonl.gz`, and `test.jsonl.gz`, plus `gifs/` when previews are requested. Each JSONL line is one complete recorded prefix and its two candidate questions.

Use `--gifs N` to render N distinct episodes randomly selected from train, validation, and test together. The selection is reproducible from `--seed-start`; the default is 0 GIFs. Previews are saved under `gifs/` and listed in the manifest. Each shows the trajectory and both candidate branches as separate replays. The global map is for visual inspection only; the model data remains partially observed.

Choose an output directory that does not already exist. Only MemoryS11 is collected: the default run contains 30 episodes, split into 24 training, 3 validation, and 3 test episodes, with two candidate queries per episode. Datasets from the earlier collection policy must be regenerated in a new directory. There is no standalone audit or test suite; generation and training remain small, separate modules.

The agent starts facing the cue, moves away, and waits beyond the cue's visibility range before the branch query. Delays of 8, 16, and 24 steps are measured from the cue's last appearance, not just the initial observation.

## First predictive models

The Python modules live directly under `src/`; run commands from the repository root. No package installation is needed.

| Module | Role |
| --- | --- |
| `generate.py` | Generate MemoryS11 episodes and optional GIFs |
| `data.py` | Load one split and build past-only candidate examples |
| `models.py` | Symbolic observation encoder and success predictor |
| `train.py` | Shared training loop and validation checkpoint selection |
| `evaluate.py` | NLL and accuracy overall and by retention delay |

Each episode provides two prediction examples. Inputs contain the current view (`current`) or last four views (`window`), their preceding actions, and the two proposed future actions. Object, color, state, direction, and action codes are one-hot encoded. Visibility metadata, cue times, rewards, and labels are excluded from model inputs. Labels are used only for the training loss and evaluation.

```bash
python -m src.train --dataset runs/minigrid/smoke --output runs/models/current --model current
python -m src.train --dataset runs/minigrid/smoke --output runs/models/window --model window
python -m src.evaluate --dataset runs/minigrid/smoke --checkpoint runs/models/current/best.pt --output runs/models/current/test.json
```

Training saves `best.pt` selected by validation NLL and `training.json` with configuration, parameter count, and epoch metrics. Evaluation defaults to the test split. The default device is CPU; add `--device cuda` for GPU execution. Training output directories must be new. These first models predict candidate success; they do not choose the agent's exploration actions.

The 30-episode smoke dataset is for execution checks. Use larger datasets and multiple training seeds before drawing conclusions about memory. The two baselines share architecture choices, not trained weights, and the window model has a larger head; measured memory/compute matching remains future work.

## Current status

The [research draft](docs/len_zone_.pdf) describes the motivation, predictive-state formulation, proposed slot lifecycle, and initial learning objectives. It is a work in progress, not a validated result. The generator and two small predictors are implemented: current-view (`current`) and four-view (`window`) baselines. They share an encoder design and training loop; their prediction heads have different parameter counts. GRU and slot memory are still planned. Small execution checks are not benchmark results.

## Planned first steps

1. Define reproducible long-horizon tasks and budget-matched baselines in the [MiniGrid protocol](docs/minigrid_protocol.md).
2. Build a shared experiment runner and evaluation records.
3. Prototype the slot memory in a small partially observed world model.
4. Integrate memory tokens with a small local language model.
5. Run ablations for slot count, update policy, consolidation, and forgetting.

## License

The code, repository documentation, and [research draft](docs/len_zone_.pdf) are released under the [MIT License](LICENSE). Copyright (c) 2026 Enzo Cognéville.
