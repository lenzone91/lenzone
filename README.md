# len(zone)

**Learning bounded latent memory states for long-horizon prediction.**

len(zone) is a research project exploring whether a fixed-size, persistent memory can retain the parts of a growing observation history that matter for future predictions. Its proposed memory is a set of latent slots. New evidence is routed to relevant slots, which can be updated or allocated; consolidation and forgetting keep the state within a fixed capacity.

The central question is empirical: under the same memory and compute budgets, does this learned state preserve useful information better than simpler ways of handling long histories?

## Research directions

- **Language models:** retain a short exact context and expose older information through learned latent memory tokens. Compare against full context where feasible, a rolling window, selective retention/pruning, and text summaries.
- **Partially observed world models:** maintain state across observations and actions, then predict future observations or hidden task-relevant variables. Start with controlled navigation tasks and compare against recent-observation windows and recurrent states.

Experiments should use the same data splits, tasks, backbone, recent-context size, and measured memory budget wherever the methods permit. Report predictive quality alongside GPU memory, runtime, and performance as the gap between evidence and query grows. An unconstrained full-history run can serve as a reference, but is not a budget-matched baseline.

## Local environment

The [Conda environment specification](env.yaml) pins the current direct dependencies: Python 3.14, CUDA-enabled PyTorch 2.14.0, Lightning 2.6.6, MiniGrid 3.1.0, and Pillow 12.3.0 for GIF previews. It targets Linux x86_64 with an NVIDIA driver supporting CUDA 13.2. It does not include model weights or datasets.

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

## Memory predictors

The training and data modules live under `src/`, and model implementations under `src/models/`; run commands from the repository root. No package installation is needed.

| Module | Role |
| --- | --- |
| `generate.py` | Generate MemoryS11 episodes and optional GIFs |
| `data.py` | Load one split and build past-only candidate examples |
| `models/` | One model per file (`gru.py`, `lenzone.py`); shared Lightning loss and metrics in `base.py` |
| `train.py` | Lightning Trainer and validation checkpoint selection |
| `evaluate.py` | NLL and accuracy overall and by retention delay |

Each episode provides two prediction examples. Inputs contain the last `N` views configured with `--window N`, their preceding actions, and the two proposed future actions. The default is 1 (the current view). Any positive integer is accepted, including 32 or 100; shorter histories are padded with zeros and masked. Object, color, state, direction, and action codes are one-hot encoded. Visibility metadata, cue times, rewards, and labels are excluded from model inputs. Labels are used only for the training loss and evaluation. Each view passes through the existing linear encoder and ReLU, then a GRUCell reads encoded views in chronological order. Masked padding leaves the state unchanged. The final state is concatenated with candidate future actions and passed to the prediction head. The state starts at zero for each example on every forward call; it is not carried between batches. `--window` limits the available history, while `--hidden` sets the encoder and GRU state width.

```bash
python -m src.train --dataset runs/minigrid/smoke --output runs/models/window1 --window 1
python -m src.train --dataset runs/minigrid/smoke --output runs/models/window32 --window 32
python -m src.evaluate --dataset runs/minigrid/smoke --checkpoint runs/models/window1/best.ckpt --output runs/models/window1/test.json
```

Training saves a Lightning `best.ckpt` selected by validation NLL and `training.json` with configuration, parameter count, and epoch metrics. Evaluation uses `Trainer.test()` and defaults to the test split, automatically using the window size saved in the checkpoint. Its JSON report includes `window`. **New checkpoints are required:** weights from the previous flattened-history predictor (`.pt` or `.ckpt`) are incompatible with the GRU architecture. Keep previous run directories and train into a new output directory. The default device is CPU; add `--device cuda` for GPU execution. Lightning manages epochs, batches, gradients, and device transfers. `Predictor` defines the loss in `training_step()` and Adam in `configure_optimizers()`. `--lr` sets a fixed learning rate for the entire training run. The learning rate used for each epoch appears in the console and as `lr` in the `training.json` history; a small callback keeps the JSON epoch history. `--epochs`, `--batch-size`, `--hidden`, `--lr`, and `--seed` keep their existing meanings. Training output directories must be new unless `--overwrite` is supplied. **`--overwrite` deletes all previous results in the output directory before recreating it.** For example: `python -m src.train --dataset runs/minigrid/smoke --output runs/models/window32 --window 32 --overwrite`. The project, current working directory, dataset, and their parent directories are protected from deletion. These first models predict candidate success; they do not choose the agent's exploration actions.

The 30-episode smoke dataset is for execution checks. Use larger datasets and multiple training seeds before drawing conclusions about memory. Runs with different window sizes share architecture choices, not trained weights. At fixed `--hidden`, parameter counts are independent of window size, but longer windows require more computation and activation memory; measured memory/compute matching remains future work.

### Lenzone slot prototype

Select `--model gru` (default) or `--model lenzone`. Both use the same dataset, binary prediction loss, Adam optimizer, validation selection and metrics. Evaluation reads the model name and architecture from the checkpoint; older GRU checkpoints without a model name still load.

```bash
python -m src.train --dataset runs/minigrid/dataset_1000 --output runs/models/lenzone_slots4 --model lenzone --window 32 --slots 4 --hidden 64 --batch-size 596
python -m src.evaluate --dataset runs/minigrid/dataset_1000 --checkpoint runs/models/lenzone_slots4/best.ckpt --output runs/models/lenzone_slots4/test.json
```

The first Lenzone prototype holds `--slots K` latent vectors of width `--hidden`. An observation-only router distributes each write between the slots and a no-write destination. A shared GRUCell proposes slot updates, weighted by this soft routing. Learned slot identities break symmetry. Candidate future actions attend to the final slots only when reading; they never enter the writer. Masked padding leaves the slots unchanged, and memory resets per example. This implements soft routing and selective updates, not the article's full allocation, consolidation, or explicit forgetting cycle. Training uses only the predictive loss.

At equal hidden width, Lenzone has K times as many recurrent state values as the GRU and additional routing/readout parameters. These settings are not matched memory or parameter budgets. A successful execution check does not establish predictive performance.

To add a model, subclass `BasePredictor` in a new file, implement `forward(observations, mask, future_actions)`, save constructor hyperparameters including `model`, and register the class in `models/__init__.py`.

## Current status

The [research draft](docs/len_zone_.pdf) describes the motivation, predictive-state formulation, proposed slot lifecycle, and initial learning objectives. It is a work in progress, not a validated result. The generator and a GRU predictor with a freely configurable observation window are implemented. The encoder, GRU, prediction head, and Lightning training setup stay the same across window sizes. A first soft-routed slot prototype is implemented; allocation, consolidation and explicit forgetting remain planned. Small execution checks are not benchmark results.

## Planned first steps

1. Define reproducible long-horizon tasks and budget-matched baselines in the [MiniGrid protocol](docs/minigrid_protocol.md).
2. Build a shared experiment runner and evaluation records.
3. Evaluate the slot prototype and extend its memory-management lifecycle.
4. Integrate memory tokens with a small local language model.
5. Run ablations for slot count, update policy, consolidation, and forgetting.

## License

The code, repository documentation, and [research draft](docs/len_zone_.pdf) are released under the [MIT License](LICENSE). Copyright (c) 2026 Enzo Cognéville.
