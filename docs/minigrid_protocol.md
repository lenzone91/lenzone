# MiniGrid measurement protocol (draft v0)

This document defines the first controlled test of len(zone) as a bounded predictive state in a partially observed environment. It is a protocol, not a result. Values marked *initial* can be changed before the first benchmark run, then must be recorded and held fixed across methods.

## Question

Can a fixed-size, recursively updated slot memory preserve information from early observations that improves future prediction after that information has left a short observation window?

The model receives a stream of observations and preceding actions. At a chosen time `t`, a reader receives its memory state and a proposed future action sequence. It predicts a future outcome. The memory writer must never receive the future actions, outcome, full map, or hidden environment state.

## Environments

1. **Initial memory test:** `MiniGrid-MemoryS11-v0`. The initial object must be remembered to identify the matching object at the end of a corridor. Evaluate at a fixed decision point before the final choice. The task is well suited to a binary candidate-action success prediction.
2. **Later stress test:** a custom seeded maze with multiple cues, distractors, revisits, and variable delays. Specify it only after the initial task exposes what the current benchmark fails to distinguish.

Use the symbolic `image` observation (object, color, state), `direction`, and action. Exclude RGB rendering and the full grid from model inputs. The `mission` may be included only if it is identical across all examples in a task. Keep hidden environment state solely for label generation, visibility checks, and optional GIF rendering.

## Data and split

- Generate trajectories from one versioned behavior policy. Start with an exploration policy that reaches the decision point reliably; random movement alone may generate too few useful examples. Record its code and version.
- Give **every method the same saved trajectories, actions, query times, and target labels**. Generate data once, then train and evaluate all methods from those files.
- Partition episodes by environment seed into train, validation, and test before generating examples. Never split different windows of one episode across partitions. Initial target: 80/10/10 percent, with counts determined by the episode count and split boundaries.
- Fix the test set and report at least three training seeds for learned methods. Record package versions, configuration, checkpoint hash, and hardware. Do not tune on test episodes.
- For each query, record the delay since the decisive cue and whether that cue is outside the recent window. Stratify results by delay; a single aggregate score can hide memory failure.

## Prediction tasks and metrics

**Primary: delayed outcome prediction in MemoryS11.** At the decision point, provide a candidate sequence of future actions without future observations. Predict whether it reaches the matching object. Score binary negative log-likelihood (NLL) and accuracy, reported by cue-to-query delay. Construct balanced correct and incorrect candidate actions in the test set. This measures predictive use of past evidence, rather than navigation-policy quality.

For all learned methods, report mean and uncertainty across training seeds, along with parameter count, state bytes per episode, peak GPU memory, training time, and prediction latency. The state-byte count includes occupancy flags and any other persistent per-episode data. Report temporary compute memory separately from persistent state.

## Comparisons

All learned predictors should share the observation encoder and prediction head where practical. Use the same training trajectories, optimizer budget, tuning procedure, and test queries. Record architectural differences that prevent exact parameter matching.

| Method | Persistent information after the recent window |
| --- | --- |
| No-memory control | None |
| Rolling window | Last `W` observations and actions |
| GRU state | One bounded recurrent vector |
| len(zone) | `K` persistent slots plus occupancy and management state |
| Full-history reference | Entire prefix; informative but not budget matched |

The initial recent window is `W = 4` steps (*initial*). Compare several state budgets after the pipeline works, for example 0.5, 2, and 8 KiB per episode (*initial*), adjusting vector width or slot count. Compare methods at equal **measured** persistent-state bytes and show quality-versus-budget curves. Include the cost of recent observations for every method. Do not claim a memory advantage from a method that also sees a larger recent window or stronger encoder.

## Current implementation

The code currently focuses on one trajectory generator for MemoryS11. It saves partial observations, actions, rewards, termination flags, paired branch labels, and a manifest in train/validation/test files. There is no standalone audit command or test suite at this stage. The generator keeps local checks for a visible initial cue, no cue reappearance during the waiting period, and one successful branch per episode.

Run the generator directly, without a subcommand:

```bash
python scripts/minigrid_data.py --output runs/minigrid/my_run --episodes-per-env 30 --seed-start 1000 --gifs 3
```

`--gifs N` selects N distinct episodes uniformly without replacement from all three splits. Selection uses a dedicated RNG initialized with `seed_start`, so previews are reproducible and do not affect trajectories. Zero GIFs is the default; the maximum is the total episode count. GIFs are saved in `gifs/` and listed in the manifest. They show the global map with the agent's local visibility highlighted, followed by separate replays of each candidate branch. The global render is not model input.

The next milestone is to implement the no-memory and rolling-window predictors before the recurrent and slot models.

## Fixed first collection policy and limitations

The MemoryS11 collection policy is fixed in `scripts/minigrid_data.py`. After the seeded reset, it places the agent at (1, 5), facing north toward the cue at (1, 4). Object identities and branch arrangements remain seeded. It turns east, advances seven times to x=8, optionally performs 0, 8, or 16 extra turns according to `seed % 3`, then advances once to x=9 before the branch choice. The starting cue is invisible from x=8 in all four orientations.

Generation tracks the starting cue by projecting its world position into each partial observation. This distinguishes it from the candidate objects at the fork. The initial observation must expose the cue, and the cue must remain invisible from the start of the waiting period through the query. `cue_t` records its last actual appearance (currently index 1), giving delays of 8, 16, and 24 steps. Visibility diagnostics and cue times describe the generated episode; future predictors must receive only past observations/actions and the candidate actions, excluding these diagnostics and target labels. The policy version changes so earlier datasets must be regenerated.

This is an initial controlled retention test: the extra turns vary the delay without introducing new facts or distractors. A later maze should test interference and multiple persistent facts. The saved episodes include target labels for training. The future training loader must slice observations at `query_t` and expose only past observations/actions and the candidate action sequence. No model-input loader is implemented yet.

## References

- [MiniGrid Memory environment](https://minigrid.farama.org/environments/minigrid/MemoryEnv/)
