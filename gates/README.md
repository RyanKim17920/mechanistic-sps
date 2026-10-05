# Golden gates

`make check` runs the CPU gates G1–G5 and G7 in parallel and fails on any difference from
`gates/golden/`. The goldens were made from the research code the paper's results came
from, so a passing `make check` means this code gives the same configs, initial weights,
training steps, data order, figures and tables.

```
make check PYTHON=<path to venv>/bin/python     # all gates, ~3 min wall (~40 CPU-min)
python gates/run.py g2 g5                         # a subset
python gates/run.py g4 --arms two_tower,frz       # G4 on some arms
make check FRZ=real                               # frz_* from the trained source checkpoints
```

The gates need no data, checkpoint or environment variable. The freeze-and-retrain configs
(G3, and G4's `frz` arm) load a frozen state tower from a source run's final checkpoint. By
default (`--frz-source=synthetic`) the gate writes that checkpoint itself: the source
config's model, initialised with a seed derived from the source run's name, state-tower
tensors only. Those results are pinned under `<config>@synthetic` / `frz@synthetic`.
`--frz-source=real` (`make check FRZ=real`) instead reads the trained checkpoints under
`$DUALSPS_OUT_ROOT/out/<source run>/`.

`gates/run.py` puts this checkout's `src/` ahead of whatever the venv's editable install
points at, and checks where `modeling` and `plotting` were imported from.

| Gate | What it pins | Golden | Time |
|---|---|---|---|
| G1 | The paper pipeline: the sha256 of the snapshotted `ledger.jsonl` and `wallclock.jsonl` inputs, the 9 figures of `paper_figures.FIGS` as PDF, and the 13 `make_tables` tables plus `numbers.tex`, byte for byte (sha256). Also every number in every table, in order, and the multiset of every number the pipeline prints. Each regenerated figure and table must also equal its committed copy in `paper/{figures,tables}`. | `g1/sha256.json`, `g1/numbers.json` (`g1/printed.txt` is the stdout, for reading) | 90 s |
| G2 | The resolved Hydra config (`+experiment=<name>`, `system.data_root=/DATA_ROOT`) of all 39 paper and freeze-and-retrain configs. | `g2/<config>.yaml` | 6 s |
| G3 | The init `state_dict` sha256 of all 39 configs at full size on CPU, seeded as `train.py` seeds rank 0. For `frz_*` this includes loading and freezing the state tower from the source checkpoint (synthetic or real, see above). Also the parameter counts and `model_args`, the dataclass saved in every checkpoint. | `g3.json` | 115 s |
| G4 | `scripts/train.py` itself, run on CPU in fp32 through `gates/cpu_train.py`: 3 optimizer steps of 2 micro-batches. Steps 0–1 fall in LR warmup, and step 2 falls in the linear decay. It pins the exact grad norm of every step (float hex), the log lines (loss, lr, eval NLL), and the sha256 of the final checkpoint's model, optimizer state and CPU RNG, plus its sampler and resume fields. The models are full width. Reduced: `block_size=128`, micro-batch 2, global batch 4, synthetic seeded corpus, and eager flex attention (`flex_compile=false`, since compiled flex has no CPU lowering). There are 10 arms: Transformer, Two-tower 12+12, tied attention, Sequential 12+12, Sequential 6+6 (pause input), Shared Two-tower, Shared Sequential, AF-SPS, 12+6 and freeze-and-retrain. | `g4.json` | 110 s |
| G5 | The sampler stream at world size 8. For each distinct seed setting in the 39 configs, it covers each rank's start offset (drawn by `training.sampler.fresh_start_offset`, the function `train.py` calls), stream length and first 64 indices, plus the resume path (every rank restores rank 0's saved offset, the legacy behaviour) at 1,000 and 300,000 samples seen. | `g5.json` | 13 s |
| G7 | pytest. A test recorded as passing in the golden, or a new test, must not fail. Recorded tests that no longer exist are reported, not failed. | `g7.json` | 75 s |

## Rules

- **Config differences** (G2, and `model.config.*` in G3's `model_args`) are allowed only
  when listed in `gates/ALLOWED_CONFIG_DIFFS.md`: `drop` for a dead key, `set` for an
  explicit pin or a replacement key.
- **Regenerating a golden** (`--update`) is allowed only when the change is intended and
  another gate proves it harmless. State the reason in the commit message.
- **Paths in goldens and snapshots.** The machine paths recorded in the result files, the
  ledger and wall-clock snapshots and the G2 goldens are written as `${DATA_ROOT}`,
  `${REPO}` and so on (`scripts/analysis/results/SOURCES.md`). No gate or generator reads
  them. G1 pins the sha256 of the two snapshots as they are committed.

## Host

G4 is bit-exact only for the same CPU type and thread count (4 threads per arm; the golden
came from an AMD EPYC Genoa CPU). G1 depends on the matplotlib version. `golden/host.json` records the fingerprint of the machine that made the goldens
(CPU model, G4 threads, torch, matplotlib, the serif font matplotlib resolves, pdfTeX and
pdftotext versions); every `--update` rewrites it. When G1 or G4 fails, `run.py` prints
each field in which this host differs, so a host difference is not mistaken for a code
change.

The goldens were made with torch 2.9.0 and matplotlib 3.10.7 (both pinned in `uv.lock`). The figure fonts ship with the repo in `src/plotting/fonts/`
(Nimbus Roman from URW base35 for text, and matplotlib's STIX fonts for mathtext), and
`src/plotting/paper_style.py` loads them from there. matplotlib embeds a figure's fonts in
the order of their file paths, so keeping them in one directory makes the figure bytes
independent of where the repo, the venv or any system font lives.

## Not covered on CPU

- **SPS training step.** SPS attention is a Triton kernel with no CPU path, so G4 has no SPS
  arm. G3 still pins SPS init, and G2 pins its config.
- **G5 and the sampler.** G5 builds `FixedRandomChunkDistributedSampler` and draws the
  start offsets with `fresh_start_offset`, the same calls `train.py` makes, but with the
  shipped corpus length and world size 8 rather than a real DDP launch.
- **Trained frozen sources.** The default `make check` covers the freeze-and-retrain code
  path with a synthetic source; only `FRZ=real` checks against the trained checkpoints.
- **GPU training.** `make smoke GPU=1` trains every model family for 3 steps on one GPU;
  "Smoke test" in the top-level README also shows a few-step run on a full node.
