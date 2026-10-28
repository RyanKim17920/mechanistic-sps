# Paper pipeline and analyses

## Rebuilding the paper (CPU, about a minute)

```
make paper PYTHON=.venv/bin/python
```

runs, from the results snapshotted in `results/`:

1. `make_tables.py` writes every table of the paper and `paper/tables/numbers.tex`, the
   macros for the numbers stated in the text (`paper_numbers.py`), to `paper/tables/`.
2. `paper_figures.py` writes the figures to `paper/figures/`.

A fresh run reproduces the committed tables and figures byte for byte (gate G1).

| File | Role |
|---|---|
| `paper_manifest.yaml` | The paper's models: labels and table names, style, seed runs, the runs the code refers to by role, and which table or figure shows which run, plus the run lists of the analysis scripts outside the paper generators (`trajectory`, `lesion_compare`, `standard_compare`). No analysis script types a run name in code; usage examples in docstrings aside, the other run names in the repo are the Hydra configs themselves and the fixtures of `gates/` and the tests. |
| `paper_data.py` | Data layer: ledger and the seed rule, result JSONs, read-reach / read-depth / probe readers, the compute frontier, architecture stats. |
| `results/arch_stats.json` | FLOPs, parameters and config facts of every arm, from its Hydra config. Rebuild after a config or FLOP-accounting change: `python scripts/analysis/paper_data.py --refresh-arch-stats`. |
| `make_tables.py`, `paper_numbers.py` | Tables and text numbers; the claims the text makes about them are assertions. Numbers still typed in the paper text, and why, are listed in `results/SOURCES.md`. |
| `paper_figures.py`, `fig_arch_paper.py`, `paper_appendix/` | Figures; axis limits are asserted to contain the data. |
| `src/plotting/paper_style.py` | Colours, markers, fonts, and the arm table loaded from the manifest. |

The seed rule: every reported mean uses the first two runs of an arm (in the manifest's
mean-seed order) that have the data; further seeds appear only in the per-run tables.

## Paths

`src/repo_paths.py` is the one place machine-specific paths live. It reads them from the
environment or `<repo>/.env`; there are no machine-specific defaults, and the paper build
(`make paper`) needs none of them:

| Variable | Default | Meaning |
|---|---|---|
| `DUALSPS_DATA_ROOT` | required by the GPU analyses | tokenized corpus (`<root>/data/<dataset>/{train,val}.bin`) |
| `DUALSPS_OUT_ROOT` | `DUALSPS_DATA_ROOT` | run outputs (`<root>/out/<run>/*.pt`) |
| `DUALSPS_RESULTS` | `scripts/analysis/results` | the paper's inputs: ledger, wallclock, analysis JSONs |

`results/SOURCES.md` records where `ledger.jsonl` and `wallclock.jsonl` came from.

## The analyses behind the result JSONs

Each GPU analysis loads trained checkpoints (`common.load`, i.e. `scripts/dualsps/eval_runs.py`)
and writes `results/<analysis>_<run>[_<variant>].json`. Run them on a GPU node with a
node-local Triton cache, which `scripts/run/eval_shim.sh` creates (they assert its marker), e.g.
`scripts/run/eval_shim.sh scripts/analysis/a5_induction.py --run <run>`.

| Script | Writes | Used for |
|---|---|---|
| `a2_knockout.py` | `a2_knockout_*`, `a2_ctx_knockout_*` (`--standard-ctx`) | read reach: Fig. state-access (a), Tab. readdepth_full |
| `a3_distance_nll.py` | `a3_distance_nll_*` | loss by repeat distance (input of g3) |
| `a5_induction.py` | `a5_induction_*` | previous-token mass, induction lift: Tab. circuit(-full) |
| `a8_headpatch.py` | `a8_headpatch_*` | head selection reused by a9 |
| `a9_patching.py` | `a9_patching_*` (`--ctrl-mode`, `--n-ctrl-draws 100`: the controls) | head ablation, controls, patching, the induction sweep: Tabs. circuit, control-counts, patching, token-gain; Fig. app-ctrl-dist |
| `a10_pathtrace.py` | `a10_pathtrace_*` | path tracing (App. patching) |
| `a12_depth_lesion.py` | `a12_depth_lesion_*` (`--ablation mean`: `_mean`) | whole-layer lesions: Figs. read-depth (b, c), gain-diag (b) |
| `a13_read_depth_cap.py` | `a13_read_depth_cap_*` (`--impose-map post`: `_impose_post`) | read-depth caps and floors: Fig. read-depth (a), Tab. readdepth_full |
| `a16_grad_orthogonality.py` | `a16_grad_orthogonality_*` (`--untrained`: `_init`) | gradient cosines: Fig. app-gradcos |
| `a17_trajectory.py` | `a17/a17_trajectory_part{A,B}_*` | gradient cosines along the checkpoint ladder (part B: App. grad) |
| `a18_read_edge_grad.py` | `a18_read_edge_grad_*` | read-edge gradients (App. grad) |
| `a20_role_probe.py` | `a20_role_probe_*` (`--untrained`: `_init`) | role probes: Figs. state-access (b), gain-diag (a), Tab. probe_floors |
| `a21_emb_divergence.py` | `a21_emb_divergence_*` | prediction-table replacement (App. archabl) |

Most analyses read only the final checkpoints (`hf_export.py download <run>`, from
[`ryankim17920/mechanistic-sps`](https://huggingface.co/ryankim17920/mechanistic-sps)). Two also
read earlier checkpoints, which are in
[`ryankim17920/mechanistic-sps-extra`](https://huggingface.co/ryankim17920/mechanistic-sps-extra)
(`hf_export.py download --extra <run>`; the exact commands are in `results/SOURCES.md`):
`a17_trajectory.py partB` reads the weights-only ladder at 1, 2, 3, 4, 6, 8, 12 and 16B tokens of
its two arms, and `a21_emb_divergence.py` reads the three embedding tables of every checkpoint of
Sequential 6+6 (seeds 1, 2), extracted into one `emb_ladder.pt` per run. a17 part A (arms
`untied`, `seq6`, `single`) needs full ladders that were not kept; its committed results are
the record.

Three CPU scripts re-aggregate committed JSONs and reproduce their outputs byte for byte:
`g3_final_read_mechanism.py` (repeat-distance gain shares and probe gaps: Tabs. finalread,
token-gain), `a12_compare_ablation.py` (`a12_depth_lesion_zero_vs_mean.json`: Tab.
lesion_mean) and `g4_standard_compare.py` (Transformer comparisons, App. readreach).

`common.py` holds what the GPU analyses share: the model registry, checkpoint loading, the
attention views of each family, sampling constants and the noise floors recorded with every
result.
