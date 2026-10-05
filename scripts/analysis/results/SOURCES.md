# Provenance of the paper inputs

`scripts/analysis/results/` is the one results directory the paper is built from
(`repo_paths.RESULTS`, overridable with `DUALSPS_RESULTS`). The analysis JSONs are written
by the scripts in `scripts/analysis/` (see `scripts/analysis/README.md`). The two files
below are snapshots of the live files that the evaluation and the benchmark append to, so
the paper rebuilds from the repository alone:

| File | Snapshot of | Written by | Rows | sha256 |
|---|---|---|---|---|
| `ledger.jsonl` | `${DATA_ROOT}/results/ledger.jsonl` | `scripts/dualsps/eval_runs.py` (full validation sweep, one row per evaluation; the paper reads `val_nll_full_sweep`, and the last non-null row of a run wins). The snapshot keeps that row of each of the 61 runs and drops 9 superseded re-evaluations; every run counts toward `\RunsTotal` | 61 rows, 61 runs | `25aff586782e40125433a1a4f3dc0c42349b61d7cdcf4f85b4446770bee224ee` |
| `wallclock.jsonl` | `${DATA_ROOT}/results/wallclock.jsonl` | `scripts/bench_wallclock.py` (the paper reads the rows tagged `final_mb12`; the snapshot keeps only those and drops 67 probe, smoke and superseded-sweep rows) | 46 rows | `a4e4bc92d69322fb3d0a707bf5a8681b204e7693745a5cc34fcfb7045a40a2c0` |

Check a copy with `sha256sum scripts/analysis/results/{ledger,wallclock}.jsonl`. Gate G1
pins both. The live files also hold rows no generator reads (superseded re-evaluations, probe and
smoke benchmarks); `make snapshot` copies them back in, which changes the two pins but no
table, figure or number.

The `job_id` field of the rows is the batch-scheduler job id of the evaluation or benchmark
job; nothing reads it, and the current code no longer records it.

## Checkpoints behind the result files

The analysis JSONs were measured on the checkpoints in two public Hugging Face repos. No login
is needed; files land in `<DUALSPS_OUT_ROOT>/out/<run>/`, where the scripts look for them.

- [`ryankim17920/mechanistic-sps`](https://huggingface.co/ryankim17920/mechanistic-sps): the
  20B final of every paper run, weights only. Every analysis that reads a final:
  `python scripts/dualsps/hf_export.py download <run>`.
- [`ryankim17920/mechanistic-sps-extra`](https://huggingface.co/ryankim17920/mechanistic-sps-extra):
  what two analyses read beyond the finals.
  - `a17/a17_trajectory_partB_{tiedattn,afsps}_<B>B.json`: the weights-only checkpoints at 1, 2,
    3, 4, 6, 8, 12 and 16B tokens (`<run>/ladder/`) and the final of `s_two_tower_afsps_20b`,
    which the main repo does not hold. The 20B point of `tiedattn` is its final in the main repo.

    ```
    python scripts/dualsps/hf_export.py download --extra s_two_tower_w0_equal_tiedattn_20b s_two_tower_afsps_20b
    python scripts/dualsps/hf_export.py download s_two_tower_w0_equal_tiedattn_20b
    scripts/run/eval_shim.sh scripts/analysis/a17_trajectory.py partB --arm tiedattn
    scripts/run/eval_shim.sh scripts/analysis/a17_trajectory.py partB --arm afsps
    ```
  - `a21_emb_divergence_s_two_tower_seq6_20b{,_seed2}.json`: `emb_ladder.pt`, the state table,
    prediction table and LM head of each of the run's 21 checkpoints (every 1B tokens, the 18B
    pre-decay branch point and the final), extracted bit for bit. With only the final (or
    nothing) in the run directory, `a21_emb_divergence.py` reads the tables from it and records
    the same checkpoint paths, so the result is the one the full checkpoints give (checked on
    CPU for both runs: the JSON is identical to the one from the full checkpoints, and equals
    the committed GPU result except two kNN overlaps per run, in the fifth decimal). `--functional` also needs the final.

    ```
    python scripts/dualsps/hf_export.py download --extra s_two_tower_seq6_20b s_two_tower_seq6_20b_seed2
    python scripts/dualsps/hf_export.py download s_two_tower_seq6_20b s_two_tower_seq6_20b_seed2
    scripts/run/eval_shim.sh scripts/analysis/a21_emb_divergence.py --run s_two_tower_seq6_20b --functional
    scripts/run/eval_shim.sh scripts/analysis/a21_emb_divergence.py --run s_two_tower_seq6_20b_seed2 --functional
    ```

The other intermediate checkpoints were not kept. The loss-vs-compute curves are read from
`ledger.jsonl`, and the a17 part A results (full ladders of `untied`, `seq6`, `single`) stand as
committed.

## Paths and host names in the data files

The result files, the two snapshots and the freeze-and-retrain G2 goldens record where each
number was measured: checkpoint and `val.bin` paths, the working directory, cache
directories and the host. The machine-specific part of each is written as a placeholder:

| Placeholder | Stands for |
|---|---|
| `${DATA_ROOT}` | `DUALSPS_DATA_ROOT` of the machine that ran the job |
| `${REPO}` | the checkout the job ran from |
| `${SCRATCH}`, `${HOME}`, `/tmp/${USER}` | other per-user directories |
| `node-<k>`, `login-node` | the host the job ran on |

Only these strings differ from what the jobs wrote. Every number, key and list is unchanged,
which was checked field by field on the parsed JSON. No generator, gate or ledger code reads
any of these fields: all inputs are located through `repo_paths` and the `DUALSPS_*`
variables.

## Files that no generator reads

`make paper` and the three CPU re-aggregators (`g3_final_read_mechanism.py`,
`a12_compare_ablation.py`, `g4_standard_compare.py`) open every result file here except 32:

- **30 back a statement of the paper's appendix that is not a number.**
  - App. patching, "Shared-weight models and AF-SPS fail the exactness gate": the
    14 `a10_pathtrace_*` files (the shared arms and AF-SPS fail the gate; the other arms pass
    it and are the contrast).
  - App. AF-SPS, "Per-stream MLP lesions cost zero because the MLP is shared", and App.
    sharing, "Shared Two-tower's unread final state layer has zero lesion cost":
    `a12_depth_lesion_s_two_tower_afsps_faithful_20b.json`,
    `a12_depth_lesion_s_two_tower_w0_shared_20b.json`.
  - App. methods, imposing the read of Sequential 12+12 matches cap k=12 bit for bit:
    `a13_read_depth_cap_impose_post_*`.
  - App. methods and App. gradients, the untrained-init and untied-tensor controls and the
    read-edge gradients: `a16_grad_orthogonality_s_two_tower_w0_equal_20b_untied.json`,
    `a18_read_edge_grad_*`.
  - App. sharing, Shared Two-tower's final next-token probe "nearly matches Shared
    Sequential's": `a20_role_probe_s_two_tower_seq12_tied_20b{,_seed2}.json`.
- **2 are inputs of kept results.** `a10_pathtrace.py` selects its heads from the run's
  `a5_induction` result, so `a5_induction_s_two_tower_w0_s1152p3456_20b.json` and
  `a5_induction_s_two_tower_w0_shared_20b.json` stay with their `a10_pathtrace` files.

## Numbers typed in the paper text

Every result number in the text is a `numbers.tex` macro (`paper_numbers.py`), except the
ones below. Stated bounds such as `\LesionMLPBound` and `\LesionTotalsWithin` are macros
too, and `paper_numbers.py` asserts that the data satisfy them.

**Two confidence bounds use a double-rounding rule.** `paper_numbers.double_round` rounds
half-up to one extra decimal, then to the printed precision:

| Macro | Data | Printed (double rounding) | Single rounding |
|---|---|---|---|
| `\SeedSDLo` (seed-SD 95% CI, lower bound) | 0.001446 | 0.0015 | 0.0014 |
| `\GradTiedAttnCI` (lower bound) | 0.04247 | 0.043 | 0.042 |

**Printed values that differ from the committed data:**

- App. finalread, 12+12 next-token probe gap: layer 4 is printed as 0.887--1.019 (data:
  0.887--1.018), layer 12 as 0.006--0.009 (data: 0.005--0.009).
- App. finalread, 6+6 next-token probe gap: layer 1 is printed as 0.764--0.811 (data:
  0.763--0.811), layer 6 as $-0.007$ to 0.041 (data: $-0.006$ to 0.041).
- App. probes, "prediction reaches 3.5--3.7": the last-layer next-token probes of SPS,
  Two-tower 12+12 and Sequential 12+12 span 3.513--3.617.
- App. induction, natural-text mean-lift ranges "4.4--6.0 versus 2.1--3.6": the a5
  aggregates give 4.4--5.2 (prediction) and 2.1--3.5 (state) over the separated main models'
  seeds.
- App. gains, "versus 2.3--3.1\% in other repeat-distance buckets" (SPS against the
  Transformer, a3): computed like `\SPSNeverGainPctA/B`, the other buckets span 2.29--3.29%.

**The pre-decay comparison** (Sec. efficiency: 2.2$\times$ fewer training FLOPs, 2.11--2.25$\times$
per seed pair, 8.0--8.5B tokens, the Transformer's pre-decay NLL 2.988 as a two-seed mean):
`python scripts/analysis/predecay_match.py`, from `ledger.jsonl`. It prints each seed pair
(per-seed reference NLLs 2.9873 and 2.9890), the two-seed mean, the token range and the
ratio range; 2.2$\times$ is the mean ratio (2.18) rounded.

**Values from inputs that are not in this directory:**

- The sampler replay and resume figures (4.28%, 0.5%, 0.04%, 7.52B, 0.40B):
  `docs/REPRODUCIBILITY.md`.
- The SPS attention profile (212 of 406 ms, 3.7$\times$, 57 ms), which is not in
  `wallclock.jsonl`.
- "Only Transformer and SPS have incremental paths, both launch-bound" (App. efficiency),
  and the decode rows in `wallclock.jsonl`: measured with incremental-decode code that is
  not in this repository (`scripts/bench_wallclock.py` measures training throughput only).

**Approximate:** "patching recovery and the last state MLP vary by $\sim$2$\times$" (App.
seeds, Sequential 6+6).

**Constants rather than results:** the training settings (10% decay, AdamW betas, weight
decay 0.1, clipping 1.0), confidence and significance levels, probe sequence ranges, gate
tolerances ($\le$0.01, $\le$1.0), the head-tying bound (1%), the read levels 7--12 of
Two-tower 12+6, and AF-SPS's published gain (0.024).
