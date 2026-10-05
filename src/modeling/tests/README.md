# Test Suite

```bash
pytest                                   # all CPU-safe tests (testpaths = src/modeling/tests)
pytest -m cuda                           # CUDA / Triton tests only
pytest --run-slow                        # include the slow diagnostics
pytest src/modeling/tests/core -v        # one directory
```

If the venv was created from another checkout, put this checkout's code first
(`PYTHONPATH=src:.`): the venv's editable install points at the checkout it was made in.

| Marker   | Meaning                                          |
|----------|--------------------------------------------------|
| `cuda`   | Requires CUDA GPU; auto-skipped when unavailable |
| `triton` | Requires Triton (implies CUDA); auto-skipped     |
| `slow`   | Long-running diagnostics; requires `--run-slow`  |

## Directory map

### `core/`

| File | What it tests |
|------|---------------|
| `test_masking.py` | `causal_mask` from `modeling/masking.py` |
| `test_document_idx.py` | `generate_document_idx`: no EOS, single/multiple EOS, boundaries, consecutive EOS, batching, left-padding |
| `test_masked_stats.py` | Distribution-stat helpers vs a torch reference, including survival under `torch.compile` with dynamic lengths |
| `test_tie_lm_head_baselines.py` | `tie_lm_head` on the standard and SPS models: tied by identity by default, exact parameter delta when untied |
| `test_checkpointing.py` | `CheckpointManager`: rolling vs named checkpoints, `ckpt` alias handling, save cadence / decay gating, and resolution order |
| `test_sampler_balance.py` | The fixed-random-chunk distributed sampler |
| `test_wandb_utils.py` | W&B dir resolution |

### `attention/triton/` (CUDA)

| File | What it tests |
|------|---------------|
| `test_full_attention.py` | Triton full-attention: CPU fallback parity and CUDA flex-attention parity |
| `test_full_flash_attention.py` | Triton document-masked causal kernel: forward/backward parity vs PyTorch, document masking, warp_specialize, dtypes |
| `test_full_long_context.py` | Long-context fwd/bwd parity for the full-attention kernel, incl. the T=4032 padding regression |
| `test_sps_attention.py` | Triton SPS sliding attention vs an SDPA reference: forward+backward, with and without document masking |
| `test_document_leakage.py` | NaN-poison cross-document leakage probe for both kernels |

### `models/`

| File | What it tests |
|------|---------------|
| `sps/test_dense_only.py` | SPS reads the `<predict>` (odd) slot logits; predict-slot interleaving; stats schema |
| `two_tower/test_two_tower_core.py` | Two-tower masks, read alignment, dead blocks, backend agreement, cross-tower sharing knobs |
| `shared/test_left_padding_invariance.py` | Logit invariance to left-padding for the standard model |

### `analysis/`

Tests of `scripts/analysis` and `src/plotting`.

## Shared utilities

- `conftest.py`: marker hooks (`cuda`, `triton`, `slow`) with auto-skip, and the `device` fixture.
- `_helpers.py`: the small standard-model factory and `forward_logits`.
