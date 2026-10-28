# Attention kernels

Two Triton attention kernels, both the Flash-Attention-2 tutorial kernel plus an edit to the
**score step**: the lines in `_attn_fwd_inner` between `qk = q·kᵀ` and the row max where a
mask is added to `qk`. The online-softmax core (`m_i`, `l_i`, rescaled `acc`), the numeric
safety conventions and the sequence padding are the same in both files.

| File                             | Entry point             | Used by  |
| -------------------------------- | ----------------------- | -------- |
| `triton_full_flash_attention.py` | `full_attention`        | Standard |
| `triton_sps_flash_attention.py`  | `sps_sliding_attention` | SPS      |

The two-tower model does not use either: it runs `flex_attention` (or SDPA / FlashAttention)
through `two_tower/attention.py`.

## The block that differs

`full` adds a single intra-document mask to plain causal attention:

```python
if HAS_DOCUMENT_MASK:
    same_doc = docs_q[:, None] == docs_k[None, :]
    qk += tl.where(same_doc, 0.0, -1.0e6)                # keep only same-document keys
```

`sps` keeps that document mask and adds a sliding window that the persistent key is exempt
from (simplified; the kernel also has an optional window on persistent keys, which no model
sets):

```python
if HAS_PREDICT_BIAS:
    rel = q_tok[:, None] - k_tok[None, :]                # query/key distance, in token pairs
    normal_bias = tl.where(rel > temporary_key_window, -1.0e6, 0.0)   # sliding window
    k_is_persistent = (offs_n_abs % 2) == 0              # persistent key = even state slot
    attn_bias = tl.where(k_is_persistent, 0.0, normal_bias)
    is_self = (offs_m[:, None] == offs_n_abs[None, :]) & k_is_persistent
    qk += tl.where(is_self, 0.0, attn_bias)              # persistent key always self-attends
```

The layout doubles the sequence: even slot = input token (state), odd slot = `<predict>`,
`tok = pos // 2`. Each pair has one persistent key (global context, window-exempt) and one
windowed `<predict>` slot whose hidden state feeds `lm_head`.

## Additional details

- **Numeric safety**: masks add a finite `-1e6` (never `-inf`) and `m_i` starts at `-1e9`, so
  a fully-masked row yields `0`, not `NaN`.
- **Sequence padding**: `N_CTX` (the doubled `2T` for SPS) is rounded up to a multiple of 128
  before launch. The flat `[B·H·N_CTX, D]` descriptor has no per-row bounds check, so an
  unaligned final block would write into the next head's rows, a cross-program race. Padded
  positions sit after all real tokens (masked out), padded query rows are sliced off the
  output, padded document ids use a `-1` sentinel, and the backward pads the incoming gradient
  and slices `dq`/`dk`/`dv` the same way. It is a no-op at aligned training lengths.
- **Backward**: recomputes the same position-derived mask instead of storing the score
  matrix, and returns only `dQ`/`dK`/`dV`.
- **Autotuning**: under pytest (`PYTEST_VERSION` set) each kernel uses one fixed config, so
  test results do not depend on the autotuner's timing.

## Tests

Under `src/modeling/tests/attention/triton/` (CUDA only). Each kernel is checked forward and
backward against an additive-mask PyTorch reference, plus NaN-poison tests for
cross-document leakage.
