"""A9 -- causal activation patching of the induction circuit, plus the natural-text arm.

A5 shows two attention SIGNATURES (previous-token mass in memory, prefix matching in
readout).  A8 shows that deleting the memory previous-token head costs the readout head
its induction lift.  Neither is a CIRCUIT measurement: an ablation removes a head's
contribution to everything downstream, so "the matching head uses the prev-token head's
output as its key" is still an inference.  This script runs the intervention that decides it.

--------------------------------------------------------------------------------------
Part 1 -- interchange (activation) patching on a minimal-pair induction probe
--------------------------------------------------------------------------------------
The probe is A5's synthetic repeated-block context (same builder, same constants), with a
TWO-TOKEN counterfactual edit so that clean and corrupted differ only in WHICH slot of
the first copy carries the query's predecessor:

    clean      ... [ B ] ...................... [ B ] ...
                     ^ i   ^ i+1                  ^ i        <- query q = COPY_B + i
    corrupted  copy A has positions i and i' SWAPPED

  * query position q, its token B[i], and the answer token B[i+1] are untouched;
  * the answer token still sits at p_ans = COPY_A + i + 1, but its predecessor is now
    B[i'], so the induction key at p_ans no longer matches the query;
  * the query's predecessor token B[i] now sits at COPY_A + i', so a working induction
    circuit retrieves p_dis = COPY_A + i' + 1 instead, whose token is B[i'+1].

That makes the metric a clean logit difference with a named counterfactual,

    LD = logit[B[i+1]] - logit[B[i'+1]]   at position q,

which is large and positive in the clean run and is driven towards the distractor in the
corrupted run.  Nothing about the token inventory, the positions, or the sequence length
changes between the two runs -- exactly two tokens move.

The patch then writes the CLEAN memory previous-token head's attention output back into
the corrupted run, at the two edited KEY slots only (p_ans, p_dis).  If the circuit is
"memory prev-token head -> readout key -> retrieval", restoring that head's output at the
key slots restores the answer; if the prev-token head merely coexists with the matching
head, it cannot.

Recovery fraction, per arm:   (LD_arm - LD_corrupted) / (LD_clean - LD_corrupted).

Arms (>= 64 examples, 95% bootstrap CI on every one):
  clean / corrupted            the two endpoints of the metric
  prev1_keyslots               top-1 memory prev-token head, both key slots      <- headline
  prev_top3_keyslots           the three A8 prev-token heads, both key slots
  prev1_answer_slot            top-1 head, ONLY the answer's key slot
  ctrl_head_keyslots           CONTROL: same blocks, a low-prev-token-mass head (A8's
                               control heads), same slots -- "an unrelated head"
  ctrl_donor_prev1             CONTROL: the right head and the right slots, but the
                               injected activations come from a DIFFERENT example's clean
                               run -- "a same-block random donor".  This is the control
                               that separates "this head's content matters" from "any
                               plausible activation in that slot matters".
  ceiling_block_all_heads      every head of the top prev-token head's block, both slots
                               -- the attainable ceiling for a single-block key patch.

--------------------------------------------------------------------------------------
Part 2 -- the natural-text arm (A9-natural)
--------------------------------------------------------------------------------------
The synthetic probe is noiseless but artificial.  Part 2 repeats A8's ablation (zero the
memory previous-token heads, everywhere) and scores it on REAL validation text, split
into three token sets defined before any ablation is run:

  induction        exists u <= t - MIN_GAP with X[u] == X[t+1] and X[u-1] == X[t]
                   (the target's earlier copy is preceded by the current token -- an
                   actual induction opportunity at distance >= MIN_GAP)
  repeat_ctrl      the target token occurred >= MIN_GAP back but NEVER with the current
                   token as its predecessor -- a repeat that induction cannot serve
  all              every scored position

and, within `induction`, the subset where the CLEAN model's top-1 was already correct
(`induction_clean_top1`), which is the set the abstract's claim is about.  Reported per
set: n, clean/ablated top-1 accuracy, clean/ablated NLL, and the deltas.

Part 2 also runs A8's UNRELATED-HEAD control on real text (`control_acc` / `control_nll`
per set).  It is the same arm with one thing changed -- which heads are zeroed: the same
number of memory-stream heads, the same `zero` mechanism, the same validation sequences
and the same scored token sets, but with the same-block unrelated heads A8 already uses
substituted for the previous-token heads (see `control_heads_for_natural`).  Without it
the accuracy drop is not attributable to the previous-token heads specifically.

Control DISTRIBUTION: `--ctrl-mode {same_block_lowprev, random_any, pattern_matched}`,
`--n-ctrl-draws N`, `--ctrl-seed S` (selection is `a8_headpatch.select_controls`; the
defaults reproduce the historical single draw and the default output bit-for-bit).  A
non-default run writes `a9_patching_<run>_ctrl-<mode>-n<N>[-s<seed>].json`; draw 0 fills
the existing control keys, and every draw's Part-2 control arm (same sequences, same
token sets) goes to `ctrl_draws`, with `ctrl_summary` giving per set the previous-token
cost against the control mean / sd and the fraction of draws costing >= it (empirical
p).  Part 1's `ctrl_head_keyslots` arm uses draw 0 only.

Usage:
  a9_patching.py --run <run|role> [--out J] [--n-ex 96] [--nat-seq 16] [--no-natural]
                 [--ctrl-mode M] [--n-ctrl-draws N] [--ctrl-seed S] [--ctrl-match-k K]
                 [--ctrl-exclude-layers L ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

import common as C                                                     # noqa: E402
import a5_induction as A5                                              # noqa: E402
import a8_headpatch as A8                                              # noqa: E402

PAIR_SEED = 20260919
N_EX = 96              # >= 64 required by the design; every arm gets a bootstrap CI
MIN_SEP = 3            # |i - i'| >= MIN_SEP so the two edits cannot touch each other
MIN_GAP = 64           # natural-text induction opportunities must be >= 64 tokens back
BOOT = 10000
BOOT_SEED = 7


# --------------------------------------------------------------------------------------
# memory-stream head IO -- capture / inject / zero, for every family
# --------------------------------------------------------------------------------------
def mem_attn_sites(model, family):
    """-> [(block_index, c_proj module, n_head, head_dim)] for the MEMORY stream.

    Every family in this study ends its attention with `c_proj(y)` where `y` is the
    concatenation of the per-head attention outputs, so a forward PRE-hook on that
    Linear sees exactly the per-head outputs just before they enter the residual
    stream.  Editing head h's slice there is precisely "head h writes X" -- the same
    intervention point A8's `_attend` wrapper uses, reached through the one module every
    family has in common.
    """
    t = model.transformer
    if family == "two_tower":
        blocks = list(t.state_h)
        return [(i, b.c_proj, int(b.n_head), int(b.head_dim)) for i, b in enumerate(blocks)]
    if family == "standard":
        blocks = list(t.h)
    else:
        assert getattr(model, "route_swap_LxT", None) is None, (
            "random pair routing is on: slot 0 of the c_proj input is then not the state "
            "stream and this patch would edit the wrong stream")
        assert len(getattr(t, "final_shared_h", [])) == 0, (
            "this model has shared trailing blocks, which A5's block indexing does not "
            "cover; head indices would not line up")
        blocks = list(t.split_h if hasattr(t, "split_h") else t.h)
    out = []
    for i, b in enumerate(blocks):
        a = b.attn
        nh = int(a.n_head)
        out.append((i, a.c_proj, nh, int(a.hidden_size) // nh))
    return out


class MemHeadIO:
    """Capture, inject or zero memory-stream head outputs.

    Layout is detected from the tensor the hook receives, so no family branch is needed
    downstream:
      (b, T, nh*hd)      two-tower state tower / single-stream transformer -> slot = pos
      (b, 2T, nh*hd)     tied SPS, interleaved                             -> slot = 2*pos
    """

    def __init__(self, model, family, block_len):
        self.sites = {b: (mod, nh, hd) for b, mod, nh, hd in mem_attn_sites(model, family)}
        self.T = int(block_len)
        self.mode = None
        self.spec = {}          # block -> [head, ...]
        self.positions = ()     # token positions to touch; () means "every position"
        self.store = {}         # (block, head, pos) -> (b, hd) float tensor
        self._handles = []

    # -- layout ------------------------------------------------------------------
    def _set(self, y, pos, h0, h1, val):
        if y.shape[1] == 2 * self.T:
            y[:, 2 * pos, h0:h1] = val
        else:
            y[:, pos, h0:h1] = val

    def _get(self, y, pos, h0, h1):
        if y.shape[1] == 2 * self.T:
            return y[:, 2 * pos, h0:h1]
        return y[:, pos, h0:h1]

    def _set_all(self, y, h0, h1, val):
        if y.shape[1] == 2 * self.T:
            y[:, 0::2, h0:h1] = val
        else:
            y[:, :, h0:h1] = val

    # -- hook --------------------------------------------------------------------
    def _hook(self, blk):
        def pre(mod, args):
            heads = self.spec.get(blk)
            if not heads or self.mode is None:
                return None
            y = args[0]
            _, nh, hd = self.sites[blk]
            if self.mode == "capture":
                for h in heads:
                    for p in self.positions:
                        self.store[(blk, h, int(p))] = \
                            self._get(y, int(p), h * hd, (h + 1) * hd).detach().clone()
                return None
            y = y.clone()
            for h in heads:
                h0, h1 = h * hd, (h + 1) * hd
                if self.mode == "zero":
                    self._set_all(y, h0, h1, 0)
                elif self.mode == "inject":
                    for p in self.positions:
                        v = self.store.get((blk, h, int(p)))
                        if v is not None:
                            self._set(y, int(p), h0, h1, v.to(y.dtype))
            return (y,) + tuple(args[1:])
        return pre

    def __enter__(self):
        for blk, (mod, _nh, _hd) in self.sites.items():
            self._handles.append(mod.register_forward_pre_hook(self._hook(blk)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    # -- driving -----------------------------------------------------------------
    def configure(self, mode, spec, positions=()):
        self.mode = mode
        self.spec = {int(b): sorted(set(int(h) for h in hs)) for b, hs in spec.items() if hs}
        self.positions = tuple(int(p) for p in positions)

    def off(self):
        self.mode, self.spec, self.positions = None, {}, ()


# --------------------------------------------------------------------------------------
# head selection (family-general; asserts equality with A8 on the two-tower arms)
# --------------------------------------------------------------------------------------
def pick_heads(a5res, n_prev=3, ctrl_mode=None, ctrl_seed=A8.CTRL_SEED, draw=0,
               head_stats=None, match_k=None, exclude_layers=()):
    """-> (matching head, top-n memory prev-token heads, control heads).

    Identical logic to `a8_headpatch.pick_heads`, with the stream names taken from the
    family instead of hard-coded, so the single-stream transformer (one stream, which is
    both the memory and the readout) and the joint families are selectable too.  For the
    two-tower family the result is asserted to equal A8's, so the two analyses are
    guaranteed to be talking about the same three heads.

    Control selection is NOT re-implemented here: it is `A8.select_controls` (mode,
    seed and draw as in `A8.pick_heads`; defaults = the historical single draw).  In the
    non-default modes the single-stream family also excludes the matching head, which
    lives in the same stream as the memory heads there.
    """
    ctrl_mode = ctrl_mode or A8.DEFAULT_CTRL_MODE
    match_k = A8.MATCH_K if match_k is None else match_k
    fam = a5res["family"]
    read_q = "single" if fam == "standard" else "pred"
    mem_q = "single" if fam == "standard" else "state"

    best = None
    for r in a5res["synthetic"]["per_head"]:
        if r["query_stream"] != read_q:
            continue
        for h, lift in enumerate(r["induction_lift"]):
            if lift is None or not np.isfinite(lift):
                continue
            if best is None or lift > best["lift"]:
                best = dict(block=int(r["block"]), head=int(h), lift=float(lift),
                            key_stream=r["key_stream"])
    assert best is not None, "no readout-query induction lift in the A5 result"

    prev_all, per_block = [], {}
    for r in a5res["natural"]["per_head"]:
        if r["query_stream"] != mem_q or "prev_token_mass" not in r:
            continue
        # A memory-query row exists once per key stream.  A previous-token head is defined
        # by WHERE ITS ATTENTION GOES, not by which key set the neighbour sits in, so the
        # distance-1 mass is SUMMED over key streams.  Two-tower and the single-stream
        # model have one key stream, so this is identical to A8 there; in the joint
        # families a pred key at distance 1 carries the previous token exactly as a state
        # key does, and ignoring it picks the wrong head (verified: it did).
        b = int(r["block"])
        m = list(r["prev_token_mass"])
        cur = per_block.get(b)
        per_block[b] = m if cur is None else [a + c for a, c in zip(cur, m)]
    for b, masses in per_block.items():
        for h, m in enumerate(masses):
            prev_all.append(dict(block=b, head=int(h), prev_token_mass=float(m)))
    prev_all.sort(key=lambda d: -d["prev_token_mass"])
    top = prev_all[:n_prev]

    exclude = ()
    if fam == "standard" and ctrl_mode != A8.DEFAULT_CTRL_MODE:
        exclude = ((best["block"], best["head"]),)
    rng = np.random.default_rng(A8.draw_seed(ctrl_seed, draw))
    ctrl = A8.select_controls(top, per_block, mode=ctrl_mode, rng=rng,
                              head_stats=head_stats, exclude=exclude, match_k=match_k,
                              exclude_layers=exclude_layers)
    if fam == "two_tower":
        b8, t8, c8 = A8.pick_heads(a5res, n_prev, ctrl_mode=ctrl_mode, ctrl_seed=ctrl_seed,
                                   draw=draw, head_stats=head_stats, match_k=match_k,
                                   exclude_layers=exclude_layers)
        assert (b8["block"], b8["head"]) == (best["block"], best["head"]), \
            f"A9 picked a different matching head than A8: {best} vs {b8}"
        assert [(d["block"], d["head"]) for d in t8] == [(d["block"], d["head"]) for d in top], \
            "A9 picked different prev-token heads than A8"
        assert [(d["block"], d["head"]) for d in c8] == [(d["block"], d["head"]) for d in ctrl], \
            "A9 picked different control heads than A8"
    return best, top, ctrl


# --------------------------------------------------------------------------------------
# the minimal-pair probe
# --------------------------------------------------------------------------------------
def build_examples(cfg, n_ex, seed=PAIR_SEED, mode="random", data=None):
    """-> list of dicts with clean/corrupted token arrays and the metric's token ids.

    `mode="random"`  A5's uniform-random repeated block.  Noiseless -- nothing but
                     induction can produce the answer -- but far out of distribution, so
                     the model's OUTPUT on it is weak even where its attention is not.
    `mode="natural"` the same construction on real validation text: a natural 256-token
                     segment is duplicated in place, so the behavioural (logit) effect is
                     measured on text the model is actually competent on.  Both probes are
                     run; the circuit claim should hold on both.
    """
    rng = np.random.default_rng(seed)
    exs = []
    starts = None
    if mode == "natural":
        starts = C.seq_starts(data, 4 * n_ex)
    n_tried = 0
    while len(exs) < n_ex:
        n_tried += 1
        assert n_tried < 50 * n_ex + 100, "could not build enough minimal pairs"
        if mode == "natural":
            j = int(starts[n_tried % len(starts)] + rng.integers(0, 1 + (C.BLOCK // 8)))
            j = max(0, min(j, len(data) - C.BLOCK - 1))
            toks = data[j:j + C.BLOCK].astype(np.int64).copy()
            toks[A5.COPY_B:A5.COPY_B + A5.BLOCK_LEN] = \
                toks[A5.COPY_A:A5.COPY_A + A5.BLOCK_LEN]
        else:
            toks = A5.synthetic_batch(cfg, rng)
        blk = toks[A5.COPY_A:A5.COPY_A + A5.BLOCK_LEN].copy()
        i = int(rng.integers(1, A5.BLOCK_LEN - 1))
        j = int(rng.integers(1, A5.BLOCK_LEN - 1))
        if abs(i - j) < MIN_SEP:
            continue
        if blk[i] == blk[j] or blk[i + 1] == blk[j + 1]:
            continue
        # the swap must be the ONLY thing that moves the match: if the query token or the
        # distractor's predecessor occurs more than once inside the first copy, the
        # counterfactual is not clean (natural text repeats tokens; random text rarely does)
        if int((blk == blk[i]).sum()) != 1 or int((blk == blk[j]).sum()) != 1:
            continue
        corr = toks.copy()
        corr[A5.COPY_A + i], corr[A5.COPY_A + j] = blk[j], blk[i]
        exs.append(dict(
            clean=toks, corrupted=corr, probe=mode,
            q=A5.COPY_B + i,
            p_ans=A5.COPY_A + i + 1, p_dis=A5.COPY_A + j + 1,
            answer=int(blk[i + 1]), distractor=int(blk[j + 1]), i=i, j=j))
    return exs


def _logits_at(model, toks, q):
    X = torch.from_numpy(toks)[None].cuda()
    Y = torch.from_numpy(np.concatenate([toks[1:], toks[:1]]))[None].cuda()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        lg = C.forward_logits(model, X, Y)
    row = lg[0, q].float().clone()
    del lg
    return row


def _metric(row, ex):
    lp = torch.log_softmax(row, dim=-1)
    return dict(logit_diff=float(row[ex["answer"]] - row[ex["distractor"]]),
                logprob_answer=float(lp[ex["answer"]]),
                top1_is_answer=bool(int(row.argmax()) == ex["answer"]),
                top1_is_distractor=bool(int(row.argmax()) == ex["distractor"]))


def _attn_metric(model, cfg, family, toks, ex, mh):
    """The CIRCUIT-level metric: the matching head's own attention on the two keys.

    The logit metric above is behavioural and therefore also pays for everything
    downstream of the matching head -- on the uniform-random probe the model's output
    head barely converts its (enormous) induction attention into a prediction at all.
    This reads the retrieval itself: how much of the matching head's probability sits on
    the answer's key versus the distractor's key.  Measured through
    `common.attention_views`, i.e. the same recomputation A5 scores, so the same patch
    hooks apply (they hang off `c_proj`, which every view path runs).
    """
    X = torch.from_numpy(toks)[None].cuda()
    Y = torch.from_numpy(np.concatenate([toks[1:], toks[:1]]))[None].cuda()
    a_ans = a_dis = float("nan")
    qpos = np.asarray([ex["q"]])
    for v in C.attention_views(model, cfg, family, X, Y, qpos):
        if int(v.block) != int(mh["block"]) or v.stream != mh["stream"]:
            continue                       # the generator is exhausted, never broken out
        ks0 = (v.k_stream == 0)
        for nm, p in (("ans", ex["p_ans"]), ("dis", ex["p_dis"])):
            sel = torch.nonzero((v.k_tok == int(p)) & ks0).flatten()
            if sel.numel() == 0:
                continue
            val = float(v.p[0, int(mh["head"]), int(sel[0])])
            if nm == "ans":
                a_ans = val
            else:
                a_dis = val
    return dict(attn_answer_key=a_ans, attn_distractor_key=a_dis,
                attn_diff=a_ans - a_dis)


def boot_ci(num, den, n_boot=BOOT, seed=BOOT_SEED):
    """95% bootstrap CI of mean(num)/mean(den) -- the aggregate recovery fraction."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].mean(1) / np.where(np.abs(den[idx].mean(1)) < 1e-9, np.nan, den[idx].mean(1))
    r = r[np.isfinite(r)]
    return [round(float(np.percentile(r, 2.5)), 4), round(float(np.percentile(r, 97.5)), 4)]


def run_patching(model, cfg, family, exs, match, top_prev, ctrl, io, mh, attn=True):
    b1 = top_prev[0]["block"]
    nh_b1 = io.sites[b1][1]
    all_spec = {b: list(range(nh)) for b, (_m, nh, _hd) in io.sites.items()}
    cap_spec = {b: set(hs) for b, hs in all_spec.items()}

    prev_spec = {d["block"]: [d["head"]] for d in top_prev[:1]}
    top3_spec = {}
    for d in top_prev:
        top3_spec.setdefault(d["block"], []).append(d["head"])
    ctrl_spec = {}
    for d in ctrl:
        ctrl_spec.setdefault(d["block"], []).append(d["head"])

    # ---- clean pass: metric + the activations every patch arm will re-use
    clean_rows, stores = [], []
    for ex in exs:
        io.configure("capture", cap_spec, (ex["p_ans"], ex["p_dis"]))
        io.store = {}
        r = _metric(_logits_at(model, ex["clean"], ex["q"]), ex)
        stores.append(io.store)
        if attn:
            io.configure("capture", cap_spec, (ex["p_ans"], ex["p_dis"]))
            io.store = {}
            r.update(_attn_metric(model, cfg, family, ex["clean"], ex, mh))
        clean_rows.append(r)
    io.off()

    rng = np.random.default_rng(PAIR_SEED + 1)
    donor_of = [(k + 1 + int(rng.integers(0, len(exs) - 1))) % len(exs) for k in range(len(exs))]

    arms = {
        "prev1_keyslots":          (prev_spec,  "both", False),
        "prev_top3_keyslots":      (top3_spec,  "both", False),
        "prev1_answer_slot":       (prev_spec,  "ans",  False),
        "prev1_distractor_slot":   (prev_spec,  "dis",  False),
        "ctrl_head_keyslots":      (ctrl_spec,  "both", False),
        "ctrl_donor_prev1":        (prev_spec,  "both", True),
        "ctrl_donor_prev1_answer_slot": (prev_spec, "ans", True),
        "prev_top3_answer_slot":   (top3_spec,  "ans",  False),
        "ceiling_block_all_heads": ({b1: list(range(nh_b1))}, "both", False),
        # sufficiency ceilings: restore EVERY memory head at the key slot(s).  If even
        # this does not rebuild the match, what the key needs is not carried by attention
        # heads at all (embedding / MLP path), and no single-head patch could have worked.
        "ceiling_allheads_answer_slot": (all_spec, "ans",  False),
        "ceiling_allheads_keyslots":    (all_spec, "both", False),
    }

    out = {"clean": clean_rows, "corrupted": []}
    for ex in exs:
        io.off()
        r = _metric(_logits_at(model, ex["corrupted"], ex["q"]), ex)
        if attn:
            r.update(_attn_metric(model, cfg, family, ex["corrupted"], ex, mh))
        out["corrupted"].append(r)
    for name, (spec, slots, donor) in arms.items():
        t0 = time.time()
        rows = []
        for k, ex in enumerate(exs):
            src = stores[donor_of[k]] if donor else stores[k]
            if donor:
                # a donor's activations live at ITS key slots; re-key them onto this
                # example's slots so the injection site is identical to the real arm
                dex = exs[donor_of[k]]
                remap = {dex["p_ans"]: ex["p_ans"], dex["p_dis"]: ex["p_dis"]}
                src = {(b, h, remap[p]): v for (b, h, p), v in src.items() if p in remap}
            io.store = src
            pos = {"ans": (ex["p_ans"],), "dis": (ex["p_dis"],)}.get(
                slots, (ex["p_ans"], ex["p_dis"]))
            io.configure("inject", spec, pos)
            r = _metric(_logits_at(model, ex["corrupted"], ex["q"]), ex)
            if attn:
                io.configure("inject", spec, pos)
                io.store = src
                r.update(_attn_metric(model, cfg, family, ex["corrupted"], ex, mh))
            rows.append(r)
        out[name] = rows
        print(f"  arm {name:24s} {time.time()-t0:5.1f}s", flush=True)
    io.off()
    return out


def summarise_patching(out, exs):
    ld = {k: np.array([r["logit_diff"] for r in v]) for k, v in out.items()}
    has_attn = "attn_diff" in out["clean"][0]
    ad = ({k: np.array([r["attn_diff"] for r in v]) for k, v in out.items()}
          if has_attn else None)
    den = ld["clean"] - ld["corrupted"]
    den_a = (ad["clean"] - ad["corrupted"]) if has_attn else None
    res, per_arm = {}, {}
    for name, v in ld.items():
        rows = out[name]
        per_arm[name] = dict(
            n=len(v),
            mean_logit_diff=round(float(v.mean()), 4),
            sem_logit_diff=round(float(v.std(ddof=1) / np.sqrt(len(v))), 4),
            top1_answer_rate=round(float(np.mean([r["top1_is_answer"] for r in rows])), 4),
            top1_distractor_rate=round(float(np.mean([r["top1_is_distractor"] for r in rows])), 4),
            mean_logprob_answer=round(float(np.mean([r["logprob_answer"] for r in rows])), 4))
        if has_attn:
            per_arm[name].update(
                mean_attn_answer_key=round(float(np.nanmean(
                    [r["attn_answer_key"] for r in rows])), 5),
                mean_attn_distractor_key=round(float(np.nanmean(
                    [r["attn_distractor_key"] for r in rows])), 5),
                mean_attn_diff=round(float(np.nanmean(ad[name])), 5))
        if name not in ("clean", "corrupted"):
            num = v - ld["corrupted"]
            per_arm[name]["recovery_fraction"] = round(float(num.mean() / den.mean()), 4)
            per_arm[name]["recovery_ci95"] = boot_ci(num, den)
            pe = num / np.where(np.abs(den) < 1e-6, np.nan, den)
            per_arm[name]["recovery_per_example_median"] = round(float(np.nanmedian(pe)), 4)
            if has_attn:
                na = ad[name] - ad["corrupted"]
                per_arm[name]["attn_recovery_fraction"] = round(
                    float(na.mean() / den_a.mean()), 4)
                per_arm[name]["attn_recovery_ci95"] = boot_ci(na, den_a)
    res["arms"] = per_arm
    res["effect_size"] = dict(
        mean_clean_minus_corrupted=round(float(den.mean()), 4),
        frac_examples_with_positive_effect=round(float((den > 0).mean()), 4),
        n_examples=len(den))
    if has_attn:
        res["effect_size"]["mean_attn_clean_minus_corrupted"] = round(float(den_a.mean()), 5)
    return res


# --------------------------------------------------------------------------------------
# Part 2 -- natural text
# --------------------------------------------------------------------------------------
def natural_sets(x, min_gap=MIN_GAP):
    """-> (induction_mask, repeat_ctrl_mask) over scored positions t in [0, T-1)."""
    T = len(x)
    ind = np.zeros(T - 1, dtype=bool)
    rep = np.zeros(T - 1, dtype=bool)
    occ = {}
    for t in range(T - 1):
        tgt = int(x[t + 1])
        us = occ.get(tgt)
        if us:
            far = [u for u in us if u <= t - min_gap]
            if far:
                rep[t] = True
                if any(u >= 1 and int(x[u - 1]) == int(x[t]) for u in far):
                    ind[t] = True
                    rep[t] = False
        occ.setdefault(int(x[t]), []).append(t)
    return ind, rep


def score_tokens(model, data, starts, chunk=512):
    """-> (nll[n_seq, T-1], correct[n_seq, T-1], token arrays)."""
    nlls, cors, xs = [], [], []
    for j in starts:
        xnp = data[j:j + C.BLOCK].astype(np.int64)
        X, Y = C.batch_of(data, j)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            lg = C.forward_logits(model, X, Y)
        n = np.zeros(C.BLOCK - 1, dtype=np.float64)
        c = np.zeros(C.BLOCK - 1, dtype=bool)
        for s in range(0, C.BLOCK - 1, chunk):
            e = min(s + chunk, C.BLOCK - 1)
            z = lg[0, s:e].float()
            y = Y[0, s:e]
            n[s:e] = (-torch.log_softmax(z, -1).gather(1, y.view(-1, 1)).squeeze(1)).cpu().numpy()
            c[s:e] = (z.argmax(-1) == y).cpu().numpy()
            del z
        del lg
        nlls.append(n); cors.append(c); xs.append(xnp)
    return np.stack(nlls), np.stack(cors), np.stack(xs)


def _zero_spec(heads):
    """-> {block: [head, ...]} with duplicates removed, preserving order."""
    spec = {}
    for d in heads:
        b, h = int(d["block"]), int(d["head"])
        hs = spec.setdefault(b, [])
        if h not in hs:
            hs.append(h)
    return spec


def control_heads_for_natural(ctrl, top_prev, match):
    """The real-text unrelated-head control set.

    Head SELECTION is not re-derived here: it is exactly the `ctrl` list that
    `pick_heads` already returns, i.e. A8's rule -- for each of the n_prev previous-token
    heads, a head in the SAME BLOCK whose previous-token mass is at or below that block's
    median, drawn with np.random.default_rng(A8.CTRL_SEED) from the eligible heads that
    are not themselves previous-token heads.  Reusing it is the point: the synthetic
    unrelated-head control (A8) and this real-text one then silence the SAME heads, so
    the two controls mean the same thing.

    Two exclusions are made explicit here rather than left to the selection rule, so a
    control head can never coincide with a head the claim is about:
      * the readout matching head, and
      * the previous-token heads themselves,
    are dropped, and duplicates (the same block can be drawn twice) are collapsed.
    """
    bad = {(int(d["block"]), int(d["head"])) for d in top_prev}
    if match is not None:
        bad.add((int(match["block"]), int(match["head"])))
    return [d for d in ctrl if (int(d["block"]), int(d["head"])) not in bad]


def run_natural(model, cfg, family, io, top_prev, n_seq, ctrl=None, match=None,
                extra_ctrls=None):
    """`extra_ctrls` (non-default control runs only): a list of control-head lists, one
    per draw, each zeroed on the SAME sequences and scored on the SAME token sets; the
    per-draw numbers are returned under `out["_ctrl_draws"]` (main() moves them to the
    top-level `ctrl_draws`).  With extra_ctrls=None the output is unchanged."""
    data = C.val_memmap(cfg)
    starts = C.seq_starts(data, n_seq)
    io.off()
    nll_c, cor_c, xs = score_tokens(model, data, starts)
    zspec = _zero_spec(top_prev)
    io.configure("zero", zspec)
    nll_a, cor_a, _ = score_tokens(model, data, starts)
    io.off()

    # ---- real-text unrelated-head control ------------------------------------------
    # Mirrors the arm above in every respect except WHICH heads are zeroed: same `zero`
    # mechanism on the same memory-stream c_proj pre-hook, same validation sequences
    # (`starts` is reused, not redrawn), same scored token sets.
    ctrl_use = control_heads_for_natural(ctrl or [], top_prev, match)
    nll_k = cor_k = None
    if ctrl_use:
        io.configure("zero", _zero_spec(ctrl_use))
        nll_k, cor_k, _ = score_tokens(model, data, starts)
        io.off()

    ind = np.zeros_like(cor_c); rep = np.zeros_like(cor_c)
    for s in range(xs.shape[0]):
        ind[s], rep[s] = natural_sets(xs[s])
    sets = dict(all=np.ones_like(cor_c),
                induction=ind,
                induction_clean_top1=ind & cor_c,
                repeat_ctrl=rep,
                repeat_ctrl_clean_top1=rep & cor_c)
    out = {}
    for name, m in sets.items():
        k = int(m.sum())
        if k == 0:
            out[name] = dict(n=0)
            continue
        out[name] = dict(
            n=k,
            clean_acc=round(float(cor_c[m].mean()), 4),
            ablated_acc=round(float(cor_a[m].mean()), 4),
            d_acc=round(float(cor_a[m].mean() - cor_c[m].mean()), 4),
            clean_nll=round(float(nll_c[m].mean()), 4),
            ablated_nll=round(float(nll_a[m].mean()), 4),
            d_nll=round(float(nll_a[m].mean() - nll_c[m].mean()), 4),
            d_nll_sem=round(float((nll_a[m] - nll_c[m]).std(ddof=1) / np.sqrt(k)), 5))
        if cor_k is not None:
            out[name].update(
                control_acc=round(float(cor_k[m].mean()), 4),
                d_acc_control=round(float(cor_k[m].mean() - cor_c[m].mean()), 4),
                control_nll=round(float(nll_k[m].mean()), 4),
                d_nll_control=round(float(nll_k[m].mean() - nll_c[m].mean()), 4),
                d_nll_control_sem=round(
                    float((nll_k[m] - nll_c[m]).std(ddof=1) / np.sqrt(k)), 5))
    out["_definition"] = dict(
        min_gap=MIN_GAP, n_val_seq=int(len(starts)), block=C.BLOCK,
        induction=("exists u <= t-MIN_GAP with x[u]==x[t+1] and x[u-1]==x[t]"),
        repeat_ctrl=("target occurred >= MIN_GAP back but never preceded by x[t]"),
        ablation=[dict(block=d["block"], head=d["head"],
                       prev_token_mass=d["prev_token_mass"]) for d in top_prev],
        control_ablation=[dict(block=d["block"], head=d["head"],
                               prev_token_mass=d["prev_token_mass"]) for d in ctrl_use],
        control_selection=("a8_headpatch.pick_heads rule, reused verbatim via "
                           "a9.pick_heads: same block as each previous-token head, "
                           "prev-token mass <= that block's median, not a previous-token "
                           "head, rng seed a8_headpatch.CTRL_SEED; the matching head and "
                           "the previous-token heads are then excluded explicitly and "
                           "duplicates collapsed"),
        control_measures=("control_acc / control_nll: the same zero-ablation of the same "
                          "number of memory-stream heads, on the same sequences, with "
                          "unrelated heads substituted for the previous-token heads"))
    if extra_ctrls:
        draws = []
        for k, heads in enumerate(extra_ctrls):
            io.configure("zero", _zero_spec(heads))
            nll_d, cor_d, _ = score_tokens(model, data, starts)
            io.off()
            per = {}
            for name, m in sets.items():
                n = int(m.sum())
                if n == 0:
                    per[name] = dict(n=0)
                    continue
                per[name] = dict(
                    n=n,
                    control_acc=round(float(cor_d[m].mean()), 4),
                    d_acc_control=round(float(cor_d[m].mean() - cor_c[m].mean()), 4),
                    control_nll=round(float(nll_d[m].mean()), 4),
                    d_nll_control=round(float(nll_d[m].mean() - nll_c[m].mean()), 4),
                    d_nll_control_sem=round(
                        float((nll_d[m] - nll_c[m]).std(ddof=1) / np.sqrt(n)), 5))
            draws.append(dict(draw=k, heads=[dict(block=d["block"], head=d["head"],
                                                  prev_token_mass=d["prev_token_mass"])
                                             for d in heads], sets=per))
            print(f"  ctrl draw {k + 1}: " + ", ".join(f"b{d['block']}h{d['head']}" for d in heads)
                  + f"  induction_clean_top1 acc {per['induction_clean_top1'].get('control_acc')}",
                  flush=True)
        out["_ctrl_draws"] = draws
    return out


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-ex", type=int, default=N_EX)
    ap.add_argument("--n-prev", type=int, default=3)
    ap.add_argument("--nat-seq", type=int, default=16)
    ap.add_argument("--natural", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--patching", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--attn", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--probes", default="random,natural")
    A8.add_ctrl_args(ap)
    args = ap.parse_args()
    sfx = A8.ctrl_suffix(args.ctrl_mode, args.n_ctrl_draws, args.ctrl_seed,
                         args.ctrl_match_k, args.ctrl_exclude_layers)
    multi = bool(sfx)

    run = C.resolve(args.run)
    out = args.out or C.result_path("a9_patching", run + sfx)
    C.assert_node_local_triton()

    a5_path = C.result_path("a5_induction", run)
    assert os.path.exists(a5_path), f"A9 reads A5's committed result; missing {a5_path}"
    with open(a5_path) as f:
        a5res = json.load(f)

    # ---- non-default control distribution (see A8.select_controls) ------------------
    loaded, stats, draws, kw = None, None, None, {}
    if multi:
        kw = dict(ctrl_mode=args.ctrl_mode, ctrl_seed=args.ctrl_seed,
                  match_k=args.ctrl_match_k, exclude_layers=args.ctrl_exclude_layers)
        if args.ctrl_mode == "pattern_matched":
            # live statistics on the SAME eval batch Part 2 scores (the natural starts)
            loaded = C.load(run)
            m_, cfg_, fam_, _ck = loaded
            data_ = C.val_memmap(cfg_)
            seqs = [data_[j:j + C.BLOCK].astype(np.int64)
                    for j in C.seq_starts(data_, args.nat_seq)]
            t0 = time.time()
            stats = A8.mem_head_stats(m_, cfg_, fam_, seqs)
            kw["head_stats"] = stats
            print(f"  head stats on {len(seqs)} natural seqs ({time.time()-t0:.1f}s)",
                  flush=True)
        draws = [pick_heads(a5res, args.n_prev, draw=k, **kw)[2]
                 for k in range(args.n_ctrl_draws)]

    match, top_prev, ctrl = pick_heads(a5res, args.n_prev, **kw)
    print(f"run={run} family={a5res['family']}\n"
          f"  matching head  block={match['block']} head={match['head']} "
          f"L0(a5)={match['lift']:.1f}", flush=True)
    for i, d in enumerate(top_prev, 1):
        print(f"  prev-token #{i} block={d['block']} head={d['head']} "
              f"mass={d['prev_token_mass']:.4f}", flush=True)
    for i, d in enumerate(ctrl, 1):
        print(f"  control   #{i} block={d['block']} head={d['head']} "
              f"mass={d['prev_token_mass']:.4f}", flush=True)

    model, cfg, family, ckpath = loaded if loaded is not None else C.load(run)
    io = MemHeadIO(model, family, C.BLOCK)

    res = dict(analysis="a9_patching", run=run, model_id=C.mid_of(run),
               label=C.label_of(run), family=family, checkpoint=ckpath,
               question=("does the readout matching head READ the memory prev-token "
                         "head's output as its key? interchange-patch the clean "
                         "prev-token head output into a corrupted run, at the key slots "
                         "only, and measure recovery of the answer-vs-distractor logit "
                         "difference"),
               matching_head=match, prev_token_heads=top_prev, control_heads=ctrl)

    mh = dict(block=int(match["block"]), head=int(match["head"]),
              stream=("single" if family == "standard" else "pred"))
    with io:
        if args.patching:
            data = C.val_memmap(cfg)
            for probe in [p for p in ("random", "natural") if p in args.probes.split(",")]:
                exs = build_examples(cfg, args.n_ex, mode=probe, data=data)
                print(f"--- patching [{probe}]: {len(exs)} minimal pairs", flush=True)
                raw = run_patching(model, cfg, family, exs, match, top_prev, ctrl, io,
                                   mh, attn=args.attn)
                pr = summarise_patching(raw, exs)
                pr["design"] = dict(
                    probe=probe,
                    probe_source=("a5_induction.synthetic_batch (verbatim)" if probe == "random"
                                  else "validation text with a 256-token segment duplicated"),
                    construction="2-token counterfactual swap inside the first copy",
                    metric_behavioural="logit[answer] - logit[distractor] at the query position",
                    metric_circuit=("the matching head's attention on the answer key minus "
                                    "its attention on the distractor key"),
                    patch_sites="the two edited key slots (COPY_A+i+1, COPY_A+j+1)",
                    matching_head=mh,
                    n_examples=len(exs), pair_seed=PAIR_SEED, min_sep=MIN_SEP,
                    bootstrap=BOOT, copies=[A5.COPY_A, A5.COPY_B],
                    block_len=A5.BLOCK_LEN)
                res.setdefault("patching", {})[probe] = pr
                print(f"\n[{probe}] arm                  LD   recov(LD)      "
                      f"attn-diff  recov(attn)", flush=True)
                for k, v in pr["arms"].items():
                    rf = v.get("recovery_fraction")
                    ra = v.get("attn_recovery_fraction")
                    print(f"  {k:30s} {v['mean_logit_diff']:7.2f}  "
                          f"{('%7.3f' % rf) if rf is not None else '    -  '}   "
                          f"{v.get('mean_attn_diff', float('nan')):9.4f}  "
                          f"{('%7.3f' % ra) if ra is not None else '    -  '}", flush=True)

        if args.natural:
            print("\n--- natural-text ablation arm", flush=True)
            if not multi:
                res["natural"] = run_natural(model, cfg, family, io, top_prev,
                                             args.nat_seq, ctrl=ctrl, match=match)
            else:
                # stream-aware exclusion: the matching head only shares a stream with the
                # memory heads in the single-stream family
                mfilt = match if family == "standard" else None
                nat_draws = [control_heads_for_natural(c, top_prev, mfilt) for c in draws]
                res["natural"] = run_natural(model, cfg, family, io, top_prev,
                                             args.nat_seq, ctrl=ctrl, match=mfilt,
                                             extra_ctrls=nat_draws[1:])
                res["ctrl_draws"], res["ctrl_summary"] = natural_ctrl_summary(
                    res["natural"], nat_draws, args.ctrl_seed)
            for k, v in res["natural"].items():
                if k.startswith("_") or not v.get("n"):
                    continue
                ca = v.get("control_acc")
                print(f"  {k:24s} n={v['n']:7d}  acc {v['clean_acc']:.3f}->"
                      f"{v['ablated_acc']:.3f}  nll {v['clean_nll']:.3f}->"
                      f"{v['ablated_nll']:.3f}  (d={v['d_nll']:+.3f})"
                      + (f"  ctrl-acc {ca:.3f}" if ca is not None else ""), flush=True)

    if multi:
        res.update(ctrl_mode=args.ctrl_mode, n_ctrl_draws=args.n_ctrl_draws,
                   ctrl_seed=args.ctrl_seed, ctrl_match_k=args.ctrl_match_k,
                   **A8.exclude_layers_record(top_prev, args.ctrl_exclude_layers))
        if stats is not None:
            res["ctrl_match_features"] = list(A8.MATCH_FEATURES)
            res["ctrl_match_stats"] = A8.stats_to_json(stats)
        cs = res.get("ctrl_summary", {}).get("headline")
        if cs:
            print(f"\n[{args.ctrl_mode}] induction_clean_top1 acc lost: prev "
                  f"{cs['prev_cost']:.4f}  ctrl {cs['ctrl_mean']:.4f} +- {cs['ctrl_sd']:.4f}"
                  f"  p(ctrl>=prev)={cs['frac_ctrl_ge_prev']:.3f}", flush=True)
    C.save_json(out, res)


NAT_SUMMARY_SETS = ("induction_clean_top1", "induction", "all", "repeat_ctrl")


def natural_ctrl_summary(nat, nat_draws, ctrl_seed):
    """-> (ctrl_draws, ctrl_summary) for Part 2.  Draw 0 is the arm `run_natural` already
    ran as the control (its `control_*` fields); draws 1.. come from `_ctrl_draws`.

    cost per set: accuracy LOST (-(d_acc)) and NLL GAINED (d_nll); the headline is
    accuracy lost on `induction_clean_top1`, the set the copying claim is about.
    """
    extra = nat.pop("_ctrl_draws", [])
    draws = []
    d0 = {}
    for name in nat:
        v = nat[name]
        if name.startswith("_") or not v.get("n") or "control_acc" not in v:
            continue
        d0[name] = {k: v[k] for k in ("n", "control_acc", "d_acc_control", "control_nll",
                                      "d_nll_control", "d_nll_control_sem")}
    draws.append(dict(draw=0, heads=nat["_definition"]["control_ablation"], sets=d0))
    for e in extra:
        draws.append(dict(draw=int(e["draw"]) + 1, heads=e["heads"], sets=e["sets"]))
    for d in draws:
        d["seed"] = A8.draw_seed(ctrl_seed, d["draw"])
    summ = {}
    for name in NAT_SUMMARY_SETS:
        v = nat.get(name, {})
        if not v.get("n"):
            continue
        acc = [-d["sets"][name]["d_acc_control"] for d in draws
               if d["sets"].get(name, {}).get("n")]
        nll = [d["sets"][name]["d_nll_control"] for d in draws
               if d["sets"].get(name, {}).get("n")]
        summ[name] = dict(acc_lost=A8.summarise_draws(acc, -v["d_acc"]),
                          nll_gained=A8.summarise_draws(nll, v["d_nll"]))
    if "induction_clean_top1" in summ:
        summ["headline"] = dict(metric="accuracy lost on induction_clean_top1",
                                **summ["induction_clean_top1"]["acc_lost"])
    return draws, summ


if __name__ == "__main__":
    main()
