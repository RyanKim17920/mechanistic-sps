#!/usr/bin/env python
"""Move run checkpoints between <DUALSPS_OUT_ROOT>/out/<run>/ and a Hugging Face model repo
(--repo, default $DUALSPS_HF_REPO, else the public ryankim17920/mechanistic-sps), stored
there as <run>/<file>. Downloading from the public repo needs no login; uploading to your own repo
needs `huggingface-cli login` and --repo (or DUALSPS_HF_REPO).

    python scripts/dualsps/hf_export.py upload [<run> ...]            # dry run
    python scripts/dualsps/hf_export.py upload --apply [<run> ...]    # upload
    python scripts/dualsps/hf_export.py upload --apply --prune <run>  # upload, then delete
                                                                      # the local copies
    python scripts/dualsps/hf_export.py download <run> [...] [--all]  # final ckpts (--all: every ckpt)
    python scripts/dualsps/hf_export.py download --extra <run> [...]  # everything of <run> in the
                                                                      # extras repo (see below)

--extra downloads from the public ryankim17920/mechanistic-sps-extra instead: the
weights-only checkpoint ladders read by scripts/analysis/a17_trajectory.py part B
(<run>/ladder/ckpt_tokens_<N>.pt; s_two_tower_w0_equal_tiedattn_20b, s_two_tower_afsps_20b,
plus the final of s_two_tower_afsps_20b, which the main repo does not hold) and the embedding
ladders read by scripts/analysis/a21_emb_divergence.py (<run>/emb_ladder.pt;
s_two_tower_seq6_20b, s_two_tower_seq6_20b_seed2). Files land in the same places under
<DUALSPS_OUT_ROOT>/out/<run>/, where those scripts look for them.

upload with no run names covers every run directory. A file already on the Hub at the
same size is not uploaded again. --prune deletes a local file only after re-reading its
size from the Hub, never a file modified in the last --min-age-sec seconds (it may still
be being written), and only for runs that are finished and evaluated: a *_final.pt
exists or was exported, and the live ledger holds the run's val_nll_full_sweep and loss
curve (eval_runs.py reads the intermediate checkpoints for that curve).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger  # noqa: E402
import repo_paths  # noqa: E402


def evaluated_runs():
    """Runs whose newest ledger record has a val NLL and a curve of >= 2 checkpoints."""
    return {run for run, r in ledger.latest_per_run().items()
            if isinstance(r.get("val_nll_full_sweep"), float)
            and len(r.get("curve_checkpoints") or []) >= 2}


def hub_sizes(api, repo):
    info = api.repo_info(repo_id=repo, repo_type="model", files_metadata=True)
    return {s.rfilename: s.size for s in info.siblings}


def upload(args, api):
    OUT = repo_paths.out_root() / "out"
    runs = args.runs or sorted(d.name for d in OUT.iterdir() if d.is_dir())
    evaluated = evaluated_runs()
    remote = hub_sizes(api, args.repo)
    now = time.time()
    for run in runs:
        pts = sorted((OUT / run).glob("*.pt"))
        prune = args.prune and run in evaluated and (
            any(p.name.endswith("_final.pt") for p in pts)
            or any(k.startswith(f"{run}/") and k.endswith("_final.pt") for k in remote))
        print(f"=== {run}: {len(pts)} checkpoint(s)"
              + ("" if not args.prune or prune else "  (not pruned: not finished and evaluated)"))
        for p in pts:
            dest, size = f"{run}/{p.name}", p.stat().st_size
            if now - p.stat().st_mtime < args.min_age_sec:
                print(f"  skip, modified in the last {args.min_age_sec}s: {p.name}")
                continue
            if remote.get(dest) != size:
                if not args.apply:
                    print(f"  would upload {dest} ({size / 2**30:.2f} GiB)")
                    continue
                print(f"  upload {dest} ({size / 2**30:.2f} GiB)", flush=True)
                api.upload_file(path_or_fileobj=str(p), path_in_repo=dest,
                                repo_id=args.repo, repo_type="model")
            if prune and args.apply:
                remote = hub_sizes(api, args.repo)    # verify on the Hub itself
                if remote.get(dest) == size:
                    p.unlink()
                    print(f"  deleted local {p.name} (verified on the Hub)")
                else:
                    print(f"  kept {p.name}: Hub size {remote.get(dest)} != local {size}")


def download(args, api):
    from huggingface_hub import snapshot_download
    OUT = repo_paths.out_root() / "out"
    pattern = "*.pt" if args.all else "*_final.pt"
    for run in args.runs:
        allow = ([f"{run}/*.pt", f"{run}/ladder/*.pt"] if args.extra
                 else [f"{run}/{pattern}"])
        path = snapshot_download(repo_id=args.repo, repo_type="model", local_dir=OUT,
                                 allow_patterns=allow)
        got = sorted(str(p.relative_to(Path(path) / run)) for a in allow
                     for p in Path(path).glob(a))
        if not got:
            sys.exit(f"{run}: nothing matching {allow} in {args.repo}")
        print(f"{run}: {got}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("upload", "download"))
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--repo", help="Hugging Face repo id (default: $DUALSPS_HF_REPO, else "
                                   f"{repo_paths.HF_REPO_DEFAULT})")
    ap.add_argument("--apply", action="store_true", help="upload: really upload (and prune)")
    ap.add_argument("--prune", action="store_true", help="upload: delete verified local copies")
    ap.add_argument("--min-age-sec", type=int, default=120)
    ap.add_argument("--all", action="store_true", help="download: every checkpoint, not just the final")
    ap.add_argument("--extra", action="store_true",
                    help=f"download: the run's files in {repo_paths.HF_EXTRA_REPO} (a17/a21 ladders)")
    args = ap.parse_intermixed_args()    # flags before or after the run names
    if args.mode == "download" and not args.runs:
        ap.error("download needs run names")
    if args.extra and args.mode != "download":
        ap.error("--extra is for download")
    args.repo = args.repo or (repo_paths.HF_EXTRA_REPO if args.extra else repo_paths.hf_repo())
    from huggingface_hub import HfApi
    (upload if args.mode == "upload" else download)(args, HfApi())


if __name__ == "__main__":
    main()
