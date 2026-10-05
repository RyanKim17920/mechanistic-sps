"""Append-only ledger of evaluation results (one JSON object per line).

eval_runs.py appends one fully provenanced record per evaluated run to the live ledger,
<DUALSPS_OUT_ROOT>/results/ledger.jsonl (repo_paths.live_ledger()); a re-evaluation appends
a new record and readers take the newest record per run (latest_per_run). The paper is
built from the snapshot in git (repo_paths.LEDGER, repo_paths.WALLCLOCK), which changes
only through --snapshot (`make snapshot`).

    python scripts/dualsps/ledger.py --list [--run X]   # one line per record
    python scripts/dualsps/ledger.py --show N           # record N (0-based), pretty-printed
    python scripts/dualsps/ledger.py --snapshot         # copy the live ledger and wall-clock
                                                        # files into the snapshot
Add --ledger PATH to read another ledger (e.g. --ledger scripts/analysis/results/ledger.jsonl).
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
import repo_paths  # noqa: E402

# Bump when the record shape changes incompatibly.
SCHEMA = 1


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def git_provenance(repo=REPO):
    """Short HEAD SHA, branch and dirty flag; None where git is unavailable."""
    def run(args):
        try:
            return subprocess.run(["git", "-C", str(repo)] + args, capture_output=True,
                                  text=True, timeout=30, check=True).stdout.strip()
        except Exception:
            return None
    porcelain = run(["status", "--porcelain"])
    return {"git_commit": run(["rev-parse", "--short", "HEAD"]),
            "git_branch": run(["rev-parse", "--abbrev-ref", "HEAD"]),
            "git_dirty": None if porcelain is None else porcelain != ""}


def sha256_file(path, chunk=8 << 20):
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def env_provenance():
    """Host and Triton-cache facts. A Triton cache shared on a network filesystem can
    return wrong kernels silently, so `triton_cache_node_local` marks a trustworthy number."""
    tcd = os.environ.get("TRITON_CACHE_DIR")
    return {
        "hostname": socket.gethostname(),
        "triton_cache_dir": tcd,
        "triton_cache_set": tcd is not None,
        "triton_cache_node_local": bool(tcd) and os.path.abspath(tcd).startswith("/tmp/"),
        "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
    }


def append(record, path=None):
    """Append one record as a single JSON line. Never rewrites existing lines."""
    path = path or repo_paths.live_ledger()
    line = json.dumps(record, default=str)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    return record


def load(path=None):
    """All records, oldest first."""
    path = path or repo_paths.live_ledger()
    if not os.path.exists(path):
        return []
    return [json.loads(line) for line in open(path) if line.strip()]


def latest_per_run(records=None, path=None):
    """Newest record per run, in first-seen order."""
    out = {}
    for r in load(path) if records is None else records:
        if r.get("run") is not None:
            out[r["run"]] = r
    return out


def cmd_list(path, run=None):
    records = [r for r in load(path) if run is None or r.get("run") == run]
    print(f"{path}  ({len(records)} records)")
    print(f"{'#':>3s} {'timestamp (UTC)':19s} {'run':40s} {'commit':>9s} {'val NLL':>9s} {'tri':>3s}")
    for i, r in enumerate(records):
        commit = "backfill" if r.get("backfilled") else (r.get("git_commit") or "-") + (
            "*" if r.get("git_dirty") else "")
        tri = {True: "ok", False: "BAD"}.get(r.get("triton_cache_node_local"), "?")
        nll = r.get("val_nll_full_sweep")
        print(f"{i:>3d} {str(r.get('timestamp'))[:19]:19s} {str(r.get('run'))[:40]:40s} "
              f"{commit:>9s} {'-' if nll is None else f'{nll:.4f}':>9s} {tri:>3s}")
    print("commit '*' = dirty tree; tri = Triton cache node-local (BAD: possibly wrong kernels)")


def snapshot():
    """Copy the live ledger and wall-clock files over the snapshot the paper is built from."""
    import shutil
    for src, dst in ((repo_paths.live_ledger(), repo_paths.LEDGER),
                     (repo_paths.live_wallclock(), repo_paths.WALLCLOCK)):
        if not src.is_file():
            raise FileNotFoundError(f"no live file {src}")
        shutil.copyfile(src, dst)
        rows = sum(1 for line in open(dst) if line.strip())
        print(f"{src} -> {dst}: {rows} rows, sha256 {sha256_file(dst)}")
    print(f"Now record the rows and sha256 in {repo_paths.RESULTS / 'SOURCES.md'}, rebuild the "
          f"paper (make paper), and regenerate the G1 golden (gates/run.py g1 --update) with "
          f"the reason in the commit message.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="one line per record (the default)")
    g.add_argument("--show", type=int, metavar="N", help="record N (0-based), pretty-printed")
    g.add_argument("--snapshot", action="store_true",
                   help="copy the live ledger and wallclock.jsonl into the snapshot in git")
    ap.add_argument("--run", help="--list: only this run")
    ap.add_argument("--ledger", help="ledger file to read (default: repo_paths.live_ledger())")
    a = ap.parse_args(argv)
    path = Path(a.ledger) if a.ledger else repo_paths.live_ledger()
    if a.snapshot:
        snapshot()
    elif a.show is not None:
        print(json.dumps(load(path)[a.show], indent=2, default=str))
    else:
        cmd_list(path, a.run)


if __name__ == "__main__":
    main()
