"""Repository plumbing: executable shims, src/repo_paths.py, the sampler's start offset."""
import importlib
import random
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]


@pytest.mark.skipif(shutil.which("git") is None or not (REPO / ".git").exists(),
                    reason="needs a git checkout")
def test_run_shell_scripts_are_executable_in_git():
    """torchrun --no-python execs rank_shim.sh directly (mode 100755 in git)."""
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-s", "scripts/run/*_shim.sh"],
                         capture_output=True, text=True, check=True).stdout
    modes = {line.split("\t")[1]: line.split()[0] for line in out.splitlines()}
    assert {"scripts/run/eval_shim.sh", "scripts/run/rank_shim.sh"} <= set(modes)
    assert all(m == "100755" for m in modes.values()), modes


def _reload_repo_paths(monkeypatch, tmp_path, env, dotenv=None):
    import repo_paths
    for k in ("DUALSPS_DATA_ROOT", "DUALSPS_OUT_ROOT", "DUALSPS_LOG_DIR", "DUALSPS_HF_REPO"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    if dotenv is not None:
        (tmp_path / ".env").write_text(dotenv)
        repo_paths._load_dotenv(tmp_path / ".env")
    return importlib.reload(repo_paths) if dotenv is None else repo_paths


def test_repo_paths_require_env_with_clear_message(monkeypatch, tmp_path):
    rp = _reload_repo_paths(monkeypatch, tmp_path, {})
    for k in ("DUALSPS_DATA_ROOT", "DUALSPS_HF_REPO"):   # in case <repo>/.env sets them
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="DUALSPS_DATA_ROOT is not set"):
        rp.data_root()
    assert rp.hf_repo() == "ryankim17920/mechanistic-sps"   # public default, overridable
    monkeypatch.setenv("DUALSPS_HF_REPO", "someone/else")
    assert rp.hf_repo() == "someone/else"
    assert rp.RESULTS == REPO / "scripts" / "analysis" / "results"   # repo-relative default


def test_repo_paths_defaults_follow_data_root(monkeypatch, tmp_path):
    rp = _reload_repo_paths(monkeypatch, tmp_path, {"DUALSPS_DATA_ROOT": str(tmp_path)})
    assert rp.out_root() == tmp_path
    assert rp.run_dir("r") == tmp_path / "out" / "r"
    assert rp.log_dir() == tmp_path / "logs"
    assert rp.live_ledger() == tmp_path / "results" / "ledger.jsonl"
    assert rp.live_ledger() != rp.LEDGER   # evaluation never appends to the snapshot


def test_dotenv_fills_unset_variables_only(monkeypatch, tmp_path):
    rp = _reload_repo_paths(monkeypatch, tmp_path, {"DUALSPS_OUT_ROOT": "/from/env"},
                            dotenv="# comment\nDUALSPS_DATA_ROOT=/from/dotenv\n"
                                   "DUALSPS_OUT_ROOT=/from/dotenv/out\nnot a line\n")
    assert rp.data_root() == Path("/from/dotenv")
    assert rp.out_root() == Path("/from/env")


@pytest.mark.parametrize("seed", [1337, 1338, 2])
def test_fresh_start_offset_matches_the_paper_runs_draw(seed):
    """The paper runs drew each rank's offset with the global RNG right after
    random.seed(seed + rank); the dedicated Random must give the same numbers."""
    from training import fresh_start_offset
    n = 27_089_110_623 - 4096
    for rank in range(8):
        random.seed(seed + rank)
        assert fresh_start_offset(seed, rank, n, 1_000_000) == random.randint(0, min(1_000_000, n // 2))
