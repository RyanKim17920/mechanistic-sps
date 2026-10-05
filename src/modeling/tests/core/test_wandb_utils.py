import os

from omegaconf import OmegaConf

from training.wandb_utils import prepare_wandb_dir_from_config


def _cfg(tmp_path, wandb_dir):
    return OmegaConf.create({"system": {"out_root": str(tmp_path / "out_root")},
                             "logging": {"wandb_dir": wandb_dir}})


def test_configured_dir_is_created_and_exported(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_DIR", raising=False)
    resolved = prepare_wandb_dir_from_config(_cfg(tmp_path, "${system.out_root}/wandb"))
    expected = tmp_path / "out_root" / "wandb"
    assert resolved == str(expected)
    assert expected.is_dir()
    assert os.environ["WANDB_DIR"] == str(expected)


def test_configured_dir_wins_but_does_not_overwrite_the_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "env"))
    resolved = prepare_wandb_dir_from_config(_cfg(tmp_path, str(tmp_path / "configured")))
    assert resolved == str(tmp_path / "configured")
    assert os.environ["WANDB_DIR"] == str(tmp_path / "env")


def test_env_is_the_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_DIR", str(tmp_path / "env"))
    assert prepare_wandb_dir_from_config(_cfg(tmp_path, None)) == str(tmp_path / "env")
    assert (tmp_path / "env").is_dir()


def test_nothing_configured_gives_none(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_DIR", raising=False)
    assert prepare_wandb_dir_from_config(_cfg(tmp_path, None)) is None
