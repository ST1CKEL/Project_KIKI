from __future__ import annotations

from pathlib import Path

from kiki.config.settings import load_settings
from kiki.paths import cache_dir, config_dir, config_path, state_dir, user_data_dir


def test_xdg_dirs_are_created(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert config_dir().is_dir()
    assert user_data_dir().is_dir()
    assert cache_dir().is_dir()
    assert state_dir().is_dir()
    assert config_path() == tmp_path / "config" / "kiki" / "config.toml"


def test_load_settings_without_file_uses_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    settings = load_settings()
    assert settings.ai.provider == "ollama"
    assert settings.ai.ollama.base_url == "http://127.0.0.1:11434"
    assert settings.tools.model_tool_use is True
