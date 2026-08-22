"""XDG and application data locations."""

from __future__ import annotations

import os
from pathlib import Path

APP_DIR_NAME = "kiki"


def _home() -> Path:
    return Path.home()


def xdg_config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME", _home() / ".config"))


def xdg_data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", _home() / ".local/share"))


def xdg_state_home() -> Path:
    return Path(os.environ.get("XDG_STATE_HOME", _home() / ".local/state"))


def xdg_cache_home() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", _home() / ".cache"))


def config_dir() -> Path:
    path = xdg_config_home() / APP_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    return config_dir() / "config.toml"


def user_data_dir() -> Path:
    path = xdg_data_home() / APP_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def state_dir() -> Path:
    path = xdg_state_home() / APP_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    path = xdg_cache_home() / APP_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_path() -> Path:
    return user_data_dir() / "kiki.sqlite3"


def log_path() -> Path:
    return state_dir() / "kiki.log"


def repo_root() -> Path | None:
    """Return the source checkout root when running from a git/dev tree."""
    here = Path(__file__).resolve()
    candidate = here.parents[2]
    if (candidate / "pyproject.toml").is_file() and (candidate / "data" / "character").is_dir():
        return candidate
    return None


def bundled_data_dir() -> Path:
    """Shipped assets (characters, icons, desktop files)."""
    env = os.environ.get("KIKI_DATA_DIR")
    if env:
        return Path(env)
    repo = repo_root()
    if repo is not None:
        return repo / "data"
    pkg_data = Path(__file__).resolve().parent / "data"
    if (pkg_data / "character").is_dir():
        return pkg_data
    return Path("/usr/share/kiki")


def character_dir(character_id: str = "kiki-adult-v3") -> Path:
    return bundled_data_dir() / "character" / character_id


def icon_search_path() -> Path:
    return bundled_data_dir() / "icons"
