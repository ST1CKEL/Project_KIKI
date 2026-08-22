from __future__ import annotations

from pathlib import Path

from kiki.main import main


def test_check_reports_voice_capabilities(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    assert main(["--check"]) == 0
    output = capsys.readouterr().out

    assert "voice_vosk=" in output
    assert "voice_model=missing" in output
    assert "tts_fallback=" in output
    assert "check=ok" in output


def test_strict_check_fails_when_enabled_voice_model_is_missing(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))

    assert main(["--check", "--strict"]) == 1
    output = capsys.readouterr().out
    assert "voice_model=missing" in output
    assert "check=failed" in output


def test_check_reports_the_active_provider_not_always_ollama(capsys, tmp_path, monkeypatch):
    """The self-check printed the Ollama model even under another provider."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    from kiki.config.settings import load_settings, save_settings
    from kiki.main import main

    settings = load_settings()
    settings.ai.provider = "kiki_harness"
    save_settings(settings)

    main(["--check"])
    out = capsys.readouterr().out
    assert "provider=kiki_harness" in out
    assert "harness_url=http://127.0.0.1:18770" in out
    assert f"model={settings.ai.kiki_harness.model}" in out
    # The unused Ollama model must not be reported as the active one.
    assert "model=qwen3-vl:4b" not in out
