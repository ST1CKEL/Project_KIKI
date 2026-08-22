from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kiki", description="KIKI – 2D-KI-Desktop-Pet")
    parser.add_argument("--debug", action="store_true", help="ausführliche Logs")
    parser.add_argument("--version", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="XDG-Pfade und Konfiguration prüfen, ohne GTK zu starten",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="mit --check bei fehlenden aktivierten Kernkomponenten fehlschlagen",
    )
    parser.add_argument(
        "--prepare-voice-model",
        action="store_true",
        help="deutsches Offline-Sprachmodell sicher herunterladen und prüfen",
    )
    args, rest = parser.parse_known_args(argv)
    if args.version:
        from kiki import __version__

        print(f"kiki {__version__}")
        return 0
    if args.prepare_voice_model:
        from kiki.voice.stt import ensure_vosk_model

        try:
            path = ensure_vosk_model()
        except Exception as exc:
            print(f"voice_model_error={exc}", file=sys.stderr)
            return 1
        print(f"voice_model={path}")
        return 0
    if args.check:
        return _check(strict=args.strict)
    os.environ.setdefault("ADW_DISABLE_PORTAL", "0")
    from kiki.logging_config import setup_logging

    setup_logging(debug=args.debug)
    from kiki.application import run_application
    from kiki.ui.gi_bootstrap import gi  # noqa: F401

    # Gtk.Application consumes argv; keep only the program name plus leftovers.
    forwarded = [sys.argv[0], *rest]
    return run_application(forwarded)


from kiki.ai.factory import active_model  # noqa: E402


def _check(*, strict: bool = False) -> int:
    from kiki import __version__
    from kiki.character.state_machine import CharacterState, CharacterStateMachine
    from kiki.config.settings import load_settings
    from kiki.paths import cache_dir, config_dir, config_path, state_dir, user_data_dir
    from kiki.voice.stt import vosk_model_ready, vosk_runtime_available
    from kiki.voice.system_tts import system_tts_available

    config_dir()
    user_data_dir()
    cache_dir()
    state_dir()
    settings = load_settings()
    machine = CharacterStateMachine()
    print(f"kiki {__version__}")
    print(f"config={config_path()}")
    print(f"provider={settings.ai.provider}")
    # Report the endpoint and model of the *active* provider. Printing the
    # Ollama pair regardless made the self-check claim a model that was not
    # in use once other providers existed.
    if settings.ai.provider == "kiki_harness":
        print(f"harness_url={settings.ai.kiki_harness.base_url}")
        print(f"quantize={settings.ai.kiki_harness.quantize}")
    elif settings.ai.provider == "openai_compatible":
        print(f"api_url={settings.ai.openai_compatible.base_url}")
    else:
        print(f"ollama_url={settings.ai.ollama.base_url}")
    print(f"model={active_model(settings)}")
    print(f"tts={settings.tts.base_url}")
    print(f"tts_speaker={settings.tts.speaker}")
    voice_runtime = vosk_runtime_available()
    voice_model = vosk_model_ready()
    tts_fallback = system_tts_available()
    print(f"voice_vosk={'ready' if voice_runtime else 'missing'}")
    print(f"voice_model={'ready' if voice_model else 'missing'}")
    print(f"tts_fallback={'ready' if tts_fallback else 'missing'}")
    print(f"workspace_roots={len(settings.workspaces.allowed_roots)}")
    print(f"opencode={settings.agents.opencode_binary}")
    print(f"plan_first={settings.agents.plan_first}")
    print(f"state={machine.state}")
    if machine.state is not CharacterState.IDLE:
        return 1
    if strict:
        missing: list[str] = []
        if settings.voice_allowed() and not voice_runtime:
            missing.append("voice_vosk")
        if settings.voice_allowed() and not voice_model:
            missing.append("voice_model")
        if settings.tts_allowed() and settings.tts.fallback_to_system and not tts_fallback:
            missing.append("tts_fallback")
        if missing:
            print(f"check=failed ({', '.join(missing)})")
            return 1
    print("check=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
