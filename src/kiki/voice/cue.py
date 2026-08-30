"""Short audible cues for the microphone window: KIKI announces listening.

A wake word that reacts within a quarter second needs an answer the ear can
catch: two quiet notes say "I hear you" (rising) and "window closed" (falling)
without anyone looking at the screen. Playback is fire-and-forget through the
same PipeWire tools the TTS fallback uses; a missing player degrades to
silence, never to an error in the voice path.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from kiki.paths import bundled_data_dir

log = logging.getLogger(__name__)

SOUNDS = {
    "listen-start": "listen-start.wav",
    "listen-stop": "listen-stop.wav",
}


def sound_path(name: str) -> Path:
    return bundled_data_dir() / "sounds" / SOUNDS[name]


def play_cue(name: str) -> None:
    """Play one cue asynchronously. Failures are logged and swallowed."""
    path = sound_path(name)
    if not path.is_file():
        log.debug("cue sound missing: %s", path)
        return
    player = shutil.which("pw-play") or shutil.which("paplay")
    if player is None:
        log.debug("no audio player for cues (pw-play/paplay)")
        return
    try:
        subprocess.Popen(
            [player, str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        log.debug("cue playback failed: %s", exc)
