"""Offline German speech-to-text via Vosk. No cloud."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import stat
import tempfile
import wave
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kiki.paths import user_data_dir

log = logging.getLogger(__name__)

VOSK_MODEL_SMALL = "vosk-model-small-de-0.15"
VOSK_MODEL_LARGE = "vosk-model-de-0.21"
# Default stays the small download; the large model is opt-in via
# [voice] stt_model in the config.
VOSK_MODEL_ID = VOSK_MODEL_SMALL
_MAX_ARCHIVE_MEMBERS = 4096
_EMPTY_MODEL_FILES = frozenset({"ivector/online_cmvn.conf"})

_COMMON_REQUIRED_FILES = (
    "am/final.mdl",
    "conf/mfcc.conf",
    "conf/model.conf",
    "graph/disambig_tid.int",
    "graph/phones/word_boundary.int",
    "ivector/final.dubm",
    "ivector/final.ie",
    "ivector/final.mat",
    "ivector/global_cmvn.stats",
    "ivector/online_cmvn.conf",
    "ivector/splice.conf",
)


@dataclass(frozen=True)
class VoskModel:
    """One pinned download. Only ids in `VOSK_MODELS` ever reach the network."""

    id: str
    url: str
    sha256: str
    max_download_bytes: int
    max_extracted_bytes: int
    required_files: tuple[str, ...]


VOSK_MODELS: dict[str, VoskModel] = {
    VOSK_MODEL_SMALL: VoskModel(
        id=VOSK_MODEL_SMALL,
        url=f"https://alphacephei.com/vosk/models/{VOSK_MODEL_SMALL}.zip",
        sha256="b7e53c90b1f0a38456f4cd62b366ecd58803cd97cd42b06438e2c131713d5e43",
        max_download_bytes=80 * 1024 * 1024,
        max_extracted_bytes=512 * 1024 * 1024,
        # 0.15 ships the two-graph layout.
        required_files=_COMMON_REQUIRED_FILES + ("graph/Gr.fst", "graph/HCLr.fst"),
    ),
    # Recognises names such as the wake word far more reliably than the small
    # model, which mishears "Kiki" as ordinary German words. Costs a ~1.9 GB
    # download and several GB of resident RAM while loaded.
    VOSK_MODEL_LARGE: VoskModel(
        id=VOSK_MODEL_LARGE,
        url=f"https://alphacephei.com/vosk/models/{VOSK_MODEL_LARGE}.zip",
        sha256="245060756f8d8394fc5b13639cf220b7620205795f30f37b6823878d4f603b2a",
        max_download_bytes=3 * 1024 * 1024 * 1024,
        max_extracted_bytes=4 * 1024 * 1024 * 1024,
        # 0.21 ships the single HCLG.fst graph.
        required_files=_COMMON_REQUIRED_FILES + ("graph/HCLG.fst",),
    ),
}


class SpeechError(Exception):
    """STT failed or the model is missing."""


def _vosk_api():
    try:
        from kiki.voice.vosk_ffi import KaldiRecognizer, Model, SetLogLevel
    except ImportError:
        try:
            # Development fallback for source checkouts without Fedora's
            # system library. Packaged KIKI takes the distro-managed path.
            from vosk import KaldiRecognizer, Model, SetLogLevel
        except ImportError as exc:
            raise SpeechError(
                "Vosk-Laufzeit fehlt. Installiere `vosk-api-devel` über Fedora."
            ) from exc
    return KaldiRecognizer, Model, SetLogLevel


def vosk_runtime_available() -> bool:
    try:
        _vosk_api()
    except SpeechError:
        return False
    return True


def vosk_model_dir(model_id: str = VOSK_MODEL_ID) -> Path:
    return user_data_dir() / "vosk" / model_id


def _model_ready(root: Path, model: VoskModel) -> bool:
    try:
        for relative in model.required_files:
            path = root / relative
            if not path.is_file():
                return False
            # This file is intentionally empty in vosk-model-small-de-0.15.
            if relative not in _EMPTY_MODEL_FILES and path.stat().st_size == 0:
                return False
    except OSError:
        return False
    return True


def vosk_model_ready(model_id: str = VOSK_MODEL_ID) -> bool:
    model = VOSK_MODELS.get(model_id)
    if model is None:
        return False
    return _model_ready(vosk_model_dir(model_id), model)


def ensure_vosk_model(model_id: str = VOSK_MODEL_ID) -> Path:
    """Download the configured German model if needed. Call off the GTK thread."""
    model = VOSK_MODELS.get(model_id)
    if model is None:
        raise SpeechError(f"Unbekanntes Vosk-Modell: {model_id}")
    dest = vosk_model_dir(model.id)
    if _model_ready(dest, model):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock_path = dest.parent / f".{model.id}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if _model_ready(dest, model):
            return dest
        return _download_vosk_model(dest, model)


def _download_vosk_model(dest: Path, model: VoskModel) -> Path:
    """Download and publish one model while the cross-process lock is held."""
    import httpx

    log.info("downloading Vosk model %s", model.url)
    fd, raw_path = tempfile.mkstemp(
        prefix=f".{model.id}.",
        suffix=".zip.part",
        dir=dest.parent,
    )
    os.close(fd)
    zip_path = Path(raw_path)
    try:
        downloaded = 0
        digest = hashlib.sha256()
        with httpx.Client(follow_redirects=True, timeout=180.0) as client:
            with client.stream("GET", model.url) as response:
                response.raise_for_status()
                length = response.headers.get("content-length")
                if length and int(length) > model.max_download_bytes:
                    raise SpeechError("Vosk-Download ist unerwartet groß.")
                with zip_path.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        downloaded += len(chunk)
                        if downloaded > model.max_download_bytes:
                            raise SpeechError("Vosk-Download überschreitet das Größenlimit.")
                        digest.update(chunk)
                        handle.write(chunk)
        if downloaded == 0:
            raise SpeechError("Vosk-Download war leer.")
        if not hmac.compare_digest(digest.hexdigest(), model.sha256):
            raise SpeechError("Vosk-Modell hat eine unerwartete SHA-256-Prüfsumme.")
        _extract_model_archive(zip_path, dest, model)
    except SpeechError:
        raise
    except (OSError, httpx.HTTPError, zipfile.BadZipFile, ValueError) as exc:
        raise SpeechError(f"Vosk-Modell konnte nicht sicher installiert werden: {exc}") from exc
    finally:
        zip_path.unlink(missing_ok=True)
    if not _model_ready(dest, model):
        raise SpeechError("Vosk-Modell konnte nicht entpackt werden.")
    return dest


def _extract_model_archive(zip_path: Path, dest: Path, model: VoskModel) -> None:
    """Validate archive boundaries and atomically publish one model directory."""
    if dest.exists():
        if _model_ready(dest, model):
            return
        raise SpeechError(f"Unvollständiges Vosk-Modell blockiert die Installation: {dest}")
    total = 0
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        if not members:
            raise SpeechError("Vosk-Archiv ist leer.")
        if len(members) > _MAX_ARCHIVE_MEMBERS:
            raise SpeechError("Vosk-Archiv enthält unerwartet viele Dateien.")
        for info in members:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise SpeechError("Vosk-Archiv enthält einen unsicheren Pfad.")
            if not path.parts or path.parts[0] != model.id:
                raise SpeechError("Vosk-Archiv enthält ein unerwartetes Wurzelverzeichnis.")
            mode = (info.external_attr >> 16) & 0o170000
            if mode == stat.S_IFLNK:
                raise SpeechError("Vosk-Archiv enthält einen unzulässigen Symlink.")
            total += max(0, int(info.file_size))
            if total > model.max_extracted_bytes:
                raise SpeechError("Entpacktes Vosk-Modell überschreitet das Größenlimit.")
        with tempfile.TemporaryDirectory(prefix=".kiki-vosk-", dir=dest.parent) as tmp:
            root = Path(tmp)
            archive.extractall(root)
            extracted = root / model.id
            if not _model_ready(extracted, model):
                raise SpeechError("Vosk-Archiv enthält kein verwendbares deutsches Modell.")
            extracted.replace(dest)


def transcribe_wav(
    path: Path,
    *,
    model_dir: Path | None = None,
    model_id: str = VOSK_MODEL_ID,
) -> str:
    KaldiRecognizer, Model, SetLogLevel = _vosk_api()
    SetLogLevel(-1)
    spec = VOSK_MODELS.get(model_id)
    if spec is None:
        raise SpeechError(f"Unbekanntes Vosk-Modell: {model_id}")
    root = model_dir if model_dir is not None else vosk_model_dir(model_id)
    if not _model_ready(root, spec):
        raise SpeechError(f"Kein verwendbares deutsches Vosk-Modell unter {root}.")
    if not path.is_file():
        raise SpeechError(f"Aufnahme fehlt: {path}")
    try:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            width = wf.getsampwidth()
            if channels != 1 or width != 2 or wf.getcomptype() != "NONE":
                raise SpeechError(
                    "Aufnahmeformat ungültig; erwartet wird PCM16, Mono und 16 kHz."
                )
            model = Model(str(root))
            rec = KaldiRecognizer(model, wf.getframerate())
            # Do not enable word details. Vosk 0.3.45 formats confidence floats
            # with LC_NUMERIC=de_DE as decimal commas, producing invalid JSON.
            # KIKI only needs the recognized text anyway.
            chunks: list[str] = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    chunks.append(_result_text(rec.Result()))
            chunks.append(_result_text(rec.FinalResult()))
    except SpeechError:
        raise
    except (OSError, wave.Error) as exc:
        raise SpeechError(f"Aufnahme kann nicht gelesen werden: {exc}") from exc
    except Exception as exc:
        raise SpeechError(f"Spracherkennung fehlgeschlagen: {exc}") from exc
    text = " ".join(part.strip() for part in chunks if part.strip()).strip()
    return text


def _result_text(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SpeechError("Vosk lieferte ein ungültiges Erkennungsergebnis.") from exc
    if not isinstance(payload, dict):
        raise SpeechError("Vosk lieferte kein JSON-Objekt.")
    return str(payload.get("text") or "")
