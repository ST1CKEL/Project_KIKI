from __future__ import annotations

import dataclasses
import hashlib
import io
import os
import struct
import subprocess
import sys
import types
import wave
import zipfile
from pathlib import Path

import pytest

import kiki.voice.stt as stt_module
from kiki.voice.stt import (
    VOSK_MODEL_ID,
    SpeechError,
    _extract_model_archive,
    _result_text,
    _vosk_api,
    ensure_vosk_model,
    transcribe_wav,
)


def _spec(model_id: str = VOSK_MODEL_ID) -> stt_module.VoskModel:
    return stt_module.VOSK_MODELS[model_id]


def _patch_spec(monkeypatch: pytest.MonkeyPatch, model_id: str = VOSK_MODEL_ID, **changes):
    spec = dataclasses.replace(stt_module.VOSK_MODELS[model_id], **changes)
    monkeypatch.setattr(
        stt_module, "VOSK_MODELS", {**stt_module.VOSK_MODELS, model_id: spec}
    )
    return spec


def _model_zip_payload(model_id: str = VOSK_MODEL_ID) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for relative in _spec(model_id).required_files:
            if relative in stt_module._EMPTY_MODEL_FILES:
                payload = b""
            else:
                payload = b"model" if relative == "am/final.mdl" else b"config"
            archive.writestr(f"{model_id}/{relative}", payload)
    return output.getvalue()


def _write_model_tree(root: Path, model_id: str = VOSK_MODEL_ID) -> None:
    for relative in _spec(model_id).required_files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"" if relative in stt_module._EMPTY_MODEL_FILES else b"model"
        path.write_bytes(payload)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.headers = {"content-length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self):
        yield self._payload


class _FakeClient:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def stream(self, _method: str, _url: str) -> _FakeResponse:
        return _FakeResponse(self._payload)


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(struct.pack("<h", 0) * 8000)


def test_transcribe_does_not_enable_locale_broken_word_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    _write_model_tree(model_dir)
    wav = tmp_path / "take.wav"
    _wav(wav)

    class FakeRecognizer:
        def __init__(self, _model: object, _rate: int) -> None:
            pass

        def SetWords(self, _enabled: bool) -> None:  # noqa: N802 - Vosk API
            raise AssertionError("word details must stay disabled")

        def AcceptWaveform(self, _data: bytes) -> bool:  # noqa: N802
            return False

        def Result(self) -> str:  # noqa: N802
            return '{"text": ""}'

        def FinalResult(self) -> str:  # noqa: N802
            return '{"text": "hallo kiki"}'

    # Patch the runtime-selection seam, not sys.modules["vosk"].
    # `_vosk_api` prefers Fedora's libvosk via kiki.voice.vosk_ffi and only falls
    # back to the pip package. Faking the pip module therefore had no effect
    # wherever vosk-api-devel is installed — which the RPM requires — so this
    # test silently stopped guarding anything on exactly the target systems.
    monkeypatch.setattr(
        stt_module,
        "_vosk_api",
        lambda: (FakeRecognizer, lambda _root: object(), lambda _level: None),
    )

    assert transcribe_wav(wav, model_dir=model_dir) == "hallo kiki"


def test_invalid_vosk_json_becomes_speech_error() -> None:
    with pytest.raises(SpeechError, match="ungültiges Erkennungsergebnis"):
        _result_text('{"conf": 1,000000, "text": "hallo"}')


def test_cffi_runtime_rejects_unrelated_shared_library() -> None:
    env = os.environ.copy()
    env["KIKI_VOSK_LIBRARY"] = "libc.so.6"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import kiki.voice.vosk_ffi",  # must fail before reporting readiness
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "vosk-api-devel" in result.stderr


def test_fedora_cffi_runtime_is_preferred_over_upstream_python_package(monkeypatch) -> None:
    fallback = types.SimpleNamespace(
        KaldiRecognizer=object(),
        Model=object(),
        SetLogLevel=object(),
    )
    upstream = types.SimpleNamespace(
        KaldiRecognizer=object(),
        Model=object(),
        SetLogLevel=object(),
    )
    monkeypatch.setitem(sys.modules, "vosk", upstream)
    monkeypatch.setitem(sys.modules, "kiki.voice.vosk_ffi", fallback)

    assert _vosk_api() == (fallback.KaldiRecognizer, fallback.Model, fallback.SetLogLevel)


def test_transcribe_rejects_missing_custom_model(tmp_path: Path) -> None:
    with pytest.raises(SpeechError, match="Kein verwendbares"):
        transcribe_wav(tmp_path / "missing.wav", model_dir=tmp_path / "model")


def test_model_archive_is_published_atomically(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.zip"
    archive_path.write_bytes(_model_zip_payload())
    dest = tmp_path / VOSK_MODEL_ID

    _extract_model_archive(archive_path, dest, _spec())

    assert (dest / "am/final.mdl").read_bytes() == b"model"
    assert (dest / "conf/model.conf").read_bytes() == b"config"
    assert not list(tmp_path.glob(".kiki-vosk-*"))


def test_model_ready_rejects_partial_or_empty_tree(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    (model_dir / "am").mkdir(parents=True)
    (model_dir / "am/final.mdl").write_bytes(b"model")
    (model_dir / "conf").mkdir()
    (model_dir / "conf/model.conf").write_bytes(b"config")

    assert not stt_module._model_ready(model_dir, _spec())

    _write_model_tree(model_dir)
    assert (model_dir / "ivector/online_cmvn.conf").stat().st_size == 0
    assert stt_module._model_ready(model_dir, _spec())

    (model_dir / "graph/Gr.fst").write_bytes(b"")
    assert not stt_module._model_ready(model_dir, _spec())


def test_large_model_requires_hclg_layout(tmp_path: Path) -> None:
    large = _spec(stt_module.VOSK_MODEL_LARGE)
    assert "graph/HCLG.fst" in large.required_files
    assert "graph/Gr.fst" not in large.required_files

    model_dir = tmp_path / "model"
    _write_model_tree(model_dir, stt_module.VOSK_MODEL_LARGE)
    assert stt_module._model_ready(model_dir, large)

    # The small model's two-graph layout is not a valid large model.
    (model_dir / "graph/HCLG.fst").unlink()
    assert not stt_module._model_ready(model_dir, large)


def test_unknown_model_id_fails_closed(tmp_path: Path) -> None:
    assert stt_module.vosk_model_ready("totally-made-up") is False
    with pytest.raises(SpeechError, match="Unbekanntes Vosk-Modell"):
        ensure_vosk_model("totally-made-up")
    assert not (tmp_path / "data").exists()


def test_model_download_requires_pinned_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _model_zip_payload()
    monkeypatch.setattr(stt_module, "user_data_dir", lambda: tmp_path / "data")
    _patch_spec(monkeypatch, sha256=hashlib.sha256(payload).hexdigest())
    import httpx

    monkeypatch.setattr(httpx, "Client", lambda **_kwargs: _FakeClient(payload))

    dest = ensure_vosk_model()

    assert (dest / "am/final.mdl").read_bytes() == b"model"


def test_model_download_rejects_wrong_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _model_zip_payload()
    monkeypatch.setattr(stt_module, "user_data_dir", lambda: tmp_path / "data")
    _patch_spec(monkeypatch, sha256="0" * 64)
    import httpx

    monkeypatch.setattr(httpx, "Client", lambda **_kwargs: _FakeClient(payload))

    with pytest.raises(SpeechError, match="SHA-256"):
        ensure_vosk_model()
    assert not (tmp_path / "data/vosk" / VOSK_MODEL_ID).exists()
    assert not list((tmp_path / "data/vosk").glob("*.part"))


@pytest.mark.parametrize(
    "member",
    [
        "../escape",
        f"{VOSK_MODEL_ID}/../../escape",
        "other-model/am/final.mdl",
    ],
)
def test_model_archive_rejects_unsafe_members(tmp_path: Path, member: str) -> None:
    archive_path = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, b"bad")
    with pytest.raises(SpeechError):
        _extract_model_archive(archive_path, tmp_path / VOSK_MODEL_ID, _spec())
    assert not (tmp_path / "escape").exists()


def test_model_archive_rejects_excessive_member_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive_path = tmp_path / "too-many.zip"
    monkeypatch.setattr(stt_module, "_MAX_ARCHIVE_MEMBERS", 2)
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(f"{VOSK_MODEL_ID}/am/final.mdl", b"model")
        archive.writestr(f"{VOSK_MODEL_ID}/conf/model.conf", b"config")
        archive.writestr(f"{VOSK_MODEL_ID}/extra", b"extra")
    with pytest.raises(SpeechError, match="viele Dateien"):
        _extract_model_archive(archive_path, tmp_path / VOSK_MODEL_ID, _spec())
