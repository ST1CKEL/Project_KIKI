"""Small CFFI binding for Fedora's system ``libvosk`` runtime.

Fedora 44 ships the Vosk C library but not the upstream Python package. KIKI
only needs the basic offline recognizer API, so this adapter keeps speech input
on the distro-maintained native runtime without a user-level Pip installation.
"""

from __future__ import annotations

import os

from cffi import FFI

_ffi = FFI()
_ffi.cdef(
    """
    typedef struct VoskModel VoskModel;
    typedef struct VoskRecognizer VoskRecognizer;

    VoskModel *vosk_model_new(const char *model_path);
    void vosk_model_free(VoskModel *model);
    int vosk_model_find_word(VoskModel *model, const char *word);

    VoskRecognizer *vosk_recognizer_new(VoskModel *model, float sample_rate);
    int vosk_recognizer_accept_waveform(
        VoskRecognizer *recognizer, const char *data, int length
    );
    const char *vosk_recognizer_result(VoskRecognizer *recognizer);
    const char *vosk_recognizer_final_result(VoskRecognizer *recognizer);
    void vosk_recognizer_reset(VoskRecognizer *recognizer);
    void vosk_recognizer_free(VoskRecognizer *recognizer);

    void vosk_set_log_level(int log_level);
    """
)

_REQUIRED_SYMBOLS = (
    "vosk_model_new",
    "vosk_model_free",
    "vosk_recognizer_new",
    "vosk_recognizer_accept_waveform",
    "vosk_recognizer_result",
    "vosk_recognizer_final_result",
    "vosk_recognizer_free",
    "vosk_set_log_level",
)

# Only the wake word needs these. An older libvosk without them must still load
# for push-to-talk, so they are probed separately instead of gating the import.
_WAKE_SYMBOLS = (
    "vosk_model_find_word",
    "vosk_recognizer_reset",
)

def _library_candidates() -> tuple[str, ...]:
    override = os.environ.get("KIKI_VOSK_LIBRARY", "").strip()
    if override:
        return (override,)
    return ("libvosk.so",)


def _load_library():
    last_error: OSError | AttributeError | None = None
    for candidate in _library_candidates():
        try:
            library = _ffi.dlopen(candidate)
            # CFFI resolves symbols lazily. Touch every API entry now so the
            # health check cannot accept an unrelated or incomplete library.
            for symbol in _REQUIRED_SYMBOLS:
                getattr(library, symbol)
            return library
        except (OSError, AttributeError) as exc:
            last_error = exc
    raise ImportError("Fedora-Paket vosk-api-devel fehlt oder ist beschädigt") from last_error


_lib = _load_library()


def wake_support_available() -> bool:
    """Whether this libvosk offers grammar mode and partial results."""
    try:
        for symbol in _WAKE_SYMBOLS:
            getattr(_lib, symbol)
    except AttributeError:
        return False
    return True


class Model:
    def __init__(self, model_path: str) -> None:
        self._handle = _lib.vosk_model_new(model_path.encode("utf-8"))
        if self._handle == _ffi.NULL:
            raise RuntimeError("Vosk-Modell konnte nicht geladen werden.")

    def __del__(self) -> None:
        handle = getattr(self, "_handle", _ffi.NULL)
        if handle != _ffi.NULL:
            _lib.vosk_model_free(handle)
            self._handle = _ffi.NULL

    def FindWord(self, word: str) -> int:  # noqa: N802 - Vosk compatibility
        """Symbol id of `word`, or -1 when the lexicon does not contain it."""
        return int(_lib.vosk_model_find_word(self._handle, word.encode("utf-8")))


class KaldiRecognizer:
    def __init__(self, model: Model, sample_rate: float) -> None:
        self._handle = _lib.vosk_recognizer_new(model._handle, float(sample_rate))
        if self._handle == _ffi.NULL:
            raise RuntimeError("Vosk-Recognizer konnte nicht erstellt werden.")

    def __del__(self) -> None:
        handle = getattr(self, "_handle", _ffi.NULL)
        if handle != _ffi.NULL:
            _lib.vosk_recognizer_free(handle)
            self._handle = _ffi.NULL

    def AcceptWaveform(self, data: bytes) -> int:  # noqa: N802 - Vosk compatibility
        result = _lib.vosk_recognizer_accept_waveform(self._handle, data, len(data))
        if result < 0:
            raise RuntimeError("Vosk konnte den Audioblock nicht verarbeiten.")
        return int(result)

    def Result(self) -> str:  # noqa: N802 - Vosk compatibility
        return _decode(_lib.vosk_recognizer_result(self._handle))

    def FinalResult(self) -> str:  # noqa: N802 - Vosk compatibility
        return _decode(_lib.vosk_recognizer_final_result(self._handle))

    def Reset(self) -> None:  # noqa: N802 - Vosk compatibility
        _lib.vosk_recognizer_reset(self._handle)


def SetLogLevel(level: int) -> None:  # noqa: N802 - Vosk compatibility
    _lib.vosk_set_log_level(int(level))


def _decode(value: object) -> str:
    if value == _ffi.NULL:
        raise RuntimeError("Vosk lieferte kein Ergebnis.")
    return _ffi.string(value).decode("utf-8")
