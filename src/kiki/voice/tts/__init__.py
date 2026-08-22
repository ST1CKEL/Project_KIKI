"""KIKI's TTS layer: neutral provider contract, policy and chunking.

Importing this package must stay free of torch, CUDA, GTK and audio devices —
the UI imports it, and nothing here may occupy a GPU as a side effect.
"""

from kiki.voice.tts.chunker import ChunkerConfig, StreamingChunker, boundaries, is_speakable
from kiki.voice.tts.controller import (
    DEFAULT_PREFETCH,
    PlaybackState,
    VoicePlaybackController,
)
from kiki.voice.tts.fake import FakeTTSProvider, NullTTSProvider
from kiki.voice.tts.models import (
    DEFAULT_AUDIO_FORMAT,
    DEFAULT_CHANNELS,
    DEFAULT_SAMPLE_RATE,
    AudioChunk,
    TTSError,
    TTSGenerationResult,
    TTSHealth,
    TTSProviderCapabilities,
    TTSProviderStatus,
    TTSRequest,
)
from kiki.voice.tts.playback import AudioSink, FakeAudioSink
from kiki.voice.tts.policy import (
    SpeechPlan,
    VoiceMode,
    VoicePolicyConfig,
    VoiceResponsePolicy,
)
from kiki.voice.tts.provider import TTSProvider

__all__ = [
    "DEFAULT_AUDIO_FORMAT",
    "DEFAULT_CHANNELS",
    "DEFAULT_PREFETCH",
    "DEFAULT_SAMPLE_RATE",
    "AudioChunk",
    "AudioSink",
    "ChunkerConfig",
    "FakeAudioSink",
    "FakeTTSProvider",
    "NullTTSProvider",
    "PlaybackState",
    "SpeechPlan",
    "StreamingChunker",
    "TTSError",
    "TTSGenerationResult",
    "TTSHealth",
    "TTSProvider",
    "TTSProviderCapabilities",
    "TTSProviderStatus",
    "TTSRequest",
    "VoiceMode",
    "VoicePlaybackController",
    "VoicePolicyConfig",
    "VoiceResponsePolicy",
    "boundaries",
    "is_speakable",
]
