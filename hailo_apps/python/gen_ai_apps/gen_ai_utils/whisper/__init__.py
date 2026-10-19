"""
Whisper speech-to-text on Hailo-10H.

``HailoWhisper`` is the entry point for recordings (``transcribe``) and live audio (``live``)
of any length and language. Prefer it to calling ``hailo_platform.genai.Speech2Text``
directly, which drops speech on long or dense audio (see ``hailo_whisper.py``).

    from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper import HailoWhisper, load_audio

    whisper = HailoWhisper(vdevice)
    result = whisper.transcribe(load_audio("speech.wav"), language="auto")
"""

from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper.audio import (
    SAMPLE_RATE,
    StreamResampler,
    load_audio,
    normalize,
    resample,
)
from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper.hailo_whisper import (
    AUTO_LANGUAGE,
    HailoWhisper,
    LiveTranscription,
    Phrase,
    Transcription,
    WhisperConfig,
)
from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper.language import LANGUAGES, language_name
from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper.segmenter import SegmenterConfig

__all__ = [
    "AUTO_LANGUAGE",
    "LANGUAGES",
    "SAMPLE_RATE",
    "HailoWhisper",
    "LiveTranscription",
    "Phrase",
    "SegmenterConfig",
    "StreamResampler",
    "Transcription",
    "WhisperConfig",
    "language_name",
    "load_audio",
    "normalize",
    "resample",
]
