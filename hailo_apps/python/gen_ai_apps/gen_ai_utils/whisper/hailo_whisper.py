"""
Whisper speech-to-text on Hailo-10H: one entry point for recordings and live audio.

``Speech2Text`` decodes 10-second windows with a fixed budget of new tokens per window (32
for the released models). Given long or dense audio directly, it drops speech: when a
window's budget runs out before a sentence ends, the rest of that window is lost, and fixed
10-second cuts land inside words. ``HailoWhisper`` avoids both, for any length of audio and
any language, without per-language settings:

1. Voice activity detection finds speech and pauses; audio is cut at pauses into phrases no
   longer than the window, shorter when the speech so far needs more tokens per second than
   a window allows. Silence is never sent to the model (Whisper invents text for it).
2. A phrase is split in two and re-transcribed only when a returned segment fills the token
   budget, i.e. the model ran out of tokens before the segment ended.
3. With automatic language detection, the language detected for the first phrases of a
   transmission is confirmed from their text and then kept for the rest of it.
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from hailo_platform import HEF, VDevice
from hailo_platform.genai import Speech2Text, Speech2TextTask

from hailo_apps.python.core.common.core import resolve_hef_path
from hailo_apps.python.core.common.defines import HAILO10H_ARCH, WHISPER_CHAT_APP
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper.audio import SAMPLE_RATE
from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper.language import (
    LANGUAGES,
    UNSPACED_LANGUAGES,
    LanguageLock,
    script_language,
)
from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper.segmenter import (
    DEFAULT_WINDOW_S,
    Piece,
    Segmenter,
    SegmenterConfig,
    least_speech_frame,
)
from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper.vad import (
    FRAME_SAMPLES,
    SileroVad,
    resolve_vad_model,
)

logger = get_logger(__name__)

AUTO_LANGUAGE = "auto"
# A recording is fed to the live path in chunks of this length, so pauses are found the same way.
_FILE_CHUNK_S = 0.1
# How fast the token-rate estimate follows new phrases.
_RATE_SMOOTHING = 0.3
# Phrases shorter than this say little about the token rate.
_MIN_RATE_PHRASE_S = 2.0
# New tokens per window of the released Hailo-10H Whisper models (64-position decoder).
DEFAULT_TOKEN_BUDGET = 32
# Names in the Whisper HEFs: the encoder's network group, and the decoder's token input.
_ENCODER_GROUP_SUFFIX = "_encoder"
_TOKEN_INPUT_SUFFIX = "input_layer2"
# Whisper's log-mel frames are 10 ms apart: the encoder input's length gives the window.
_MEL_FRAME_S = 0.01


@dataclass(frozen=True)
class WhisperConfig:
    """
    Tuning of ``HailoWhisper``. The defaults suit the released Hailo-10H Whisper models.

    Attributes:
        segmenter: Where phrases start and end.
        min_piece_s: Shortest length limit for a phrase; pauses can still end one sooner.
        budget_fill: Share of the token budget a phrase is sized to use.
        full_budget_margin: A segment counts as cut off when its text has at least
            ``token_budget - full_budget_margin`` tokens (a window also spends a few tokens
            on timestamps).
        max_split_depth: Maximum number of times a phrase is halved.
        min_split_piece_s: Phrases shorter than twice this are never split.
        timeout_ms: Timeout of each ``Speech2Text`` call.
    """

    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    min_piece_s: float = 4.0
    budget_fill: float = 0.8
    full_budget_margin: int = 2
    max_split_depth: int = 3
    min_split_piece_s: float = 2.0
    timeout_ms: int = 60000


@dataclass
class Phrase:
    """
    One transcribed phrase.

    Attributes:
        text: The text.
        language: Language it was transcribed in; None when automatic detection had not
            settled yet (the model chose for this phrase alone).
        start_s: Start in the audio, in seconds.
        end_s: End in the audio, in seconds.
        new_transmission: True for the first phrase after a long silence.
    """

    text: str
    language: str | None
    start_s: float
    end_s: float
    new_transmission: bool


@dataclass
class Transcription:
    """
    The transcription of a recording.

    Attributes:
        text: The whole text.
        language: The language used, or detected; None if detection never settled.
        phrases: The phrases, in order.
    """

    text: str
    language: str | None
    phrases: list[Phrase]


def join_texts(texts: Iterable[str], language: str | None) -> str:
    """
    Join phrases: without spaces for Chinese and Japanese, with spaces otherwise.

    Args:
        texts: Text of each phrase.
        language: The language, or None if it was detected per phrase.

    Returns:
        The joined text.
    """
    texts = [t for t in texts if t]
    script = language or script_language("".join(texts))
    return ("" if script in UNSPACED_LANGUAGES else " ").join(texts)


def _read_model_limits(hef_path: str | Path) -> tuple[float, int]:
    """
    Read the window length and the token budget of a Whisper HEF from its input shapes.

    The encoder's input holds one window of log-mel frames. HailoRT 5.4 generates half as many
    new tokens per window as the decoder's token input has positions (the other half is
    reserved, as in openai/whisper, for previous-text tokens the runtime never sends). Any value
    that cannot be read falls back to the released models' 10 s and 32 tokens.

    Args:
        hef_path: Path to the Whisper HEF.

    Returns:
        Window length in seconds and new tokens per window.
    """
    window_s, token_budget = DEFAULT_WINDOW_S, DEFAULT_TOKEN_BUDGET
    try:
        hef = HEF(str(hef_path))
        for group in hef.get_network_group_names():
            for info in hef.get_input_vstream_infos(group):
                if group.endswith(_ENCODER_GROUP_SUFFIX):
                    window_s = info.shape[-2] * _MEL_FRAME_S
                elif info.name.endswith(_TOKEN_INPUT_SUFFIX):
                    token_budget = int(info.shape[-2]) // 2
    except Exception as e:  # HEF layout may differ between models and HailoRT versions
        logger.debug(f"Using the default Whisper window and token budget ({e})")
    logger.debug(f"Whisper window {window_s:.1f} s, {token_budget} tokens per window")
    return window_s, token_budget


def _parse_language(language: str | None) -> str | None:
    if language is None or language == AUTO_LANGUAGE:
        return None
    if language not in LANGUAGES:
        raise ValueError(
            f"Unknown language '{language}'. Use '{AUTO_LANGUAGE}' or one of: "
            + ", ".join(sorted(LANGUAGES))
        )
    return language


class HailoWhisper:
    """
    Whisper speech-to-text on Hailo-10H for audio of any length and any language.

    Example:
        whisper = HailoWhisper(vdevice)
        result = whisper.transcribe(load_audio("announcement.wav"), language="auto")
        print(result.language, result.text)

        live = whisper.live(language="fr")
        for chunk in microphone:
            for phrase in live.feed(chunk):
                print(phrase.text)
    """

    def __init__(
        self,
        vdevice: VDevice,
        hef_path: str | Path | None = None,
        config: WhisperConfig | None = None,
        vad_model_path: str | Path | None = None,
    ):
        """
        Args:
            vdevice: The Hailo device.
            hef_path: Whisper HEF; the app's default Whisper model when omitted.
            config: Tuning; the defaults suit the released Hailo-10H Whisper models.
            vad_model_path: Silero VAD model; the installed one when omitted (downloaded if
                missing).

        Raises:
            ValueError: If no Whisper HEF can be found.
            FileNotFoundError: If the Silero VAD model is missing and cannot be downloaded.
        """
        if hef_path is None:
            hef_path = resolve_hef_path(None, app_name=WHISPER_CHAT_APP, arch=HAILO10H_ARCH)
            if hef_path is None:
                raise ValueError(
                    "Could not resolve the Whisper HEF. Run: "
                    "hailo-download-resources --group whisper_chat"
                )
        self.config = config or WhisperConfig()
        self.window_s, self.token_budget = _read_model_limits(hef_path)
        self.vad_model_path = resolve_vad_model(vad_model_path)
        logger.debug(f"Loading Whisper model {hef_path}")
        self.speech2text = Speech2Text(vdevice, str(hef_path))

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None = "en",
        task: Speech2TextTask = Speech2TextTask.TRANSCRIBE,
    ) -> Transcription:
        """
        Transcribe (or translate to English) a recording of any length.

        Args:
            audio: Mono float32 audio at 16 kHz (see ``load_audio``).
            language: Whisper language code (``LANGUAGES``), or "auto" to detect it.
            task: TRANSCRIBE, or TRANSLATE to English.

        Returns:
            The text, the language and the phrases.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"Expected mono audio (a 1-D array), got shape {audio.shape}")
        live = self.live(language, task)
        step = int(_FILE_CHUNK_S * SAMPLE_RATE)
        phrases = []
        for i in range(0, len(audio), step):
            phrases += live.feed(audio[i : i + step])
        phrases += live.flush()
        if not phrases:  # a single short reply ("yes", "stop") holds too little speech for a phrase
            phrases = [live._transcribe(piece) for piece in live._segmenter.short_pieces]
        settled = live.language
        return Transcription(join_texts((p.text for p in phrases), settled), settled, phrases)

    def live(
        self,
        language: str | None = "en",
        task: Speech2TextTask = Speech2TextTask.TRANSCRIBE,
    ) -> "LiveTranscription":
        """
        Start transcribing a live stream (e.g. a microphone).

        Args:
            language: Whisper language code (``LANGUAGES``), or "auto" to detect it for each
                transmission.
            task: TRANSCRIBE, or TRANSLATE to English.

        Returns:
            A session: ``feed`` it audio chunks, ``flush`` it at the end.
        """
        return LiveTranscription(self, _parse_language(language), task)

    def release(self) -> None:
        """Release the model. Release the VDevice afterwards, not before."""
        if self.speech2text is not None:
            self.speech2text.release()
            self.speech2text = None


class LiveTranscription:
    """A live transcription session; create it with ``HailoWhisper.live``."""

    def __init__(self, whisper: HailoWhisper, language: str | None, task: Speech2TextTask):
        self._whisper = whisper
        self._config = whisper.config
        self._task = task
        self._fixed_language = language
        self._lock = LanguageLock(language=language)
        self._segmenter = Segmenter(
            SileroVad(whisper.vad_model_path), self._config.segmenter, whisper.window_s
        )
        self._tokens_per_s: float | None = None
        self.calls = 0
        self.splits = 0

    @property
    def language(self) -> str | None:
        """The fixed or detected language; None while automatic detection has not settled."""
        return self._lock.language

    def feed(self, chunk: np.ndarray) -> list[Phrase]:
        """
        Add audio and transcribe every phrase it completes.

        Args:
            chunk: Mono float32 audio at 16 kHz (any length).

        Returns:
            The phrases completed by this chunk, in order (often none).
        """
        return [self._transcribe(piece) for piece in self._segmenter.feed(chunk)]

    def flush(self) -> list[Phrase]:
        """
        Transcribe the audio still buffered (call when the stream ends).

        Returns:
            The remaining phrases.
        """
        return [self._transcribe(piece) for piece in self._segmenter.flush()]

    def _transcribe(self, piece: Piece) -> Phrase:
        if piece.new_transmission and self._fixed_language is None:
            self._lock.reset()
        language = self._lock.language
        splits_before = self.splits
        texts = self._transcribe_piece(piece.audio, piece.probabilities, language, depth=0)
        text = join_texts(texts, language)
        seconds = len(piece.audio) / SAMPLE_RATE
        self._update_token_rate(texts, seconds, overflowed=self.splits > splits_before)
        self._lock.observe(text, seconds)
        return Phrase(text, language, piece.start_s, piece.end_s, piece.new_transmission)

    def _transcribe_piece(
        self, audio: np.ndarray, probabilities: np.ndarray, language: str | None, depth: int
    ) -> list[str]:
        cfg = self._config
        self.calls += 1
        segments = self._whisper.speech2text.generate_all_segments(
            audio_data=np.ascontiguousarray(audio, dtype=np.float32),
            task=self._task,
            language=language,
            timeout_ms=cfg.timeout_ms,
        )
        # Text without a letter or digit ("...", "-") is what Whisper writes for non-speech.
        texts = [s.text.strip() for s in segments or [] if any(c.isalnum() for c in s.text)]
        too_short = len(audio) < 2 * cfg.min_split_piece_s * SAMPLE_RATE
        if not any(map(self._fills_budget, texts)) or depth >= cfg.max_split_depth or too_short:
            return texts
        self.splits += 1
        frames = len(probabilities)
        middle = least_speech_frame(probabilities, frames // 3, frames - frames // 3)
        cut = middle * FRAME_SAMPLES
        logger.debug("A segment used the whole token budget; transcribing the phrase as two")
        return self._transcribe_piece(
            audio[:cut], probabilities[:middle], language, depth + 1
        ) + self._transcribe_piece(audio[cut:], probabilities[middle:], language, depth + 1)

    def _fills_budget(self, text: str) -> bool:
        limit = self._whisper.token_budget - self._config.full_budget_margin
        return len(self._whisper.speech2text.tokenize(" " + text)) >= limit

    def _update_token_rate(self, texts: list[str], seconds: float, overflowed: bool) -> None:
        """Size the next phrases so that the speech heard so far fits one window's budget."""
        if seconds < _MIN_RATE_PHRASE_S or not texts:
            return
        tokens = sum(len(self._whisper.speech2text.tokenize(" " + t)) for t in texts)
        rate = tokens / seconds
        if overflowed:  # the first attempt ran out of tokens: at least a full budget was needed
            rate = max(rate, self._whisper.token_budget / seconds)
        if self._tokens_per_s is not None:
            rate = (1 - _RATE_SMOOTHING) * self._tokens_per_s + _RATE_SMOOTHING * rate
        self._tokens_per_s = rate
        target_tokens = self._config.budget_fill * (
            self._whisper.token_budget - self._config.full_budget_margin
        )
        self._segmenter.max_piece_s = float(
            np.clip(target_tokens / rate, self._config.min_piece_s, self._whisper.window_s)
        )
