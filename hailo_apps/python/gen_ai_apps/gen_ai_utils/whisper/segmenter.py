"""Cuts a stream of audio into phrases at pauses, using voice activity detection."""

from dataclasses import dataclass

import numpy as np

from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper.audio import SAMPLE_RATE
from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper.vad import FRAME_SAMPLES, FRAME_SECONDS

# Audio per window of the released Hailo-10H Whisper models; no phrase is longer.
DEFAULT_WINDOW_S = 10.0


@dataclass(frozen=True)
class SegmenterConfig:
    """
    Where phrases start and end.

    Attributes:
        speech_threshold: A frame whose speech probability reaches this is speech.
        silence_threshold: A frame below this is silence (between the two, a frame continues
            whatever came before it).
        pause_s: Silence after speech that ends a phrase.
        transmission_gap_s: Silence that ends a transmission.
        min_speech_s: A phrase with less speech than this is dropped (Whisper invents text for
            near-silent audio).
        pre_speech_s: Audio kept before the first speech frame of a phrase.
        cut_search_s: A phrase that reaches the length limit without a pause is cut at the
            least speech-like frame within this span before the limit.
        vad_reset_s: Silence after which the voice activity detector starts afresh, so that
            loud speech does not mask quieter speech after it.
    """

    speech_threshold: float = 0.5
    silence_threshold: float = 0.35
    pause_s: float = 0.4
    transmission_gap_s: float = 3.0
    min_speech_s: float = 0.5
    pre_speech_s: float = 0.3
    cut_search_s: float = 2.0
    vad_reset_s: float = 1.0


@dataclass
class Piece:
    """
    A phrase of audio ready for transcription.

    Attributes:
        audio: Mono float32 audio at 16 kHz.
        probabilities: Speech probability of each 32 ms frame of ``audio``.
        start_s: Start in the stream, in seconds.
        end_s: End in the stream, in seconds.
        new_transmission: True for the first phrase after a long silence.
    """

    audio: np.ndarray
    probabilities: np.ndarray
    start_s: float
    end_s: float
    new_transmission: bool


def least_speech_frame(probabilities: np.ndarray, lo: int, hi: int) -> int:
    """
    The frame in ``[lo, hi)`` least likely to be speech; the middle of a run of equal minima.

    Args:
        probabilities: Speech probability per frame.
        lo: First frame of the search span.
        hi: End of the search span (exclusive).

    Returns:
        A frame index in ``[lo, hi)``.
    """
    span = probabilities[lo:hi]
    if not len(span):
        return hi
    lowest = span.min()
    best = int(np.argmin(span))
    last = best
    while last + 1 < len(span) and span[last + 1] == lowest:
        last += 1
    return lo + (best + last) // 2


class Segmenter:
    """
    Buffers streamed audio and its speech probabilities, and hands out a phrase whenever a
    pause ends one or the phrase reaches ``max_piece_s``.
    """

    def __init__(
        self,
        vad,
        config: SegmenterConfig | None = None,
        max_piece_s: float = DEFAULT_WINDOW_S,
    ):
        """
        Args:
            vad: Voice activity detector with ``feed(audio) -> probability per frame``.
            config: Segmentation settings.
            max_piece_s: Initial length limit of a phrase: the model's window (the caller may
                lower it later).
        """
        self.vad = vad
        self.config = config or SegmenterConfig()
        self.max_piece_s = max_piece_s
        self._audio = np.zeros(0, np.float32)  # starts at a frame boundary
        self._probabilities = np.zeros(0, np.float32)  # one per complete frame of _audio
        self._start = 0  # stream sample index of _audio[0]
        self._frames_seen = 0  # frames scored so far in the stream
        self._last_speech_frame: int | None = None
        self._silent_run = 0  # consecutive frames below the silence threshold
        self._in_transmission = False
        self._new_transmission = True
        # Phrases dropped for holding too little speech; a caller transcribing one short clip
        # (e.g. a one-word reply) can still use them when nothing else was found.
        self.short_pieces: list[Piece] = []

    def feed(self, chunk: np.ndarray) -> list[Piece]:
        """
        Add audio and return the phrases it completes.

        Args:
            chunk: Mono float32 audio at 16 kHz (any length).

        Returns:
            Finished phrases, in order (often none).
        """
        chunk = np.asarray(chunk, np.float32).reshape(-1)
        self._audio = np.concatenate([self._audio, chunk])
        new = self.vad.feed(chunk)
        self._probabilities = np.concatenate([self._probabilities, new])
        reset_frames = max(1, int(self.config.vad_reset_s / FRAME_SECONDS))
        for i, p in enumerate(new):
            if p >= self.config.speech_threshold:
                self._last_speech_frame = self._frames_seen + i
            self._silent_run = self._silent_run + 1 if p < self.config.silence_threshold else 0
            if self._silent_run == reset_frames:
                self.vad.reset_state()
        self._frames_seen += len(new)
        pieces = []
        while (piece := self._next_piece()) is not None:
            if piece.audio.size:
                pieces.append(piece)
        return pieces

    def flush(self) -> list[Piece]:
        """
        Return the last phrase still buffered (call when the stream ends).

        Returns:
            The remaining phrase, if it holds enough speech.
        """
        frames = len(self._probabilities)
        piece = self._take(frames, include_tail=True) if frames else None
        self._start += len(self._audio)
        self._audio = np.zeros(0, np.float32)
        self._probabilities = np.zeros(0, np.float32)
        self.vad.reset()  # a partial frame was flushed: the next audio starts a fresh frame
        return [piece] if piece is not None and piece.audio.size else []

    def _speech_frames(self, probabilities: np.ndarray) -> int:
        return int((probabilities >= self.config.speech_threshold).sum())

    def _next_piece(self) -> Piece | None:
        cfg = self.config
        frames = len(self._probabilities)
        if not frames:
            return None
        if not self._speech_frames(self._probabilities):
            self._drop_silence(frames)
            return None

        limit = int(self.max_piece_s / FRAME_SECONDS)
        if frames >= limit:
            search = int(cfg.cut_search_s / FRAME_SECONDS)
            cut = least_speech_frame(self._probabilities, max(1, limit - search), limit)
            return self._take(cut)

        trailing = 0
        for p in self._probabilities[::-1]:
            if p >= cfg.silence_threshold:
                break
            trailing += 1
        if trailing >= int(cfg.pause_s / FRAME_SECONDS):
            return self._take(frames - trailing // 2)
        return None

    def _drop_silence(self, frames: int) -> None:
        """No speech buffered: keep a short lead-in, and notice the end of a transmission."""
        keep = min(frames, int(self.config.pre_speech_s / FRAME_SECONDS))
        drop = frames - keep
        self._audio = self._audio[drop * FRAME_SAMPLES :]
        self._probabilities = self._probabilities[drop:]
        self._start += drop * FRAME_SAMPLES
        silent_frames = self._frames_seen - (
            self._last_speech_frame + 1 if self._last_speech_frame is not None else 0
        )
        if self._in_transmission and silent_frames * FRAME_SECONDS >= (
            self.config.transmission_gap_s
        ):
            self._in_transmission = False
            self._new_transmission = True

    def _take(self, end_frame: int, include_tail: bool = False) -> Piece:
        """Remove the first ``end_frame`` frames from the buffer and return them as a phrase."""
        end_sample = len(self._audio) if include_tail else end_frame * FRAME_SAMPLES
        audio, probabilities = self._audio[:end_sample], self._probabilities[:end_frame]
        start_s = self._start / SAMPLE_RATE
        end_s = (self._start + end_sample) / SAMPLE_RATE
        self._audio = self._audio[end_sample:]
        self._probabilities = self._probabilities[end_frame:]
        self._start += end_sample
        speech_frames = self._speech_frames(probabilities)
        if speech_frames * FRAME_SECONDS < self.config.min_speech_s:
            if speech_frames:
                self.short_pieces.append(
                    Piece(audio, probabilities, start_s, end_s, self._new_transmission)
                )
            return Piece(np.zeros(0, np.float32), probabilities, start_s, end_s, False)
        piece = Piece(audio, probabilities, start_s, end_s, self._new_transmission)
        self.short_pieces.clear()
        self._in_transmission = True
        self._new_transmission = False
        return piece
