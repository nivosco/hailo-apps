"""Audio in the format Whisper expects: mono float32 samples in [-1, 1) at 16 kHz."""

from math import ceil, gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

# Whisper's front end (log-mel spectrogram) is defined at 16 kHz; HailoRT does not expose it.
SAMPLE_RATE = 16000

# 8-bit WAV is unsigned: its silence level is the middle of the range.
_UINT8_MIDPOINT = 2**7
# soundfile delivers integer PCM of any width (8, 16, 24, 32 bit) scaled to int32.
_FLOAT_SUBTYPES = {"FLOAT", "DOUBLE"}
# resample_poly's default anti-aliasing filter spans this many samples on each side of the
# slower rate; a streaming resampler keeps that much context around each block.
_RESAMPLE_FILTER_HALF_WIDTH = 10


def normalize(samples: np.ndarray, channels: int = 1) -> np.ndarray:
    """
    Convert PCM samples to mono float32 in [-1, 1).

    Integer samples are divided by their full scale (2**15 for 16-bit, as openai/whisper and
    faster-whisper do); float samples are kept as they are. Interleaved channels are averaged.

    Args:
        samples: 1-D array of interleaved samples (uint8, int16, int32 or float).
        channels: Number of interleaved channels.

    Returns:
        Mono float32 audio.
    """
    samples = np.asarray(samples)
    if samples.dtype == np.uint8:
        audio = (samples.astype(np.float32) - _UINT8_MIDPOINT) / _UINT8_MIDPOINT
    elif np.issubdtype(samples.dtype, np.integer):
        full_scale = float(np.iinfo(samples.dtype).max) + 1.0
        audio = samples.astype(np.float32) / full_scale
    else:
        audio = samples.astype(np.float32)
    if channels > 1:
        audio = audio[: len(audio) // channels * channels].reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


def resample(audio: np.ndarray, rate: int) -> np.ndarray:
    """
    Resample mono audio from ``rate`` to ``SAMPLE_RATE``.

    Args:
        audio: Mono float32 audio.
        rate: Its sample rate in Hz.

    Returns:
        Mono float32 audio at 16 kHz (``audio`` itself when it already is).
    """
    if rate == SAMPLE_RATE:
        return audio
    factor = gcd(int(rate), SAMPLE_RATE)
    up, down = SAMPLE_RATE // factor, int(rate) // factor
    return resample_poly(audio, up, down).astype(np.float32)


def load_audio(path: str | Path) -> np.ndarray:
    """
    Load an audio file as mono float32 audio at 16 kHz.

    Reads WAV (integer PCM of any width, or float), FLAC and OGG. Other formats (MP3, WebM,
    ...) can be converted first, e.g. ``ffmpeg -i input.webm -ac 1 -ar 16000 output.wav``.

    Args:
        path: Path to the audio file, any sample rate and channel count.

    Returns:
        Mono float32 audio at 16 kHz.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file cannot be decoded (``soundfile.LibsndfileError``).
    """
    import soundfile  # gen-ai extra; only needed to read files

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    info = soundfile.info(str(path))
    dtype = "float32" if info.subtype in _FLOAT_SUBTYPES else "int32"
    samples, rate = soundfile.read(str(path), dtype=dtype)
    return resample(normalize(samples.reshape(-1), info.channels), rate)


class StreamResampler:
    """
    Resamples audio that arrives in chunks to 16 kHz without artefacts at chunk boundaries.

    Each block is resampled with enough surrounding input that the filter never sees an
    artificial edge; the output lags the input by that margin, which ``flush`` releases.
    """

    def __init__(self, rate: int):
        """
        Args:
            rate: Sample rate of the incoming audio in Hz.
        """
        factor = gcd(int(rate), SAMPLE_RATE)
        self.up, self.down = SAMPLE_RATE // factor, int(rate) // factor
        margin = _RESAMPLE_FILTER_HALF_WIDTH * max(self.up, self.down) / self.up
        self.pad = self.down * ceil(margin / self.down)  # input samples, a multiple of ``down``
        self.passthrough = self.up == self.down
        self._buffer = np.zeros(self.pad, np.float32)  # leading context: silence
        self._samples_in = 0
        self._samples_out = 0

    def feed(self, chunk: np.ndarray) -> np.ndarray:
        """
        Add audio and return what can already be resampled.

        Args:
            chunk: Mono float32 audio at the input rate.

        Returns:
            Mono float32 audio at 16 kHz (possibly empty).
        """
        chunk = np.asarray(chunk, np.float32).reshape(-1)
        if self.passthrough:
            return chunk
        self._samples_in += len(chunk)
        return self._resample(chunk)

    def _resample(self, chunk: np.ndarray) -> np.ndarray:
        self._buffer = np.concatenate([self._buffer, chunk])
        usable = (len(self._buffer) - 2 * self.pad) // self.down * self.down
        if usable <= 0:
            return np.zeros(0, np.float32)
        block = self._buffer[: usable + 2 * self.pad]
        out = resample_poly(block, self.up, self.down)
        first = self.pad * self.up // self.down
        out = out[first : first + usable * self.up // self.down]
        self._buffer = self._buffer[usable:]
        self._samples_out += len(out)
        return out.astype(np.float32)

    def flush(self) -> np.ndarray:
        """
        Return the remaining audio (call when the stream ends).

        Returns:
            Mono float32 audio at 16 kHz.
        """
        if self.passthrough:
            return np.zeros(0, np.float32)
        tail = self._resample(np.zeros(2 * self.pad, np.float32))
        # The padding that pushed the tail out is not audio: stop where one-shot resampling would.
        total = ceil(self._samples_in * self.up / self.down)
        tail = tail[: max(total - (self._samples_out - len(tail)), 0)]
        self._buffer = np.zeros(self.pad, np.float32)
        self._samples_in = self._samples_out = 0
        return tail
