"""
Voice activity detection with Silero VAD (MIT license), run on the CPU with onnxruntime.

faster-whisper uses the same model to decide where speech is before transcribing it.
"""

from pathlib import Path

import numpy as np

from hailo_apps.python.core.common.defines import (
    HAILO10H_ARCH,
    RESOURCES_ONNX_DIR_NAME,
    RESOURCES_ROOT_PATH_DEFAULT,
    WHISPER_CHAT_APP,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper.audio import SAMPLE_RATE

logger = get_logger(__name__)

# Listed in the onnx section of resources_config.yaml.
VAD_MODEL_NAME = "silero_vad.onnx"
VAD_MODEL_PATH = Path(RESOURCES_ROOT_PATH_DEFAULT) / RESOURCES_ONNX_DIR_NAME / VAD_MODEL_NAME

# The model scores 512-sample (32 ms) windows at 16 kHz, each preceded by the last 64 samples
# of the previous window, and carries a recurrent state between calls.
FRAME_SAMPLES = 512
FRAME_SECONDS = FRAME_SAMPLES / SAMPLE_RATE
_CONTEXT_SAMPLES = 64
_STATE_SHAPE = (2, 1, 128)


def _download_vad_model() -> None:
    """Fetch the model with the hailo-apps resource downloader, as the installer would."""
    from hailo_apps.installation.download_resources import (
        DEFAULT_RESOURCES_CONFIG_PATH,
        ResourceDownloader,
        load_config,
    )

    logger.info(f"Silero VAD model not found; downloading it to {VAD_MODEL_PATH.parent}")
    downloader = ResourceDownloader(
        config=load_config(Path(DEFAULT_RESOURCES_CONFIG_PATH)),
        hailo_arch=HAILO10H_ARCH,
        resource_root=Path(RESOURCES_ROOT_PATH_DEFAULT),
    )
    downloader.collect_onnx_by_tag(WHISPER_CHAT_APP)
    downloader.execute(parallel=False)


def resolve_vad_model(path: str | Path | None = None) -> Path:
    """
    Path to the Silero VAD model.

    The model is installed with the app's resources (``hailo-download-resources --group
    whisper_chat``); when it is missing it is downloaded the same way on first use.

    Args:
        path: Explicit model path, used instead of the installed model.

    Returns:
        Path to the model.

    Raises:
        FileNotFoundError: If the model is not found and cannot be downloaded.
    """
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"VAD model not found: {path}")
        return path
    if not VAD_MODEL_PATH.is_file():
        try:
            _download_vad_model()
        except Exception as e:
            logger.debug(f"Silero VAD model download failed: {e}")
    if not VAD_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Silero VAD model not found at {VAD_MODEL_PATH}. Run: "
            f"hailo-download-resources --group {WHISPER_CHAT_APP}"
        )
    return VAD_MODEL_PATH


class SileroVad:
    """
    Streaming speech probabilities: feed audio in any chunk sizes, get one probability per
    32 ms frame as soon as the frame is complete.
    """

    def __init__(self, model_path: str | Path | None = None):
        """
        Args:
            model_path: Path to ``silero_vad.onnx``; resolved with ``resolve_vad_model``
                when omitted.
        """
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1  # one 32 ms frame at a time: threads only add overhead
        self._session = onnxruntime.InferenceSession(
            str(resolve_vad_model(model_path)),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._sample_rate = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Start a new stream."""
        self._state = np.zeros(_STATE_SHAPE, np.float32)
        self._context = np.zeros(_CONTEXT_SAMPLES, np.float32)
        self._pending = np.zeros(0, np.float32)

    def feed(self, audio: np.ndarray) -> np.ndarray:
        """
        Add audio and score every frame it completes.

        Args:
            audio: Mono float32 audio at 16 kHz (any length).

        Returns:
            Speech probability (0..1) of each completed 32 ms frame, in order.
        """
        self._pending = np.concatenate([self._pending, np.asarray(audio, np.float32).reshape(-1)])
        frames = len(self._pending) // FRAME_SAMPLES
        probabilities = np.empty(frames, np.float32)
        for i in range(frames):
            frame = self._pending[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            window = np.concatenate([self._context, frame])[np.newaxis, :]
            probability, self._state = self._session.run(
                None, {"input": window, "state": self._state, "sr": self._sample_rate}
            )
            probabilities[i] = float(probability.reshape(-1)[0])
            self._context = frame[-_CONTEXT_SAMPLES:]
        self._pending = self._pending[frames * FRAME_SAMPLES :]
        return probabilities
