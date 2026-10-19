"""Unit tests for HailoWhisper (gen_ai_utils/whisper). No Hailo device needed."""

import enum
import sys
import types
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

if "hailo_platform" not in sys.modules:
    try:
        import hailo_platform  # noqa: F401
    except ImportError:  # run the pure-Python logic anywhere: stub the HailoRT bindings

        def _placeholder(name: str) -> type:
            return type(name, (), {})

        platform = types.ModuleType("hailo_platform")
        platform.__getattr__ = _placeholder  # any other HailoRT name: an empty class
        genai = types.ModuleType("hailo_platform.genai")
        genai.__getattr__ = _placeholder
        genai.Speech2TextTask = enum.Enum("Speech2TextTask", "TRANSCRIBE TRANSLATE")
        platform.genai = genai
        sys.modules["hailo_platform"] = platform
        sys.modules["hailo_platform.genai"] = genai

from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper import (
    SAMPLE_RATE,
    HailoWhisper,
    StreamResampler,
    WhisperConfig,
    hailo_whisper,
    language as lang,
    language_name,
    load_audio,
    normalize,
    resample,
)
from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper.hailo_whisper import DEFAULT_TOKEN_BUDGET
from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper.segmenter import DEFAULT_WINDOW_S
from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper.vad import FRAME_SAMPLES

pytestmark = pytest.mark.genai

SPEECH_RMS = 0.1


# ----------------------------------------------------------------------------- fakes


@dataclass
class FakeSegment:
    text: str


class FakeSpeech2Text:
    """Stands in for hailo_platform.genai.Speech2Text: one segment per call.

    A call on more than ``full_above_s`` seconds returns a segment whose text fills the token
    budget (one token per word), as the device does when it runs out of tokens.
    """

    def __init__(self, full_above_s: float = 1e9, words_when_full: int = 31):
        self.full_above_s = full_above_s
        self.words_when_full = words_when_full
        self.call_seconds: list[float] = []
        self.call_languages: list[str | None] = []

    def generate_all_segments(self, audio_data, task, language, timeout_ms):
        seconds = len(audio_data) / SAMPLE_RATE
        self.call_seconds.append(seconds)
        self.call_languages.append(language)
        n = len(self.call_seconds)
        if seconds > self.full_above_s:
            return [FakeSegment(" ".join(f"w{n}_{i}" for i in range(self.words_when_full)))]
        return [FakeSegment(f"phrase{n}")]

    def tokenize(self, text):
        return text.split()

    def release(self):
        pass


class EnergyVad:
    """Stands in for Silero VAD: a 32 ms frame is speech when it is loud."""

    def __init__(self, *_):
        self.reset()

    def reset(self):
        self._pending = np.zeros(0, np.float32)

    def feed(self, audio):
        self._pending = np.concatenate([self._pending, np.asarray(audio, np.float32)])
        frames = len(self._pending) // FRAME_SAMPLES
        rms = np.sqrt(
            (self._pending[: frames * FRAME_SAMPLES] ** 2).reshape(-1, FRAME_SAMPLES).mean(axis=1)
        )
        self._pending = self._pending[frames * FRAME_SAMPLES :]
        return (rms > SPEECH_RMS / 4).astype(np.float32)


@pytest.fixture(autouse=True)
def energy_vad(monkeypatch):
    monkeypatch.setattr(hailo_whisper, "SileroVad", EnergyVad)


def make_whisper(fake: FakeSpeech2Text, config: WhisperConfig | None = None) -> HailoWhisper:
    whisper = object.__new__(HailoWhisper)
    whisper.config = config or WhisperConfig()
    whisper.window_s, whisper.token_budget = DEFAULT_WINDOW_S, DEFAULT_TOKEN_BUDGET
    whisper.vad_model_path = None
    whisper.speech2text = fake
    return whisper


def speech(seconds: float, seed: int = 0) -> np.ndarray:
    """Noise standing in for speech."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, SPEECH_RMS, int(seconds * SAMPLE_RATE)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), np.float32)


def stream(live, audio: np.ndarray, chunk_s: float = 0.1) -> list:
    step = int(chunk_s * SAMPLE_RATE)
    phrases = []
    for i in range(0, len(audio), step):
        phrases += live.feed(audio[i : i + step])
    return phrases + live.flush()


@pytest.fixture
def identify_as_french(monkeypatch):
    monkeypatch.setattr(lang, "identify_language", lambda text: "fr")


# ----------------------------------------------------------------------------- audio


def test_normalize_uses_the_int16_full_scale():
    audio = normalize(np.array([-(2**15), 2**14, 2**15 - 1], np.int16))
    assert audio.dtype == np.float32
    assert audio[0] == -1.0 and audio[1] == 0.5 and audio[2] < 1.0


def test_normalize_centres_8_bit_audio_and_mixes_channels():
    assert normalize(np.array([0, 128, 255], np.uint8))[1] == 0.0
    stereo = np.array([2**14, -(2**14), 2**13, 2**13], np.int16)
    assert normalize(stereo, channels=2).tolist() == [0.0, 0.25]


def test_load_audio_converts_to_16_khz_mono(tmp_path):
    rate, seconds = 44100, 1.0
    left = (np.sin(np.linspace(0, 440 * 2 * np.pi, int(rate * seconds))) * 2**14).astype("<i2")
    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(2)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(np.stack([left, left], axis=1).tobytes())

    audio = load_audio(path)

    assert audio.dtype == np.float32
    assert len(audio) == pytest.approx(SAMPLE_RATE * seconds, abs=2)
    assert np.abs(audio).max() == pytest.approx(0.5, abs=0.02)


@pytest.mark.parametrize(
    ("subtype", "full_scale"), [("PCM_24", 2**23), ("FLOAT", 1.0), ("PCM_U8", 2**7)]
)
def test_load_audio_reads_24_bit_float_and_8_bit_wav(tmp_path, subtype, full_scale):
    soundfile = pytest.importorskip("soundfile")
    path = tmp_path / f"{subtype}.wav"
    soundfile.write(str(path), np.full(SAMPLE_RATE, 0.5, np.float32), SAMPLE_RATE, subtype=subtype)

    audio = load_audio(path)

    assert len(audio) == SAMPLE_RATE
    assert audio[SAMPLE_RATE // 2] == pytest.approx(0.5, abs=1.0 / full_scale + 1e-6)


def test_load_audio_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"missing\.wav"):
        load_audio(tmp_path / "missing.wav")


@pytest.mark.parametrize("rate", [48000, 44100, 22050, 11025, 8000])
def test_stream_resampler_matches_one_shot_resampling(rate):
    rng = np.random.default_rng(0)
    audio = rng.normal(0.0, 0.1, rate * 2).astype(np.float32)
    resampler = StreamResampler(rate)
    parts, i = [], 0
    while i < len(audio):
        n = int(rng.integers(100, 3000))
        parts.append(resampler.feed(audio[i : i + n]))
        i += n
    streamed = np.concatenate([*parts, resampler.flush()])
    reference = resample(audio, rate)
    assert len(streamed) == len(reference)
    edge = 200
    np.testing.assert_allclose(
        streamed[edge : len(reference) - edge], reference[edge:-edge], atol=1e-6
    )


# ----------------------------------------------------------------------------- language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("合金基本上由兩種或以上金屬混合而成", "zh"),
        ("这份文件将提及边界争端", "zh"),
        ("잊지 마세요 원소는 주기율표에 많습니다", "ko"),
        ("ローマは イタリアの 首都です", "ja"),
    ],
)
def test_east_asian_languages_are_identified_by_script(text, expected):
    assert lang.identify_language(text) == expected


def test_language_lock_needs_agreeing_phrases(identify_as_french):
    lock = lang.LanguageLock()
    lock.observe("bonjour à tous", 3.0)
    assert lock.language is None  # one short phrase is not enough
    lock.observe("mesdames et messieurs", 4.0)
    assert lock.language == "fr"
    lock.reset()
    assert lock.language is None


def test_language_lock_ignores_a_disagreeing_minority(monkeypatch):
    answers = iter(["en", "fr", "fr"])
    monkeypatch.setattr(lang, "identify_language", lambda text: next(answers))
    lock = lang.LanguageLock()
    lock.observe("i am going to make a video", 3.0)  # a hallucination on a short phrase
    lock.observe("mesdames et messieurs", 4.0)
    assert lock.language is None
    lock.observe("bienvenue à bord de ce vol", 5.0)
    assert lock.language == "fr"


def test_fixed_language_is_never_replaced(identify_as_french):
    lock = lang.LanguageLock(language="de")
    lock.observe("x" * 100, 20.0)
    assert lock.language == "de"


def test_language_names_and_validation():
    assert language_name("sv") == "Swedish (sv)"
    assert language_name(None) == "automatic detection"
    with pytest.raises(ValueError, match="Unknown language"):
        make_whisper(FakeSpeech2Text()).live("xx")


# ----------------------------------------------------------------------------- transcription


def test_phrases_end_at_pauses():
    audio = np.concatenate([speech(3.0), silence(0.8), speech(4.0, seed=1), silence(0.5)])
    fake = FakeSpeech2Text()
    phrases = stream(make_whisper(fake).live("en"), audio)

    assert [p.text for p in phrases] == ["phrase1", "phrase2"]
    assert 3.0 < phrases[0].end_s < 3.8  # inside the pause
    assert phrases[0].new_transmission and not phrases[1].new_transmission


def test_long_speech_is_cut_into_pieces_that_fit_the_window():
    fake = FakeSpeech2Text()
    make_whisper(fake).transcribe(speech(25.0), language="en")

    assert all(seconds <= DEFAULT_WINDOW_S + 1e-6 for seconds in fake.call_seconds)
    assert sum(fake.call_seconds) == pytest.approx(25.0, abs=0.1)


def test_a_short_reply_is_still_transcribed():
    fake = FakeSpeech2Text()
    audio = np.concatenate([silence(0.5), speech(0.3), silence(0.5)])

    result = make_whisper(fake).transcribe(audio, language="en")

    assert result.text == "phrase1"


def test_a_short_blip_in_longer_speech_is_dropped():
    fake = FakeSpeech2Text()
    audio = np.concatenate([speech(3.0), silence(1.0), speech(0.2, seed=1), silence(1.0)])

    result = make_whisper(fake).transcribe(audio, language="en")

    assert [p.text for p in result.phrases] == ["phrase1"]
    assert len(fake.call_seconds) == 1


def test_a_session_can_continue_after_flush():
    fake = FakeSpeech2Text()
    live = make_whisper(fake).live("en")
    first = stream(live, np.concatenate([speech(2.0), silence(0.01)]))  # ends inside a frame
    second = stream(live, np.concatenate([speech(2.0, seed=1), silence(0.8)]))

    assert len(first) == len(second) == 1
    assert second[0].start_s == pytest.approx(first[0].end_s, abs=0.05)
    assert second[0].end_s - second[0].start_s == pytest.approx(2.4, abs=0.4)


def test_silence_is_not_transcribed():
    audio = np.concatenate([speech(5.0), silence(10.0), speech(5.0, seed=1)])
    fake = FakeSpeech2Text()
    result = make_whisper(fake).transcribe(audio, language="en")

    assert len(fake.call_seconds) == 2
    assert sum(fake.call_seconds) < 12.0
    assert result.text == "phrase1 phrase2"


def test_a_long_silence_starts_a_new_transmission():
    audio = np.concatenate([speech(3.0), silence(4.0), speech(3.0, seed=1)])
    phrases = stream(make_whisper(FakeSpeech2Text()).live("en"), audio)
    assert [p.new_transmission for p in phrases] == [True, True]


def test_phrase_is_split_only_when_a_segment_fills_the_token_budget():
    audio = np.concatenate(
        [speech(4.5), silence(0.2), speech(4.5, seed=1)]
    )  # pause too short to cut
    fake = FakeSpeech2Text(full_above_s=6.0)
    live = make_whisper(fake).live("ar")
    phrases = stream(live, audio)

    assert live.splits == 1
    assert len(fake.call_seconds) == 3  # the whole phrase, then its two halves
    assert fake.call_seconds[1] == pytest.approx(4.6, abs=0.2)  # split in the short pause
    assert [p.text for p in phrases] == ["phrase2 phrase3"]


def test_no_split_when_the_budget_is_not_reached():
    fake = FakeSpeech2Text(full_above_s=0.0, words_when_full=10)
    live = make_whisper(fake).live("en")
    stream(live, speech(9.0))
    assert live.splits == 0
    assert len(fake.call_seconds) == 1


def test_split_depth_is_bounded():
    fake = FakeSpeech2Text(full_above_s=0.0)
    live = make_whisper(fake, WhisperConfig(max_split_depth=2)).live("zh")
    stream(live, speech(9.0))
    assert live.splits <= 3  # 1 at depth 0, 2 at depth 1


def test_dense_speech_gets_shorter_pieces():
    fake = FakeSpeech2Text(full_above_s=0.0, words_when_full=26)  # ~26 tokens per piece
    make_whisper(fake).transcribe(speech(40.0), language="ar")
    assert fake.call_seconds[0] == pytest.approx(10.0, abs=2.1)
    assert max(fake.call_seconds[2:-1]) < 9.0


def test_chinese_and_japanese_are_joined_without_spaces():
    audio = np.concatenate([speech(4.0), silence(1.0), speech(4.0, seed=1)])
    result = make_whisper(FakeSpeech2Text()).transcribe(audio, language="zh")
    assert result.text == "phrase1phrase2"


def test_auto_language_locks_after_the_first_phrases(identify_as_french):
    audio = np.concatenate([speech(9.0), silence(1.0), speech(9.0, seed=1), silence(1.0)])
    fake = FakeSpeech2Text(full_above_s=0.0, words_when_full=8)
    result = make_whisper(fake).transcribe(audio, language="auto")

    assert fake.call_languages[0] is None
    assert fake.call_languages[-1] == "fr"
    assert result.language == "fr"


def test_rejects_multichannel_audio():
    with pytest.raises(ValueError, match="mono"):
        make_whisper(FakeSpeech2Text()).transcribe(np.zeros((SAMPLE_RATE, 2), np.float32))


def test_model_limits_are_read_from_the_hef_input_shapes(monkeypatch):
    @dataclass
    class Info:
        name: str
        shape: tuple

    class FakeHef:  # the input shapes of Whisper-Medium, with a 20 s / 128-position variant
        def __init__(self, _path):
            self.inputs = {
                "whisper_medium_20s_encoder": [Info("encoder/input_layer1", (1, 2000, 80))],
                "whisper_medium_20s_decoder": [
                    Info("decoder/input_layer1", (1, 1000, 1024)),
                    Info("decoder/input_layer2", (1, 128, 1024)),
                ],
            }

        def get_network_group_names(self):
            return list(self.inputs)

        def get_input_vstream_infos(self, group):
            return self.inputs[group]

    monkeypatch.setattr(hailo_whisper, "HEF", FakeHef)

    assert hailo_whisper._read_model_limits("whisper.hef") == (20.0, 64)


def test_model_limits_fall_back_to_the_released_models(tmp_path):
    assert hailo_whisper._read_model_limits(tmp_path / "missing.hef") == (10.0, 32)


# ----------------------------------------------------------------------------- Silero VAD


def test_silero_vad_scores_silence_as_non_speech():
    pytest.importorskip("onnxruntime")
    from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper import vad

    if not vad.VAD_MODEL_PATH.is_file():
        pytest.skip("Silero VAD model not installed (unit tests do not download it)")
    detector = vad.SileroVad(vad.VAD_MODEL_PATH)
    probabilities = detector.feed(np.zeros(SAMPLE_RATE, np.float32))
    assert len(probabilities) == SAMPLE_RATE // vad.FRAME_SAMPLES
    assert probabilities.max() < 0.2


def test_vad_model_is_installed_with_the_whisper_resources(tmp_path, monkeypatch):
    from hailo_apps.installation import download_resources
    from hailo_apps.installation.download_resources import (
        DEFAULT_RESOURCES_CONFIG_PATH,
        ResourceDownloader,
        load_config,
    )
    from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper import vad

    # The downloader looks up the Model Zoo version from the installed HailoRT.
    monkeypatch.setattr(
        download_resources, "get_model_zoo_version_for_arch", lambda arch: ("v5.4.0", arch)
    )

    downloader = ResourceDownloader(
        config=load_config(Path(DEFAULT_RESOURCES_CONFIG_PATH)),
        hailo_arch="hailo10h",
        resource_root=tmp_path,
    )
    downloader.collect_onnx_by_tag("whisper_chat")

    destinations = [task.dest_path for task in downloader._tasks]
    assert destinations == [
        tmp_path / vad.VAD_MODEL_PATH.relative_to(vad.VAD_MODEL_PATH.parents[1])
    ]


def test_group_download_fetches_the_vad_model_only_for_its_apps(tmp_path, monkeypatch):
    from hailo_apps.installation import download_resources
    from hailo_apps.installation.download_resources import (
        DEFAULT_RESOURCES_CONFIG_PATH,
        ResourceDownloader,
        load_config,
    )

    monkeypatch.setattr(
        download_resources, "get_model_zoo_version_for_arch", lambda arch: ("v5.4.0", arch)
    )
    # Collecting a group's models looks them up online; only the shared files matter here.
    monkeypatch.setattr(ResourceDownloader, "collect_models_for_app", lambda *_, **__: None)
    config = load_config(Path(DEFAULT_RESOURCES_CONFIG_PATH))

    def onnx_tasks(group, arch):
        downloader = ResourceDownloader(config=config, hailo_arch=arch, resource_root=tmp_path)
        downloader.collect_group_resources(group)
        return [t.name for t in downloader._tasks if t.resource_type == "onnx"]

    assert onnx_tasks("whisper_chat", "hailo10h") == ["silero_vad.onnx"]
    assert onnx_tasks("detection", "hailo8") == []


def test_missing_vad_model_fails_with_the_install_command(tmp_path, monkeypatch):
    from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper import vad

    monkeypatch.setattr(vad, "VAD_MODEL_PATH", tmp_path / "onnx" / "silero_vad.onnx")
    monkeypatch.setattr(vad, "_download_vad_model", lambda: None)  # e.g. no network

    with pytest.raises(FileNotFoundError, match="hailo-download-resources --group whisper_chat"):
        vad.resolve_vad_model()


def test_explicit_vad_model_path_must_exist(tmp_path):
    from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper import vad

    with pytest.raises(FileNotFoundError):
        vad.resolve_vad_model(tmp_path / "missing.onnx")
