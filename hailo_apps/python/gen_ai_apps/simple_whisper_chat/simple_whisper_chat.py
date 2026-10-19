import argparse
import queue
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from hailo_platform import VDevice

repo_root = None
for p in Path(__file__).resolve().parents:
    if (p / "hailo_apps" / "config" / "config_manager.py").exists():
        repo_root = p
        break
if repo_root is not None:
    sys.path.insert(0, str(repo_root))

from hailo_apps.python.core.common.core import handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.defines import (
    HAILO10H_ARCH,
    SHARED_VDEVICE_GROUP_ID,
    WHISPER_CHAT_APP,
)
from hailo_apps.python.core.common.hailo_logger import get_logger
from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper import (
    AUTO_LANGUAGE,
    SAMPLE_RATE,
    HailoWhisper,
    LiveTranscription,
    StreamResampler,
    language_name,
    load_audio,
)

# Initialize logger
logger = get_logger(__name__)

CHUNK_S = 0.1  # live audio is handled in chunks of this length


def file_chunks(audio: np.ndarray) -> Iterator[np.ndarray]:
    """
    Yield a recording in chunks at real-time pace, as a microphone would deliver it.

    Args:
        audio: Mono float32 audio at 16 kHz.

    Yields:
        Consecutive chunks of ``audio``.
    """
    step = int(CHUNK_S * SAMPLE_RATE)
    start = time.perf_counter()
    for i in range(0, len(audio), step):
        delay = start + i / SAMPLE_RATE - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        yield audio[i : i + step]


def microphone_chunks(device: int | None) -> Iterator[np.ndarray]:
    """
    Yield microphone audio as mono float32 chunks at 16 kHz until interrupted.

    The microphone is opened at 16 kHz when it supports it; otherwise at its native rate,
    and the audio is resampled as it arrives.

    Args:
        device: Input device index (see ``python3 -m sounddevice``), or None for the default.

    Yields:
        Consecutive chunks of microphone audio.
    """
    import sounddevice as sd  # only needed for live microphone input

    chunks: queue.Queue = queue.Queue()

    def on_audio(indata, frames, time_info, status):
        if status:
            logger.debug(f"Microphone status: {status}")
        chunks.put(indata[:, 0].copy())

    def open_stream(rate: int):
        return sd.InputStream(
            samplerate=rate,
            blocksize=int(CHUNK_S * rate),
            device=device,
            channels=1,
            dtype="float32",
            callback=on_audio,
        )

    try:
        stream, rate = open_stream(SAMPLE_RATE), SAMPLE_RATE
    except Exception as e:
        rate = int(sd.query_devices(device, "input")["default_samplerate"])
        logger.info(f"Microphone does not support {SAMPLE_RATE} Hz ({e}); using {rate} Hz")
        stream = open_stream(rate)
    resampler = StreamResampler(rate)
    with stream:
        while True:
            yield resampler.feed(chunks.get())


def run_live(live: LiveTranscription, chunks: Iterator[np.ndarray]) -> None:
    """
    Transcribe a live stream, printing each phrase as soon as it is transcribed.

    Args:
        live: The live transcription session.
        chunks: Audio chunks (mono float32, 16 kHz) in real time.
    """
    start = time.perf_counter()
    shown_language = live.language

    def show(phrases):
        nonlocal shown_language
        for phrase in phrases:
            if phrase.new_transmission:
                print()
            delay = max(time.perf_counter() - start - phrase.end_s, 0.0)
            print(f"[{phrase.start_s:7.1f}s +{delay:.1f}s] {phrase.text}", flush=True)
        if live.language != shown_language and live.language is not None:
            print(f"  (detected language: {language_name(live.language)})", flush=True)
        shown_language = live.language

    try:
        for chunk in chunks:
            show(live.feed(chunk))
    except KeyboardInterrupt:
        print("\nStopping...")
    show(live.flush())


def describe_language(choice: str, default: str) -> str:
    """The language line shown before transcribing."""
    if choice == AUTO_LANGUAGE:
        return "Language: automatic detection (the detected language is shown as it settles)"
    note = ""
    if choice == default:
        note = " (the default; use --language to change it, or --language auto)"
    return f"Language: {language_name(choice)}{note}"


def main():
    """Main function for Whisper Speech-to-Text Example."""
    default_language = "en"
    # Parse arguments
    parser = argparse.ArgumentParser(description="Whisper Speech-to-Text Example")
    parser.add_argument("--hef-path", type=str, default=None, help="Path to HEF model file")
    parser.add_argument("--list-models", action="store_true", help="List available models")
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file (default: audio.wav in same directory)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=default_language,
        help="Whisper language code of the audio, e.g. en, fr, es, de, ar, zh, sv "
        f"(default: {default_language}). Use '{AUTO_LANGUAGE}' to detect it.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Transcribe live microphone input (Ctrl+C to stop). With --audio, the file is "
        "played into the live transcriber at real-time pace instead.",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="Microphone device index for --live (default: system default input)",
    )

    # Handle --list-models flag before full initialization
    handle_list_models_flag(parser, WHISPER_CHAT_APP)

    args = parser.parse_args()

    # Resolve HEF path with auto-download (Whisper is Hailo-10H only)
    hef_path = resolve_hef_path(args.hef_path, app_name=WHISPER_CHAT_APP, arch=HAILO10H_ARCH)
    if hef_path is None:
        logger.error("Failed to resolve HEF path for Whisper model.")
        sys.exit(1)

    logger.info(f"Using HEF: {hef_path}")
    print(f"✓ Model file found: {hef_path}")

    vdevice = None
    whisper = None

    try:
        print("\n[1/4] Initializing Hailo device...")
        params = VDevice.create_params()
        params.group_id = SHARED_VDEVICE_GROUP_ID
        vdevice = VDevice(params)
        print("✓ Hailo device initialized")

        print("[2/4] Loading Whisper model...")
        whisper = HailoWhisper(vdevice, hef_path)
        live = whisper.live(args.language)  # validates the language before any audio is read
        print("✓ Model loaded successfully")
        print(describe_language(args.language, default_language))

        if args.live:
            if args.audio:
                print(f"\nStreaming {args.audio} at real-time pace")
                chunks = file_chunks(load_audio(args.audio))
            else:
                print("\nListening. Press Ctrl+C to stop.")
                chunks = microphone_chunks(args.input_device)
            print("Each line: [time in the stream, delay after the phrase ended] text")
            run_live(live, chunks)
            print("\n✓ Example completed successfully")
            return

        audio_path = Path(args.audio) if args.audio else Path(__file__).parent / "audio.wav"
        print(f"[3/4] Loading audio file: {audio_path}")
        audio_data = load_audio(audio_path)
        duration_s = len(audio_data) / SAMPLE_RATE
        print(f"✓ Audio loaded ({duration_s:.2f}s, converted to {SAMPLE_RATE} Hz mono)")

        print("[4/4] Transcribing audio with Whisper...")
        start = time.perf_counter()
        result = whisper.transcribe(audio_data, language=args.language)
        elapsed = time.perf_counter() - start

        if result.text:
            if args.language == AUTO_LANGUAGE:
                settled = "not settled (detected per phrase)"
                if result.language:
                    settled = language_name(result.language)
                print(f"Detected language: {settled}")
            print(f"\n✓ Transcription completed in {elapsed:.1f}s:")
            print("-" * 60)
            print(result.text.strip())
            print("-" * 60)
            print("\n✓ Example completed successfully")
        else:
            print("\n⚠ No transcription generated")

    except FileNotFoundError as e:  # the audio file, or the Silero VAD model
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Clean up resources
        if whisper:
            try:
                whisper.release()
            except Exception as e:
                logger.warning(f"Error releasing the Whisper model: {e}")

        if vdevice:
            try:
                vdevice.release()
            except Exception as e:
                logger.warning(f"Error releasing VDevice: {e}")


if __name__ == "__main__":
    main()
