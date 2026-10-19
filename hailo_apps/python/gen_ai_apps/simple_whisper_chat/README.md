# Hailo Simple Whisper Chat Example

A simple example application demonstrating audio transcription using Hailo's Whisper speech-to-text model,
for recordings and for live microphone input, in any of Whisper's languages.

## Features

- Audio file loading and processing (WAV, FLAC or OGG: resampled to 16 kHz, mixed down to mono)
- Speech-to-text transcription of recordings of any length
- Live transcription of a microphone (`--live`), phrase by phrase as the speaker pauses
- Multilingual: pass the Whisper language code with `--language`, or `auto`; the language
  used (or detected) is always printed
- Default audio file: `audio.wav` (in same directory)
- Auto-downloads model on first run

## Prerequisites

- Hailo AI accelerator device (H10 or compatible)
- Python 3.10+
- HailoRT runtime library and Python bindings

## Installation

### Linux
Before running this example, ensure GenAI dependencies are installed:

```bash
# From the repository root directory
pip install -e ".[gen-ai]"
```
### Windows
Use a clean virtual environment before installing the dependencies:
```bash
py -m venv wind_venv
.\wind_venv\Scripts\Activate.ps1
pip install .\hailort-<version>-cp<python>-cp<python>-win_amd64.whl

# From the repository root directory
pip install -e ".[gen-ai]"
```

This will install required packages including:
- NumPy and SciPy
- onnxruntime (runs the Silero voice activity detector; its 2 MB model is installed with the app's
  resources by `hailo-download-resources --group whisper_chat`, or fetched on first use)
- langdetect (for `--language auto`)
- sounddevice (for `--live`)

**For complete installation instructions, see:** [GenAI Applications Installation Guide](../README.md#installation)

## Files

- `simple_whisper_chat.py` - Main example script demonstrating Whisper transcription functionality
- `audio.wav` - Sample audio file for testing

## Usage

Run the example with the default audio file:

```bash
python -m hailo_apps.python.gen_ai_apps.simple_whisper_chat.simple_whisper_chat
```

Or specify a custom audio file:

```bash
python -m hailo_apps.python.gen_ai_apps.simple_whisper_chat.simple_whisper_chat --audio /path/to/your/audio.wav
```

Transcribe another language (Whisper language code), or let the model detect it:

```bash
python -m hailo_apps.python.gen_ai_apps.simple_whisper_chat.simple_whisper_chat --audio announcement_fr.wav --language fr
python -m hailo_apps.python.gen_ai_apps.simple_whisper_chat.simple_whisper_chat --audio announcement.wav --language auto
```

Setting the language explicitly is more reliable than `auto` when it is known.

Transcribe the microphone live (Ctrl+C to stop), or play a recording into the live path at
real-time pace to see how it behaves:

```bash
python -m hailo_apps.python.gen_ai_apps.simple_whisper_chat.simple_whisper_chat --live --language auto
python -m hailo_apps.python.gen_ai_apps.simple_whisper_chat.simple_whisper_chat --live --audio announcement.wav
```

Each live line shows when the phrase started and how long after it ended its text appeared.

### Command Line Options

- `--hef-path PATH` - Specify a custom path to the HEF model file
- `--list-models` - List available models for this application
- `--audio PATH` - Path to audio file (default: audio.wav in same directory)
- `--language CODE` - Whisper language code of the audio (`en`, `fr`, `es`, `de`, `ar`, `zh`, `sv`, ...),
  or `auto` to let the model detect it (default: `en`)
- `--live` - Transcribe the microphone (or, with `--audio`, play the file at real-time pace)
- `--input-device N` - Microphone device index for `--live` (list them with `python3 -m sounddevice`)

### Example Output

The example will:
1. Initialize the Hailo device and load the Whisper model
2. Load the audio file (default: audio.wav in the same directory)
3. Convert the audio to 16 kHz mono and transcribe it
4. Display the complete transcription and the time it took
5. Clean up resources

## Model Requirements

The example uses the `WHISPER_MODEL_NAME_H10` model which is automatically downloaded on first run. No manual download required.

**Note:** Models are downloaded automatically when you run the example for the first time.

## Audio File Format

The example reads WAV (integer PCM of any width, or float), FLAC and OGG files with:
- Sample rate: any (resampled to the 16 kHz Whisper expects)
- Channels: any (averaged to mono)

Other formats (MP3, WebM, ...) can be converted first, for example:

```bash
ffmpeg -i input.webm -ac 1 -ar 16000 output.wav
```

## Troubleshooting

### Model not found error
- Ensure Hailo models are properly installed
- Check model paths in the resource directory
- The model will be downloaded automatically on first run

### Device initialization failed
- Verify Hailo device is connected and recognized
- Check device permissions

### Audio file not found
- Verify the audio file exists at the specified path
- If using default, ensure `audio.wav` is in the same directory as the script
- Check file permissions

### Silero VAD model not found
- Run `hailo-download-resources --group whisper_chat` (it installs the model into `<resources>/onnx/`)
- Without network access, copy `silero_vad.onnx` there by hand

### Error reading the audio file
- Ensure the file is WAV, FLAC or OGG (convert other formats with ffmpeg, see above)
- Check that the file is not corrupted
- Verify the file is readable

### Import errors
- Ensure HailoRT runtime library and Python bindings is properly installed
- Verify Python environment has all required packages (NumPy)

## How it works

The example uses `HailoWhisper` (`gen_ai_utils/whisper/`), the entry point for Whisper on
Hailo-10H in your own code too:

```python
from hailo_apps.python.gen_ai_apps.gen_ai_utils.whisper import HailoWhisper, load_audio

whisper = HailoWhisper(vdevice)                     # default Whisper HEF, or hef_path=...
result = whisper.transcribe(load_audio("speech.wav"), language="auto")
print(result.language, result.text)

live = whisper.live(language="fr")                  # e.g. a microphone
for chunk in chunks:                                # mono float32, 16 kHz, any chunk size
    for phrase in live.feed(chunk):
        print(phrase.text)
for phrase in live.flush():
    print(phrase.text)

whisper.release()                                   # before vdevice.release()
```

Use it for audio of any length and language rather than calling `Speech2Text` directly. The
Hailo-10H Whisper models process audio in 10-second windows, with a fixed budget of generated
tokens per window. Handed a long recording, or dense speech (many languages, fast speakers),
`Speech2Text` can use up a window's budget before a sentence ends and lose the rest of that
window, and its fixed 10-second cuts land inside words. `HailoWhisper` avoids this without any
per-language tuning:

- a voice activity detector (Silero VAD) finds speech and pauses; audio is cut at pauses into
  phrases of at most 10 seconds, shorter for speech that needs more tokens per second;
- audio without speech is never sent to the model, which tends to invent text for it;
- when a returned segment shows that a window ran out of tokens, that phrase is split at its
  quietest point and the two halves are transcribed;
- with `language="auto"`, the language detected for the first phrases of a transmission is
  confirmed from their text and then kept for the rest of it.

This is a simplified example. For voice interaction (wake word, push-to-talk, LLM answers),
see the full [Voice Assistant](../voice_assistant/) application.
