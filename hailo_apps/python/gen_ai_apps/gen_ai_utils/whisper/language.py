"""Whisper languages and the per-transmission language lock used for automatic detection."""

from dataclasses import dataclass, field

from hailo_apps.python.core.common.hailo_logger import get_logger

logger = get_logger(__name__)

# The languages of Whisper's multilingual models (code -> English name), as in openai/whisper.
LANGUAGES = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian",
    "ko": "Korean", "fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish",
    "pl": "Polish", "ca": "Catalan", "nl": "Dutch", "ar": "Arabic", "sv": "Swedish",
    "it": "Italian", "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay", "cs": "Czech",
    "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian",
    "th": "Thai", "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian",
    "la": "Latin", "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali", "sr": "Serbian",
    "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada", "et": "Estonian",
    "mk": "Macedonian", "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian",
    "ne": "Nepali", "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian",
    "sw": "Swahili", "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala",
    "km": "Khmer", "sn": "Shona", "yo": "Yoruba", "so": "Somali", "af": "Afrikaans",
    "oc": "Occitan", "ka": "Georgian", "be": "Belarusian", "tg": "Tajik", "sd": "Sindhi",
    "gu": "Gujarati", "am": "Amharic", "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek",
    "fo": "Faroese", "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen",
    "nn": "Norwegian Nynorsk", "mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish",
    "my": "Burmese", "bo": "Tibetan", "tl": "Tagalog", "mg": "Malagasy", "as": "Assamese",
    "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa", "ba": "Bashkir",
    "jw": "Javanese", "su": "Sundanese",
}  # fmt: skip

# Languages written without spaces between words.
UNSPACED_LANGUAGES = frozenset({"zh", "ja"})

# langdetect codes that differ from Whisper's.
_LANGDETECT_TO_WHISPER = {"zh-cn": "zh", "zh-tw": "zh", "jv": "jw"}


def language_name(code: str | None) -> str:
    """
    Readable name of a Whisper language code.

    Args:
        code: Whisper language code, or None for automatic detection.

    Returns:
        E.g. "French (fr)", or "automatic detection".
    """
    if code is None:
        return "automatic detection"
    return f"{LANGUAGES.get(code, code)} ({code})"


def _count(text: str, *ranges: tuple[str, str]) -> int:
    return sum(1 for ch in text if any(lo <= ch <= hi for lo, hi in ranges))


_HANGUL = (("가", "힯"), ("ᄀ", "ᇿ"))
_KANA = (("぀", "ヿ"),)
_HAN = (("一", "鿿"), ("㐀", "䶿"))


def script_language(text: str) -> str | None:
    """
    Chinese, Japanese or Korean from the script the text is written in.

    Statistical identifiers confuse these three on short text; the script decides it.

    Args:
        text: Transcribed text.

    Returns:
        "zh", "ja" or "ko", or None when the text is not mostly in these scripts.
    """
    hangul, kana, han = _count(text, *_HANGUL), _count(text, *_KANA), _count(text, *_HAN)
    letters = sum(1 for ch in text if ch.isalpha())
    if not letters or hangul + kana + han < letters / 2:
        return None
    if hangul > (kana + han) / 2:
        return "ko"
    if kana:
        return "ja"
    return "zh"


_langdetect_warned = False


def identify_language(text: str) -> str | None:
    """
    Whisper language code of a transcribed text.

    Chinese, Japanese and Korean are told apart by script; other languages use the optional
    ``langdetect`` package.

    Args:
        text: Transcribed text.

    Returns:
        A code from ``LANGUAGES``, or None when it cannot be identified.
    """
    global _langdetect_warned
    by_script = script_language(text)
    if by_script is not None:
        return by_script
    try:
        from langdetect import DetectorFactory, detect
        from langdetect.lang_detect_exception import LangDetectException
    except ImportError:
        if not _langdetect_warned:
            logger.warning(
                "langdetect is not installed, so automatic language detection cannot lock a "
                "language and each phrase is detected on its own. Install it with "
                "'pip install langdetect'."
            )
            _langdetect_warned = True
        return None
    DetectorFactory.seed = 0  # langdetect is randomised; make it repeatable
    try:
        code = detect(text)
    except LangDetectException:  # no usable features, e.g. only digits or punctuation
        return None
    code = _LANGDETECT_TO_WHISPER.get(code, code)
    return code if code in LANGUAGES else None


@dataclass
class LanguageLock:
    """
    Settles the language of a transmission once its transcribed phrases agree on it.

    Until then ``language`` is None and phrases use the runtime's automatic detection, which
    is unreliable on a few seconds of audio. Each phrase's text is identified separately; the
    language is locked when phrases totalling ``min_seconds`` agree on it, with at least
    ``share`` of the identified audio, and either two phrases or one of ``single_phrase_s``.

    Attributes:
        min_seconds: Audio that must agree on the language before locking.
        single_phrase_s: A single phrase at least this long may lock on its own.
        share: Fraction of the identified audio that must agree.
        min_chars: Shortest text that votes (a third of it for Chinese, Japanese and Korean,
            whose characters carry more information).
        language: The locked language code, or None while undecided.
    """

    min_seconds: float = 6.0
    single_phrase_s: float = 8.0
    share: float = 0.7
    min_chars: int = 12
    language: str | None = None
    _votes: dict[str, float] = field(default_factory=dict)
    _phrases: dict[str, int] = field(default_factory=dict)

    def observe(self, text: str, seconds: float) -> None:
        """
        Add one transcribed phrase; lock the language when enough phrases agree.

        Args:
            text: Text of the phrase.
            seconds: Duration of the phrase's audio.
        """
        text = text.strip()
        if self.language is not None or not text:
            return
        cjk_share = (_count(text, *_HANGUL, *_KANA, *_HAN) / len(text)) if text else 0.0
        if len(text) < (self.min_chars // 3 if cjk_share > 1 / 3 else self.min_chars):
            return
        code = identify_language(text)
        if code is None:
            return
        self._votes[code] = self._votes.get(code, 0.0) + seconds
        self._phrases[code] = self._phrases.get(code, 0) + 1
        best = max(self._votes, key=self._votes.get)
        agreed, total = self._votes[best], sum(self._votes.values())
        enough = self._phrases[best] >= 2 or agreed >= self.single_phrase_s
        if agreed >= self.min_seconds and enough and agreed >= self.share * total:
            self.language = best
            logger.info(f"Detected language: {language_name(best)}")

    def reset(self) -> None:
        """Forget the language, e.g. when a new transmission starts."""
        self.language = None
        self._votes = {}
        self._phrases = {}
