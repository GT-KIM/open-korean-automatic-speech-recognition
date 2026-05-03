import re
import unicodedata

try:
    from openkoasr.dataset.KsponSpeech.utils import normalize_text as normalize_kspon_text
except Exception:  # pragma: no cover - optional when Kspon deps are unavailable.
    normalize_kspon_text = None


PUNCTUATION_RE = re.compile(r"[^\w\s가-힣]", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")


def _compact_whitespace(text):
    return WHITESPACE_RE.sub(" ", text).strip()


def _strict(text):
    return _compact_whitespace(unicodedata.normalize("NFC", text or ""))


def _punctuation_agnostic(text):
    normalized = _strict(text).lower()
    normalized = PUNCTUATION_RE.sub(" ", normalized)
    return _compact_whitespace(normalized)


def _kspon(text):
    if normalize_kspon_text is None:
        raise RuntimeError("Kspon normalization is unavailable.")
    return _compact_whitespace(normalize_kspon_text(text or ""))


NORMALIZATION_PRESETS = {
    "raw": lambda text: text or "",
    "strict": _strict,
    "punctuation_agnostic": _punctuation_agnostic,
    "kspon": _kspon,
}


def available_presets():
    return sorted(NORMALIZATION_PRESETS.keys())


def normalize_text(text, preset="punctuation_agnostic"):
    if preset not in NORMALIZATION_PRESETS:
        raise ValueError(
            f"Unknown normalization preset '{preset}'. "
            f"Available presets: {', '.join(available_presets())}"
        )
    return NORMALIZATION_PRESETS[preset](text)
