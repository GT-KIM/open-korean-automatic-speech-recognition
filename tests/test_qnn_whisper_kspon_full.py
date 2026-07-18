import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_qnn_whisper_kspon_full.py"
)
SPEC = importlib.util.spec_from_file_location("run_qnn_whisper_kspon_full", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sample(index, *, wer, cer, exact, words=10, chars=20, qnn_ms=1000):
    return {
        "index": index,
        "audio_duration": 2.0,
        "exact_match": exact,
        "is_outlier": cer > 1.0,
        "wer": wer,
        "word_errors": round(wer * words),
        "reference_words": words,
        "cer": cer,
        "char_errors": round(cer * chars),
        "reference_chars": chars,
        "encoder_ms": qnn_ms * 0.75,
        "decoder_ms": qnn_ms * 0.25,
    }


def test_aggregate_scores_matches_leaderboard_outlier_policy():
    aggregate = MODULE.aggregate_scores(
        [
            sample(1, wer=0.1, cer=0.2, exact=False),
            sample(2, wer=0.3, cer=0.4, exact=True),
            sample(3, wer=2.0, cer=1.1, exact=False),
        ]
    )

    assert aggregate["total_samples"] == 3
    assert aggregate["valid_samples"] == 2
    assert aggregate["outlier_count"] == 1
    assert aggregate["macro_average"]["wer"] == pytest.approx(0.2)
    assert aggregate["micro_average"]["wer"] == pytest.approx(0.2)
    assert aggregate["macro_average"]["cer"] == pytest.approx(0.3)
    assert aggregate["micro_average"]["cer"] == pytest.approx(0.3)
    assert aggregate["macro_average"]["ser"] == pytest.approx(0.5)
    assert aggregate["runtime_all_samples"]["qnn_rtfx"] == pytest.approx(2.0)
