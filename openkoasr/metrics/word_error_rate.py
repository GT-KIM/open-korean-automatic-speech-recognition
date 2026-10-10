# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.

# Code adapted from https://github.com/hyeonsangjeon/computing-Korean-STT-error-rates
# Copyright 2022 Hyeon Sang Jeon, Licensed under the MIT License.
# License is provided for attribution purposes only, Not a Contribution

from openkoasr.metrics.utils import levenshtein
from typing import Any, Dict, List, Tuple, Union

def _measure_wer(
        prediction: str, target: str
) -> Tuple[int, int, int, int]:
    """정답에서 예측으로의 편집 통계를 (일치, 대체, 삭제, 삽입) 순서로 반환합니다."""
    reference_words = target.split()
    prediction_words = prediction.split()
    _, (substitutions, deletions, insertions) = levenshtein(reference_words, prediction_words)
    hits = len(reference_words) - substitutions - deletions

    return hits, substitutions, deletions, insertions


def word_error_rate(sentence1: str, sentence2: str) -> dict:
    hits, substitutions, deletions, insertions = _measure_wer(sentence1, sentence2)
    total_words = len(sentence2.split())

    if total_words == 0:
        wer = float('inf') if len(sentence1.split()) > 0 else 0.0
    else:
        wer = (substitutions + deletions + insertions) / total_words

    return {
        "wer": wer,
        "hits": hits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }
