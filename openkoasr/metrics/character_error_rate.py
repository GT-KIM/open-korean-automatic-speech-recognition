# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.

from openkoasr.metrics.utils import levenshtein
from typing import Tuple

def _measure_cer(
        prediction: str, target: str
) -> Tuple[int, int, int, int]:
    """
    소스 문자열을 대상 문자열로 변환하는 데 필요한 편집 작업(삭제, 삽입, 바꾸기)의 수를 확인합니다.
    """
    pred_chars = list(prediction)
    target_chars = list(target)

    _, (s, i, d) = levenshtein(pred_chars, target_chars)
    
    substitutions = s
    deletions = d
    insertions = i
    hits = len(target_chars) - (substitutions + deletions)

    return hits, substitutions, deletions, insertions


def character_error_rate(sentence1: str, sentence2: str) -> dict:
    """
    음절 오류율 (Character Error Rate, CER)을 계산합니다.
    한국어 ASR에서는 보통 음절(Syllable) 단위 오류율을 의미합니다.
    """
    hits, substitutions, deletions, insertions = _measure_cer(sentence1, sentence2)
    total_chars = len(sentence2)

    if total_chars == 0:
        cer = float('inf') if len(sentence1) > 0 else 0.0
    else:
        cer = (substitutions + deletions + insertions) / total_chars

    return {
        "cer": cer,
        "hits": hits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }
