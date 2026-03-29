# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.

import jamo
from openkoasr.metrics.utils import levenshtein
from typing import Tuple, List

def _get_jamos(text: str) -> List[str]:
    """주어진 텍스트를 자소 단위로 분리하여 리스트로 반환합니다."""
    return list(jamo.hangul_to_jamo(text))

def _measure_jer(
        prediction: str, target: str
) -> Tuple[int, int, int, int]:
    """
    자소 단위로 소스 문자열을 대상 문자열로 변환하는 데 필요한 편집 작업을 계산합니다.
    """
    pred_jamos = _get_jamos(prediction)
    target_jamos = _get_jamos(target)

    _, (s, i, d) = levenshtein(pred_jamos, target_jamos)
    
    substitutions = s
    deletions = d
    insertions = i
    hits = len(target_jamos) - (substitutions + deletions)

    return hits, substitutions, deletions, insertions


def jamo_error_rate(sentence1: str, sentence2: str) -> dict:
    """
    자소 오류율 (Jamo Error Rate, JER)을 계산합니다.
    """
    hits, substitutions, deletions, insertions = _measure_jer(sentence1, sentence2)
    total_jamos = len(_get_jamos(sentence2))

    if total_jamos == 0:
        jer = float('inf') if len(_get_jamos(sentence1)) > 0 else 0.0
    else:
        jer = (substitutions + deletions + insertions) / total_jamos

    return {
        "jer": jer,
        "hits": hits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }
