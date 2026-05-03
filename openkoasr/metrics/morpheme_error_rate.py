# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.

from kiwipiepy import Kiwi
from openkoasr.metrics.utils import levenshtein
from typing import Tuple, List

def _get_morphemes(text: str, kiwi: Kiwi) -> List[str]:
    """주어진 텍스트를 형태소 단위로 분석하여 리스트로 반환합니다."""
    return [f"{m.form}/{m.tag}" for m in kiwi.analyze(text)[0][0]]

def _measure_mer(
        prediction: str, target: str, kiwi: Kiwi
) -> Tuple[int, int, int, int]:
    """
    형태소 단위로 소스 문자열을 대상 문자열로 변환하는 데 필요한 편집 작업을 계산합니다.
    """
    pred_morphemes = _get_morphemes(prediction, kiwi)
    target_morphemes = _get_morphemes(target, kiwi)

    _, (s, i, d) = levenshtein(pred_morphemes, target_morphemes)
    
    substitutions = s
    deletions = d
    insertions = i
    hits = len(target_morphemes) - (substitutions + deletions)

    return hits, substitutions, deletions, insertions


def morpheme_error_rate(sentence1: str, sentence2: str, kiwi: Kiwi = None) -> dict:
    """
    형태소 오류율 (Morpheme Error Rate, MER)을 계산합니다.
    """
    if kiwi is None:
        kiwi = Kiwi()
    hits, substitutions, deletions, insertions = _measure_mer(sentence1, sentence2, kiwi)
    total_morphemes = len(_get_morphemes(sentence2, kiwi))

    if total_morphemes == 0:
        mer = float('inf') if len(_get_morphemes(sentence1, kiwi)) > 0 else 0.0
    else:
        mer = (substitutions + deletions + insertions) / total_morphemes

    return {
        "mer": mer,
        "hits": hits,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }
