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
    """
    소스 문자열을 대상 문자열로 변환하는 데 필요한 편집 작업(삭제, 삽입, 바꾸기)의 수를 확인합니다.
    hints 횟수는 소스 문자열의 전체 길이에서 삭제 및 대체 횟수를 빼서 제공할 수 있습니다.

    :param transcription: 대상 단어
    :param reference: 소스 단어
    :return: a tuple of #hits, #substitutions, #deletions, #insertions
    """

    ref, hyp = [], []

    ref.append(prediction)
    hyp.append(target)

    wer_s, wer_i, wer_d, wer_n = 0, 0, 0, 0
    sen_err = 0

    for n in range(len(ref)):
        # update WER statistics
        _, (s, i, d) = levenshtein(hyp[n].split(), ref[n].split())
        wer_s += s
        wer_i += i
        wer_d += d
        wer_n += len(ref[n].split())
        # update SER statistics
        if s + i + d > 0:
            sen_err += 1

    substitutions = wer_s
    deletions = wer_d
    insertions = wer_i
    hits = len(prediction.split()) - (substitutions + deletions)  # correct words between refs and trans

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
