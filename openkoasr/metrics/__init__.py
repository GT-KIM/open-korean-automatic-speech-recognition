# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.

from .word_error_rate import word_error_rate
from .character_error_rate import character_error_rate
from .sentence_error_rate import sentence_error_rate
from .performance import (
    real_time_factor,
    latency,
    get_flops,
    get_num_parameters,
)
from .evaluator import Evaluator


def morpheme_error_rate(sentence1: str, sentence2: str) -> dict:
    from .morpheme_error_rate import morpheme_error_rate as _metric
    return _metric(sentence1, sentence2)


def jamo_error_rate(sentence1: str, sentence2: str) -> dict:
    from .jamo_error_rate import jamo_error_rate as _metric
    return _metric(sentence1, sentence2)

__all__ = [
    "word_error_rate",
    "character_error_rate",
    "morpheme_error_rate",
    "jamo_error_rate",
    "sentence_error_rate",
    "real_time_factor",
    "latency",
    "get_flops",
    "get_num_parameters",
    "Evaluator",
]
