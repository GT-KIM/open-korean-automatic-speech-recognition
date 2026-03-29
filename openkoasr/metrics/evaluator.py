# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.

import torch
from typing import List, Dict, Any
from openkoasr.metrics.word_error_rate import word_error_rate
from openkoasr.metrics.character_error_rate import character_error_rate
from openkoasr.metrics.morpheme_error_rate import morpheme_error_rate
from openkoasr.metrics.jamo_error_rate import jamo_error_rate
from openkoasr.metrics.sentence_error_rate import sentence_error_rate
from openkoasr.metrics.performance import (
    real_time_factor,
    latency,
    get_flops,
    get_num_parameters,
)

class Evaluator:
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.metric_functions = {
            "wer": word_error_rate,
            "cer": character_error_rate,
            "mer": morpheme_error_rate,
            "jer": jamo_error_rate,
            "ser": sentence_error_rate,
            "rtf": real_time_factor,
            "latency": latency,
            "flops": get_flops,
            "params": get_num_parameters,
        }

    def evaluate(self, **kwargs: Any) -> Dict[str, Any]:
        results = {}
        for metric in self.metrics:
            if metric in self.metric_functions:
                if metric == "flops" or metric == "params":
                    continue  # Skip model-level metrics here
                func = self.metric_functions[metric]
                # 필요한 인자만 필터링하여 함수에 전달
                params = {
                    p: kwargs[p]
                    for p in func.__code__.co_varnames[:func.__code__.co_argcount]
                    if p in kwargs
                }
                results.update(func(**params))
        return results
