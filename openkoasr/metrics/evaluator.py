# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.

from typing import List, Dict, Any
from openkoasr.metrics.word_error_rate import word_error_rate
from openkoasr.metrics.character_error_rate import character_error_rate
from openkoasr.metrics.sentence_error_rate import sentence_error_rate
from openkoasr.metrics.performance import (
    real_time_factor,
    latency,
    get_flops,
    get_num_parameters,
)


def morpheme_error_rate(sentence1: str, sentence2: str, kiwi=None) -> dict:
    from openkoasr.metrics.morpheme_error_rate import morpheme_error_rate as _metric
    return _metric(sentence1, sentence2, kiwi=kiwi)


def jamo_error_rate(sentence1: str, sentence2: str) -> dict:
    from openkoasr.metrics.jamo_error_rate import jamo_error_rate as _metric
    return _metric(sentence1, sentence2)

class Evaluator:
    def __init__(self, metrics: List[str]):
        self.metrics = metrics
        self.kiwi = None
        if "mer" in metrics:
            try:
                from kiwipiepy import Kiwi
                self.kiwi = Kiwi()
            except Exception:
                self.kiwi = None
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
                if metric == "mer" and self.kiwi is not None:
                    params["kiwi"] = self.kiwi
                metric_results = func(**params)
                for key, value in metric_results.items():
                    if key == metric:
                        results[key] = value
                    else:
                        results[f"{metric}_{key}"] = value
        return results
