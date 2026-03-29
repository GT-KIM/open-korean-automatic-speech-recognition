# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.

def sentence_error_rate(sentence1: str, sentence2: str) -> dict:
    """
    문장 오류율 (Sentence Error Rate, SER)을 계산합니다.
    문장 전체가 정확히 일치해야 정답으로 인정됩니다.
    """
    is_error = 1 if sentence1 != sentence2 else 0
    
    return {
        "ser": float(is_error),
        "error_sentences": is_error,
        "total_sentences": 1,
    }
