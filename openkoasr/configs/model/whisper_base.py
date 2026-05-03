whisper_config = {
    "name": "whisper_base",
    "repo_name": "openai/whisper-base", # 동작 확인된 모델 리스트는 문서 참조 바랍니다. SungBeom/whisper-small-ko
    "dtype": "float16",
    "device": "cuda:0",
    "language": "ko",
    "task": "transcribe",
    "max_new_tokens": 128,
    "num_beams": 1,
    "temperature": 0.0,
    "condition_on_prev_tokens": False,
    "evaluation": {
        "metrics": [
            "wer",
            "cer",
            "mer",
            "jer",
            "ser",
            "rtf",
            "latency",
            "flops",
            "params"
        ]
    }
}
