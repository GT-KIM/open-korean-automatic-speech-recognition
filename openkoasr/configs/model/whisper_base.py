whisper_config = {
    "name": "whisper_base",
    "repo_name": "openai/whisper-base", # 동작 확인된 모델 리스트는 문서 참조 바랍니다. SungBeom/whisper-small-ko
    "dtype": "float16",
    "device": "cuda:0",
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
