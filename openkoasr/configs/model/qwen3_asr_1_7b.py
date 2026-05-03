qwen3_asr_config = {
    "name": "qwen3_asr_1_7b",
    "repo_name": "Qwen/Qwen3-ASR-1.7B",
    "dtype": "bfloat16",
    "device": "cuda:0",
    "language": "Korean",
    "max_inference_batch_size": 32,
    "max_new_tokens": 256,
    "evaluation": {
        "metrics": [
            "wer",
            "cer",
            "mer",
            "jer",
            "ser",
            "rtf",
            "latency"
        ]
    }
}
