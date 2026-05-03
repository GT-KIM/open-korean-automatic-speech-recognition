qwen3_asr_config = {
    "name": "qwen3_asr_0_6b",
    "repo_name": "Qwen/Qwen3-ASR-0.6B",
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
