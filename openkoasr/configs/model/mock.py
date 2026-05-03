mock_asr_config = {
    "name": "mock",
    "family": "mock",
    "repo_name": "local/mock-asr",
    "dtype": "float32",
    "device": "cpu",
    "mode": "metadata_prediction",
    "evaluation": {
        "metrics": [
            "wer",
            "cer",
            "ser",
            "rtf",
            "latency",
        ]
    },
}
