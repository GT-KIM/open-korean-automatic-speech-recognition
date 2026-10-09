from openkoasr.protocol import evaluation_protocol_id


def public_metadata(normalization="kspon"):
    return {
        "normalization_preset": normalization,
        "evaluation_protocol": evaluation_protocol_id(normalization, {"metric": "cer", "threshold": 1}),
        "metadata_status": "recorded",
        "reproducibility": {
            "model_revision": None,
            "code": {"commit": None, "dirty": None, "source_sha256": "a" * 64},
            "model_config": {}, "environment": {},
            "execution": {"batch_size": 1, "num_workers": 0, "warmup_samples": 0, "limit": None},
        },
    }
