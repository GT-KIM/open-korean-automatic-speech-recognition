"""Versioned text-normalization and outlier rules; no model dependencies."""
import math


STANDARD_PROTOCOL = "v1/kspon/cer>1.0"
NORMALIZATION_PRESETS = ("raw", "strict", "punctuation_agnostic", "kspon")
OUTLIER_METRICS = ("wer", "cer", "mer", "jer", "ser", "rtfx", "latency")


def evaluation_protocol_id(normalization, policy):
    if normalization not in NORMALIZATION_PRESETS:
        raise ValueError("normalization_preset must name a supported preset")
    if not isinstance(policy, dict) or policy.get("metric") not in OUTLIER_METRICS:
        raise ValueError("outlier_policy.metric must name a supported metric")
    threshold = policy.get("threshold")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("outlier_policy.threshold must be finite and nonnegative")
    threshold = float(threshold) if threshold else 0.0
    return f"v1/{normalization}/{policy['metric']}>{threshold}"
