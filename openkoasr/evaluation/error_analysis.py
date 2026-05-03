from difflib import SequenceMatcher


def sequence_error_ops(reference_items, prediction_items):
    matcher = SequenceMatcher(a=reference_items, b=prediction_items)
    ops = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        ops.append(
            {
                "operation": tag,
                "reference": reference_items[i1:i2],
                "prediction": prediction_items[j1:j2],
            }
        )
    return ops


def sample_error_analysis(sample):
    return {
        "index": sample.index,
        "sample_id": sample.sample_id,
        "metrics": {
            key: sample.metrics[key]
            for key in ("wer", "cer", "mer", "jer", "ser")
            if key in sample.metrics
        },
        "word_ops": sequence_error_ops(
            sample.normalized_reference.split(),
            sample.normalized_prediction.split(),
        ),
        "char_ops": sequence_error_ops(
            list(sample.normalized_reference),
            list(sample.normalized_prediction),
        ),
    }
