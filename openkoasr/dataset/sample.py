DEFAULT_SAMPLE_RATE = 16000


def identity_collate(batch):
    return batch


def iter_samples(batch):
    if isinstance(batch, list):
        for sample in batch:
            yield sample
        return

    if isinstance(batch, tuple) and len(batch) >= 2:
        audio_batch, text_batch = batch[0], batch[1]
        batch_size = _batch_size(audio_batch, text_batch)
        if batch_size is not None and batch_size > 1:
            for index in range(batch_size):
                yield (audio_batch[index], text_batch[index])
            return

    yield batch


def _batch_size(audio_batch, text_batch):
    if isinstance(text_batch, (list, tuple)):
        return len(text_batch)
    shape = getattr(audio_batch, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[0])
    return None


def _first_sequence_value(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def _first_tensor_like(value):
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 2:
        return value[0]
    return value


def _first_batched_sequence(value):
    if not isinstance(value, (list, tuple)) or not value:
        return value
    first = value[0]
    if isinstance(first, (int, float, complex, str, bytes)):
        return value
    return first


def _to_python_scalar(value):
    value = _first_sequence_value(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def get_sample_audio(sample):
    if isinstance(sample, dict):
        audio = sample.get("audio")
    else:
        audio = sample[0]
    return _first_tensor_like(_first_batched_sequence(audio))


def get_sample_text(sample):
    if isinstance(sample, dict):
        text = sample.get("text", "")
    else:
        text = sample[1]
    text = _to_python_scalar(text)
    return "" if text is None else str(text)


def get_sample_rate(sample, default=DEFAULT_SAMPLE_RATE):
    if isinstance(sample, dict):
        sample_rate = sample.get("sample_rate", default)
    else:
        sample_rate = default
    sample_rate = _to_python_scalar(sample_rate)
    return int(sample_rate or default)


def get_sample_metadata(sample):
    if not isinstance(sample, dict):
        return {}
    metadata = sample.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        return {"metadata": metadata}
    return {key: _to_python_scalar(value) for key, value in metadata.items()}


def get_sample_id(sample, index):
    metadata = get_sample_metadata(sample)
    for key in ("id", "sample_id", "utt_id", "audio_path"):
        if key in metadata and metadata[key] is not None:
            return str(metadata[key])
    return str(index)


def get_audio_duration_seconds(audio, sample_rate):
    shape = getattr(audio, "shape", None)
    if shape is not None and len(shape) > 0:
        length = int(shape[-1])
    elif hasattr(audio, "__len__"):
        length = len(audio)
    else:
        length = 0
    if sample_rate <= 0:
        return 0.0
    return float(length) / float(sample_rate)
