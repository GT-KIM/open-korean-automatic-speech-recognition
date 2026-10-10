"""Fixed-input warmed ASR timing, separate from full-corpus quality evaluation."""
import hashlib
import io
import json
import math
import time
from collections import Counter
from pathlib import Path

GROUPS = ("clean", "other", "telephone")


def termination_policy(protocol):
    policy = protocol.get("termination", {}).get("policy", "require_eos")
    if policy not in ("require_eos", "retain_budget_terminated"):
        raise ValueError("Unknown generation termination policy")
    return policy


def summarize_termination(requests, warmups, policy):
    """Count all audited outputs, including capped ones, outside inference timing."""
    if policy not in ("require_eos", "retain_budget_terminated"):
        raise ValueError("Unknown generation termination policy")

    def counts(rows, measured):
        total = capped = 0
        for row in rows:
            audit = row["generation_audit"]
            expected = row["batch_size"] if measured else 1
            if audit["sequences"] != expected:
                raise ValueError("Generation audit count differs from output count")
            budget = audit["sequences_at_token_limit"]
            if not 0 <= budget <= expected or budget != audit["sequences_without_eos"]:
                raise ValueError("Unclassified generation termination")
            total += expected
            capped += budget
        return {"sequences": total, "eos": total - capped, "token_limit": capped,
                "token_limit_rate": capped / total if total else 0.0}

    measured = counts(requests, True)
    warmup = counts(warmups, False)
    repeats = []
    for repeat in sorted({row["repeat"] for row in requests}):
        rows = [row for row in requests if row["repeat"] == repeat]
        repeats.append({"repeat": repeat, "overall": counts(rows, True),
                        "groups": {g: counts([row for row in rows if row["group"] == g], True)
                                   for g in GROUPS}})
    all_eos = measured["token_limit"] + warmup["token_limit"] == 0
    return {"policy": policy, "all_generations_reached_eos": all_eos,
            "contract_passed": all_eos or policy == "retain_budget_terminated",
            "measured": measured, "warmup": warmup, "repeats": repeats}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def load_inputs(root, manifest_name, expected_hash, expected_count):
    import numpy as np
    from openkoasr.normalization import normalize_text

    root = Path(root).resolve()
    raw = (root / manifest_name).read_bytes()
    if digest(raw) != expected_hash:
        raise ValueError("Sealed input manifest hash mismatch")
    records = [json.loads(line) for line in raw.splitlines() if line]
    if len(records) != expected_count:
        raise ValueError("Sealed input count mismatch")
    samples = []
    for row in records:
        path = (root / row["audio_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Audio path escapes sealed input directory")
        audio_bytes = path.read_bytes()
        if digest(audio_bytes) != row["audio_file_sha256"]:
            raise ValueError("Sealed audio file hash mismatch")
        audio = np.load(io.BytesIO(audio_bytes), allow_pickle=False)
        if audio.dtype != np.dtype("<f4") or audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Invalid sealed waveform")
        if len(audio) != row["frames"] or row["sample_rate"] != 16000:
            raise ValueError("Sealed waveform dimensions changed")
        if digest(audio.tobytes()) != row["waveform_sha256"]:
            raise ValueError("Sealed waveform hash mismatch")
        text = row["text"]
        normalized = normalize_text(text, preset="kspon")
        if digest(text.encode()) != row["reference_sha256"] or digest(normalized.encode()) != row["normalized_reference_sha256"]:
            raise ValueError("Sealed reference hash mismatch")
        selection = row["selection"]
        if len(normalized) != selection["reference_characters"]:
            raise ValueError("Sealed reference length mismatch")
        if abs(len(audio) / 16000 - selection["audio_duration"]) > 1 / 16000 + 1e-12:
            raise ValueError("Sealed duration mismatch")
        audio.setflags(write=False)
        samples.append({"audio": audio, "sample_rate": 16000, "text": text,
                        "metadata": {"id": "/".join((selection["dataset"], selection["subset"], selection["sample_id"])),
                                     "workload_group": selection["workload_group"],
                                     "selection_order": selection["order"]}})
    counts = Counter(s["metadata"]["workload_group"] for s in samples)
    if counts != dict.fromkeys(GROUPS, expected_count // 3):
        raise ValueError("Sealed group counts mismatch")
    expected_groups = [g for g in GROUPS for _ in range(expected_count // 3)]
    if [s["metadata"]["workload_group"] for s in samples] != expected_groups:
        raise ValueError("Sealed group order changed")
    if len({s["metadata"]["id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate sealed sample identity")
    return samples


def measure_workload(model, samples, batch_size, repetitions, warmup_samples, on_request,
                     synchronize, clock=time.perf_counter, after_call=lambda: {}, on_warmup=lambda row: None):
    """Reuse the supplied model. All input loading, audits and callbacks stay outside timing."""
    if batch_size not in (1, 4) or repetitions < 1:
        raise ValueError("Expected batch 1 or 4 and positive repetitions")
    if batch_size > 1 and not getattr(model, "supports_batch_transcribe", False):
        raise ValueError("Batch-4 track requires native batch transcription")
    groups = {g: [s for s in samples if s["metadata"]["workload_group"] == g] for g in GROUPS}
    for group in GROUPS:
        if len(groups[group]) < warmup_samples or len(groups[group]) % batch_size:
            raise ValueError("Insufficient warmup inputs or incomplete batch")
    total = 0
    for repeat in range(1, repetitions + 1):
        for group, items in groups.items():
            for sample in items[:warmup_samples]:
                prediction = model.transcribe(sample, sampling_rate=sample["sample_rate"])
                if not isinstance(prediction, str):
                    raise ValueError("Invalid warmup transcript")
                on_warmup({"repeat": repeat, "group": group, "id": sample["metadata"]["id"],
                           "generation_audit": after_call()})
            synchronize()
            for offset in range(0, len(items), batch_size):
                batch = items[offset:offset + batch_size]
                rates = [s["sample_rate"] for s in batch]
                synchronize()
                start = clock()
                if batch_size == 1:
                    predictions = [model.transcribe(batch[0], sampling_rate=rates[0])]
                else:
                    predictions = model.transcribe_batch(batch, sampling_rates=rates)
                synchronize()
                elapsed = clock() - start
                if not math.isfinite(elapsed) or elapsed <= 0:
                    raise ValueError("Invalid synchronized request time")
                if len(predictions) != len(batch) or any(not isinstance(p, str) for p in predictions):
                    raise ValueError("Missing or invalid batch transcript")
                audit = after_call()
                on_request({"repeat": repeat, "group": group, "batch_index": offset // batch_size,
                            "batch_size": len(batch), "seconds": elapsed,
                            "audio_seconds": sum(len(s["audio"]) / s["sample_rate"] for s in batch),
                            "samples": batch, "predictions": predictions, "generation_audit": audit})
                total += len(batch)
    return total


def quantile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_requests(requests, batch_size):
    summaries = []
    for repeat in sorted({r["repeat"] for r in requests}):
        per_group = {}
        for group in GROUPS:
            rows = [r for r in requests if r["repeat"] == repeat and r["group"] == group]
            seconds = sum(r["seconds"] for r in rows)
            audio_seconds = sum(r["audio_seconds"] for r in rows)
            stats = {"samples": sum(r["batch_size"] for r in rows), "requests": len(rows),
                     "audio_seconds": audio_seconds, "processing_seconds": seconds,
                     "throughput_rtfx": audio_seconds / seconds}
            if batch_size == 1:
                stats.update({f"request_latency_p{p}_ms": quantile([r["seconds"] * 1000 for r in rows], p / 100)
                              for p in (50, 95)})
            per_group[group] = stats
        summaries.append({"repeat": repeat, "groups": per_group,
                          "overall_throughput_rtfx": sum(g["throughput_rtfx"] for g in per_group.values()) / 3})
    values = [s["overall_throughput_rtfx"] for s in summaries]
    metrics = ["throughput_rtfx"] + (["request_latency_p50_ms", "request_latency_p95_ms"] if batch_size == 1 else [])
    groups = {}
    for group in GROUPS:
        groups[group] = {}
        for metric in metrics:
            observed = [s["groups"][group][metric] for s in summaries]
            groups[group][metric] = {"median": quantile(observed, .5), "min": min(observed), "max": max(observed)}
    return {"repeats": summaries, "groups": groups,
            "overall_throughput_rtfx": {"median": quantile(values, .5), "min": min(values), "max": max(values)}}


class GenerationAudit:
    """Retain generated IDs; inspect EOS outside the measured call, without changing decoding."""
    def __init__(self, backend, max_new_tokens, generation_mixin=None, strict=True):
        if generation_mixin is None:
            from transformers.generation.utils import GenerationMixin
            generation_mixin = GenerationMixin
        self.backend = backend
        self.maximum = max_new_tokens
        self.strict = strict
        self.owner = generation_mixin
        self.original = generation_mixin.generate
        self.calls = []
        self.targets = (backend, getattr(backend, "thinker", None))

        def generate(instance, *args, **kwargs):
            return self.capture(instance, *args, **kwargs)

        # Whisper strips EOS above this layer; Qwen delegates here through its thinker.
        # One benchmark process owns one model, and close() restores the original method.
        generation_mixin.generate = generate

    def close(self):
        self.owner.generate = self.original

    def capture(self, instance, *args, **kwargs):
        if not any(instance is target for target in self.targets):
            return self.original(instance, *args, **kwargs)
        config = kwargs.get("generation_config") or instance.generation_config
        maximum = kwargs.get("max_new_tokens", getattr(config, "max_new_tokens", None))
        eos = kwargs.get("eos_token_id", config.eos_token_id)
        output = self.original(instance, *args, **kwargs)
        prefix = 0
        if getattr(instance.config, "is_encoder_decoder", False):
            decoder_ids = kwargs.get("decoder_input_ids")
            prefix = int(decoder_ids.shape[-1]) if decoder_ids is not None else 1
        else:
            inputs = kwargs.get("input_ids")
            if inputs is None and args:
                inputs = args[0]
            if inputs is None:
                raise ValueError("Cannot audit decoder-only generation prefix")
            prefix = int(inputs.shape[-1])
        self.calls.append((output, prefix, maximum, eos))
        return output

    def check(self):
        lengths = []
        reasons = []
        missing_eos = 0
        if not self.calls:
            raise ValueError("No generation call was observed")
        try:
            for output, prefix, maximum, eos in self.calls:
                eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
                if maximum != self.maximum:
                    raise ValueError("Generation token budget differs from protocol")
                sequences = output.sequences if hasattr(output, "sequences") else output
                for sequence in sequences.detach().cpu().tolist():
                    generated = sequence[prefix:]
                    stops = [i for i, token in enumerate(generated) if token in eos_ids]
                    if not stops:
                        missing_eos += 1
                        if len(generated) != self.maximum:
                            raise ValueError("Generation ended without EOS before its token budget")
                        if self.strict:
                            raise ValueError("Generation ended without EOS; token limit may have truncated output")
                    lengths.append(stops[0] + 1 if stops else len(generated))
                    reasons.append("eos" if stops else "token_limit")
            return {"sequences": len(lengths), "all_reached_eos": missing_eos == 0,
                    "sequences_without_eos": missing_eos,
                    "sequences_at_token_limit": missing_eos,
                    "termination_reasons": reasons, "generated_tokens": lengths,
                    "max_generated_tokens_observed": max(lengths)}
        finally:
            self.calls.clear()
