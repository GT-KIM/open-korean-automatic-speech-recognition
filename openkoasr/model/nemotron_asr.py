"""Offline, native-batch Nemotron 3.5 RNN-T transcription (Transformers >=5.13)."""
import numpy as np
import torch
from transformers import AutoModelForRNNT, AutoProcessor

from openkoasr.dataset.sample import get_sample_audio
from openkoasr.model.base import BaseASRInferenceModel
from openkoasr.model.revisions import (
    pinned_revisions, verify_loaded_revision, verify_processor_revision,
)


class NemotronASRInferenceModel(BaseASRInferenceModel):
    supports_batch_transcribe = True

    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        self.model_revision, self.processor_revision = pinned_revisions(model_config)
        self.initialize_model()
        self.initialize_processor()
        self.last_generation = None
        self.generation_overrides = {"do_sample": False, "num_beams": 1}
        self.transcription_options = {"language": model_config.language,
                                      "mode": "offline_full_utterance",
                                      "num_lookahead_tokens": model_config.num_lookahead_tokens,
                                      "max_symbols_per_step": self.model.max_symbols_per_step}

    def initialize_model(self):
        self.model = AutoModelForRNNT.from_pretrained(
            self.model_config.repo_name, revision=self.model_revision,
            dtype=self.TORCH_DTYPE[self.model_config.dtype],
        ).to(self.model_config.device).eval()
        verify_loaded_revision(self.model, self.model_revision)
        if self.model.config.model_type != "nemotron3_5_asr":
            raise ValueError("Expected the multilingual Nemotron 3.5 RNN-T model")

    def initialize_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_config.repo_name, revision=self.processor_revision,
        )
        self.processor_revision_source = verify_processor_revision(
            self.processor, self.processor_revision)
        self.processor.set_num_lookahead_tokens(self.model_config.num_lookahead_tokens)

    def inference_sample(self, sample, sampling_rate=16000):
        return self.transcribe_batch([sample], [sampling_rate])[0]

    def transcribe_batch(self, samples, sampling_rates=None):
        self.last_generation = None
        if sampling_rates is None:
            sampling_rates = [16000] * len(samples)
        if len(samples) != len(sampling_rates):
            raise ValueError("Sample/rate count mismatch")
        if not samples:
            return []
        expected_rate = self.processor.feature_extractor.sampling_rate
        if any(rate != expected_rate for rate in sampling_rates):
            raise ValueError(f"Nemotron requires audio resampled to {expected_rate} Hz")
        waveforms = []
        for sample in samples:
            audio = get_sample_audio(sample)
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().float().numpy()
            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
                raise ValueError("Expected a nonempty finite mono waveform")
            waveforms.append(audio)
        inputs = self.processor(waveforms, sampling_rate=expected_rate,
                                language=self.model_config.language, return_tensors="pt")
        # Keep prompt IDs and masks integral. Only floating features adopt model dtype.
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            output = self.model.generate(**inputs, return_dict_in_generate=True,
                                         **self.generation_overrides)
        predictions = self.processor.batch_decode(output.sequences, skip_special_tokens=True)
        if len(predictions) != len(samples) or any(not isinstance(p, str) for p in predictions):
            raise ValueError("Nemotron returned invalid batch transcripts")
        # Retain tensors; the benchmark performs the audit after the timed call.
        self.last_generation = (output, inputs["attention_mask"], inputs["prompt_ids"])
        return predictions

    def audit_last_generation(self):
        """Require every input frame to be consumed; RNN-T has no EOS contract."""
        if self.last_generation is None:
            raise ValueError("No Nemotron generation to audit")
        output, mask, prompts = self.last_generation
        self.last_generation = None
        lengths = self.model._get_subsampling_output_length(mask.sum(-1)).cpu().tolist()
        sequences = output.sequences.cpu().tolist()
        durations = output.durations.cpu().tolist()
        prompt_ids = prompts.cpu().tolist()
        expected_prompt = self.processor.prompt_dictionary[self.model_config.language]
        if not (len(lengths) == len(sequences) == len(durations) == len(prompt_ids)):
            raise ValueError("RNN-T audit batch count mismatch")
        rows = []
        blank = self.model.config.blank_token_id
        for length, tokens, advances, prompt in zip(lengths, sequences, durations, prompt_ids):
            if prompt != expected_prompt or len(tokens) != len(advances) or length <= 0:
                raise ValueError("Invalid RNN-T audit inputs")
            if not tokens or tokens[0] != blank or advances[0] != 0:
                raise ValueError("Unexpected RNN-T decoder prefix")
            consumed = forced = emitted = 0
            for token, advance in zip(tokens[1:], advances[1:]):
                if advance not in (0, 1):
                    raise ValueError("Invalid RNN-T frame advance")
                if consumed >= length:
                    break  # Finished rows can be padded while longer batch rows decode.
                emitted += int(token != blank)
                forced += int(token != blank and advance == 1)
                consumed += advance
            if consumed != length:
                raise ValueError(f"RNN-T stopped before consuming all frames: {consumed}/{length}")
            rows.append({"encoder_frames": length, "consumed_frames": consumed,
                         "forced_frame_advances": forced, "emitted_tokens": emitted,
                         "prompt_id": prompt, "reason": "encoder_exhausted"})
        return {"sequences": len(rows), "policy": "require_encoder_exhaustion",
                "all_encoder_frames_consumed": True, "rows": rows}
