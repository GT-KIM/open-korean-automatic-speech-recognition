import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openkoasr.evaluation import EvaluationRunner
from openkoasr.model.revisions import pinned_revisions, verify_loaded_revision, verify_processor_revision

MODEL_SHA = "b" * 40
PROCESSOR_SHA = "c" * 40


class RevisionContractTest(unittest.TestCase):
    def test_pins_are_immutable_and_processor_defaults_to_model(self):
        self.assertEqual(pinned_revisions(SimpleNamespace(model_revision=MODEL_SHA)), (MODEL_SHA, MODEL_SHA))
        self.assertEqual(pinned_revisions(SimpleNamespace()), (None, None))
        for config in (SimpleNamespace(model_revision="main"), SimpleNamespace(model_revision="b" * 7),
                       SimpleNamespace(processor_revision=PROCESSOR_SHA)):
            with self.assertRaises(ValueError):
                pinned_revisions(config)

    def test_missing_or_different_loaded_model_revision_fails_closed(self):
        for actual in (None, "a" * 40):
            with self.assertRaisesRegex(ValueError, "does not match"):
                verify_loaded_revision(SimpleNamespace(config=SimpleNamespace(_commit_hash=actual)), MODEL_SHA)

    def test_processor_evidence_distinguishes_pin_from_observation_and_rejects_mismatch(self):
        self.assertEqual(verify_processor_revision(SimpleNamespace(), MODEL_SHA), "pinned_loader_argument")
        processor = SimpleNamespace(tokenizer=SimpleNamespace(init_kwargs={"_commit_hash": MODEL_SHA}))
        self.assertEqual(verify_processor_revision(processor, MODEL_SHA), "loaded_metadata")
        with self.assertRaisesRegex(ValueError, "processor revision"):
            verify_processor_revision(processor, PROCESSOR_SHA)

    def test_cli_model_overrides_do_not_mutate_shared_configs(self):
        before = EvaluationRunner.from_names("mock", "whisper_tiny")
        changed = EvaluationRunner.from_names("mock", "whisper_tiny", model_overrides={
            "dtype": "bfloat16", "model_revision": MODEL_SHA})
        after = EvaluationRunner.from_names("mock", "whisper_tiny")
        self.assertEqual(changed.model_config.dtype, "bfloat16")
        self.assertEqual(after.model_config.dtype, before.model_config.dtype)
        self.assertFalse(hasattr(after.model_config, "model_revision"))
        for overrides in ({"model_revision": "main"}, {"api_key": "secret"}, {"max_inference_batch_size": 0}):
            with self.assertRaises(ValueError):
                EvaluationRunner.from_names("mock", "whisper_tiny", model_overrides=overrides)


@unittest.skipUnless(importlib.util.find_spec("transformers"), "requires transformers")
class LoaderRevisionTest(unittest.TestCase):
    def config(self):
        return SimpleNamespace(repo_name="org/model", dtype="float16", device="cpu",
                               model_revision=MODEL_SHA, processor_revision=PROCESSOR_SHA,
                               max_inference_batch_size=4, max_new_tokens=256)

    def backend(self):
        return SimpleNamespace(config=SimpleNamespace(_commit_hash=MODEL_SHA), to=Mock(), eval=Mock())

    def test_whisper_loads_model_and_processor_at_their_pins(self):
        from openkoasr.model.whisper import WhisperASRInferenceModel
        with patch("openkoasr.model.whisper.AutoModelForSpeechSeq2Seq.from_pretrained", return_value=self.backend()) as model_loader, patch(
            "openkoasr.model.whisper.AutoProcessor.from_pretrained", return_value=SimpleNamespace()) as processor_loader:
            adapter = WhisperASRInferenceModel(self.config())
        self.assertEqual(model_loader.call_args.kwargs["revision"], MODEL_SHA)
        self.assertEqual(processor_loader.call_args.kwargs["revision"], PROCESSOR_SHA)
        self.assertEqual(adapter.processor_revision, PROCESSOR_SHA)

    def test_ctc_loads_model_and_processor_at_their_pins(self):
        from openkoasr.model.hf_ctc import HfCtcASRInferenceModel
        with patch("openkoasr.model.hf_ctc.AutoModelForCTC.from_pretrained", return_value=self.backend()) as model_loader, patch(
            "openkoasr.model.hf_ctc.AutoProcessor.from_pretrained", return_value=SimpleNamespace()) as processor_loader:
            adapter = HfCtcASRInferenceModel(self.config())
        self.assertEqual(model_loader.call_args.kwargs["revision"], MODEL_SHA)
        self.assertEqual(processor_loader.call_args.kwargs["revision"], PROCESSOR_SHA)
        self.assertEqual(adapter.processor_revision, PROCESSOR_SHA)

    @unittest.skipUnless(importlib.util.find_spec("qwen_asr"), "requires qwen-asr")
    def test_qwen_pins_processor_separately_instead_of_using_mutable_sdk_default(self):
        from openkoasr.model.qwen3_asr import Qwen3ASRInferenceModel
        backend, processor = self.backend(), SimpleNamespace()
        with patch("openkoasr.model.qwen3_asr.AutoModel.from_pretrained", return_value=backend) as model_loader, patch(
            "openkoasr.model.qwen3_asr.AutoProcessor.from_pretrained", return_value=processor) as processor_loader, patch(
            "openkoasr.model.qwen3_asr.Qwen3ASRModel") as wrapper:
            wrapper.return_value = SimpleNamespace(model=backend, processor=processor)
            adapter = Qwen3ASRInferenceModel(self.config())
        wrapper.from_pretrained.assert_not_called()
        self.assertEqual(model_loader.call_args.kwargs["revision"], MODEL_SHA)
        self.assertEqual(processor_loader.call_args.kwargs["revision"], PROCESSOR_SHA)
        self.assertEqual(wrapper.call_args.kwargs["max_inference_batch_size"], 4)
        self.assertIs(adapter.processor, processor)


if __name__ == "__main__":
    unittest.main()
