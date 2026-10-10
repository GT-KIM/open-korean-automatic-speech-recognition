import unittest
from types import SimpleNamespace

from openkoasr.evaluation.speed import (GenerationAudit, measure_workload, summarize_requests,
                                       summarize_termination, termination_policy)


def samples(count=20):
    return [{"audio": [0.] * 16, "sample_rate": 16, "text": "reference",
             "metadata": {"id": f"{group}/{i}", "workload_group": group}}
            for group in ("clean", "other", "telephone") for i in range(count)]


class SpeedRunnerTest(unittest.TestCase):
    def test_one_model_reused_and_warmup_repeated_without_entering_measurements(self):
        events, results = [], []
        state = {"clock": 0.}

        class Model:
            supports_batch_transcribe = True

            def transcribe(self, sample, sampling_rate):
                state["clock"] += 2
                events.append(("single", sample["metadata"]["id"]))
                return ""

            def transcribe_batch(self, batch, sampling_rates):
                state["clock"] += 2
                events.append(("batch", [s["metadata"]["id"] for s in batch]))
                return [""] * len(batch)

        def synchronize():
            state["clock"] += .5

        def record(row):
            state["clock"] += 1000  # File writing/scoring must not affect measured seconds.
            results.append(row)

        total = measure_workload(Model(), samples(), 4, 3, 16, record, synchronize,
                                 clock=lambda: state["clock"])
        self.assertEqual(total, 180)
        self.assertEqual(len(results), 45)
        self.assertTrue(all(r["seconds"] == 2.5 for r in results))
        self.assertEqual(sum(e[0] == "single" for e in events), 16 * 3 * 3)
        self.assertEqual(events[:16], [("single", f"clean/{i}") for i in range(16)])
        self.assertEqual(events[16], ("batch", [f"clean/{i}" for i in range(4)]))
        for repeat in (1, 2, 3):
            measured = [s["metadata"]["id"] for r in results if r["repeat"] == repeat for s in r["samples"]]
            self.assertEqual(measured, [s["metadata"]["id"] for s in samples()])

    def test_rejects_missing_batch_predictions_instead_of_dropping_inputs(self):
        model = SimpleNamespace(supports_batch_transcribe=True,
                                transcribe_batch=lambda batch, sampling_rates: [""])
        ticks = iter((0., 1.))
        with self.assertRaisesRegex(ValueError, "batch transcript"):
            measure_workload(model, samples(4), 4, 1, 0, lambda r: None,
                             lambda: None, clock=lambda: next(ticks))

    def test_sums_batch_time_once_and_does_not_publish_batch_latency_as_request_latency(self):
        rows = [{"repeat": 1, "group": group, "batch_size": 4,
                 "seconds": 2., "audio_seconds": 8.} for group in ("clean", "other", "telephone")]
        result = summarize_requests(rows, 4)
        self.assertEqual(result["overall_throughput_rtfx"]["median"], 4.)
        self.assertEqual(result["repeats"][0]["groups"]["clean"]["processing_seconds"], 2.)
        self.assertNotIn("request_latency_p95_ms", result["repeats"][0]["groups"]["clean"])

    def test_group_latency_summaries_keep_repeat_variation(self):
        rows = [{"repeat": repeat, "group": group, "batch_size": 1,
                 "seconds": seconds, "audio_seconds": 1.}
                for repeat, seconds in ((1, 1.), (2, 3.), (3, 2.))
                for group in ("clean", "other", "telephone")]
        result = summarize_requests(rows, 1)
        self.assertEqual(result["groups"]["clean"]["request_latency_p95_ms"],
                         {"median": 2000., "min": 1000., "max": 3000.})

    def test_generation_audit_excludes_decoder_prompt_eos_and_rejects_truncation(self):
        class Tensor:
            shape = (1, 2)

            def __init__(self, rows): self.rows = rows
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return self.rows

        class Mixin:
            def generate(self, **kwargs): return self.result

        backend = Mixin()
        backend.config = SimpleNamespace(is_encoder_decoder=False)
        backend.generation_config = SimpleNamespace(eos_token_id=9)
        backend.result = Tensor([[9, 1, 2, 3]])
        original = Mixin.generate
        audit = GenerationAudit(backend, 2, Mixin)
        try:
            backend.generate(input_ids=Tensor([[9, 1]]), max_new_tokens=2)
            with self.assertRaisesRegex(ValueError, "without EOS"):
                audit.check()
            self.assertEqual(audit.calls, [])
            backend.result = Tensor([[9, 1, 2, 9]])
            backend.generate(input_ids=Tensor([[9, 1]]), max_new_tokens=2)
            self.assertTrue(audit.check()["all_reached_eos"])
        finally:
            audit.close()
        self.assertIs(Mixin.generate, original)
        backend.result = Tensor([[9, 1, 2, 3]])
        audit = GenerationAudit(backend, 2, Mixin, strict=False)
        try:
            backend.generate(input_ids=Tensor([[9, 1]]), max_new_tokens=2)
            diagnostic = audit.check()
            self.assertFalse(diagnostic["all_reached_eos"])
            self.assertEqual(diagnostic["sequences_without_eos"], 1)
        finally:
            audit.close()

    def test_generation_audit_observes_eos_before_whisper_postprocessing(self):
        class Tensor:
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return [[1, 2, 9]]

        class Mixin:
            def generate(self, **kwargs): return Tensor()

        class Whisper(Mixin):
            config = SimpleNamespace(is_encoder_decoder=True)
            generation_config = SimpleNamespace(eos_token_id=9, max_new_tokens=128)

            def generate(self, **kwargs):
                original = super().generate(generation_config=self.generation_config)
                return original.tolist()[0][:-1]

        backend = Whisper()
        audit = GenerationAudit(backend, 128, Mixin)
        try:
            self.assertEqual(backend.generate(), [1, 2])
            self.assertTrue(audit.check()["all_reached_eos"])
        finally:
            audit.close()

    def test_capped_outputs_stay_in_denominator_and_legacy_contract_stays_strict(self):
        audit = {"sequences": 4, "sequences_without_eos": 1, "sequences_at_token_limit": 1}
        rows = [{"repeat": repeat, "group": group, "batch_size": 4,
                 "generation_audit": dict(audit)} for repeat in (1, 2, 3)
                for group in ("clean", "other", "telephone")]
        warmups = [{"repeat": 1, "group": "clean", "generation_audit": {
            "sequences": 1, "sequences_without_eos": 1, "sequences_at_token_limit": 1}}]
        retained = summarize_termination(rows, warmups, "retain_budget_terminated")
        self.assertTrue(retained["contract_passed"])
        self.assertFalse(retained["all_generations_reached_eos"])
        self.assertEqual(retained["measured"], {"sequences": 36, "eos": 27,
                                              "token_limit": 9, "token_limit_rate": .25})
        self.assertEqual(retained["warmup"]["token_limit_rate"], 1.)
        self.assertEqual(retained["repeats"][1]["groups"]["other"]["token_limit_rate"], .25)
        self.assertEqual(termination_policy({}), "require_eos")
        self.assertFalse(summarize_termination(rows, warmups, termination_policy({}))["contract_passed"])
        with self.assertRaisesRegex(ValueError, "Unknown"):
            termination_policy({"termination": {"policy": "ignore_failures"}})
        rows[0]["generation_audit"]["sequences"] = 3
        with self.assertRaisesRegex(ValueError, "count"):
            summarize_termination(rows, warmups, "retain_budget_terminated")

    def test_retaining_budget_termination_never_accepts_unexplained_early_stop(self):
        class Tensor:
            shape = (1, 2)
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return [[1, 2, 3]]

        class Mixin:
            config = SimpleNamespace(is_encoder_decoder=False)
            generation_config = SimpleNamespace(eos_token_id=9)
            def generate(self, **kwargs): return Tensor()

        backend = Mixin()
        audit = GenerationAudit(backend, 128, Mixin, strict=False)
        try:
            backend.generate(input_ids=Tensor(), max_new_tokens=128)
            with self.assertRaisesRegex(ValueError, "before its token budget"):
                audit.check()
            self.assertEqual(audit.calls, [])
        finally:
            audit.close()


if __name__ == "__main__":
    unittest.main()
