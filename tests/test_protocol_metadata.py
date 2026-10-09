import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openkoasr.evaluation.provenance import capture_reproducibility, code_metadata
from openkoasr.protocol import STANDARD_PROTOCOL, evaluation_protocol_id
from scripts.leaderboard_metadata import aggregate_metadata, recover_run_metadata
from scripts.validate_leaderboard_data import validate_row, validate_rows
from test_validate_leaderboard_data import _row


class ProtocolMetadataTest(unittest.TestCase):
    def test_protocol_checks_normalization_and_outlier_rules(self):
        self.assertEqual(evaluation_protocol_id("kspon", {"metric": "cer", "threshold": 1}), STANDARD_PROTOCOL)
        for preset, policy in (
            (None, {"metric": "cer", "threshold": 1}),
            ("unknown", {"metric": "cer", "threshold": 1}),
            ("kspon", {"metric": "unknown", "threshold": 1}),
            ("kspon", {"metric": "cer", "threshold": True}),
            ("kspon", {"metric": "cer", "threshold": float("nan")}),
            ("kspon", {"metric": "cer", "threshold": -1}),
        ):
            with self.subTest(preset=preset, policy=policy), self.assertRaises(ValueError):
                evaluation_protocol_id(preset, policy)

    def test_recovers_only_matching_summary_without_changing_scores(self):
        row = _row()
        for key in ("normalization_preset", "evaluation_protocol", "metadata_status", "reproducibility"):
            del row[key]
        original = copy.deepcopy(row)
        metadata = {
            "run_id": row["run_id"], "model_name": row["model"],
            "dataset_name": row["dataset"], "dataset_subset": row["subset"],
            "evaluated_samples": row["evaluated_samples"], "outlier_policy": row["outlier_policy"],
            "normalization_preset": "kspon", "batch_size": 2, "num_workers": 0,
            "warmup_samples": 1, "limit": None, "environment": {"python": "3.13"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
            raw = path.read_bytes()
            recovered = recover_run_metadata(row, path)
            self.assertEqual(row, original)
            self.assertEqual(path.read_bytes(), raw)
            self.assertTrue(all(recovered[key] == value for key, value in original.items()))
            self.assertEqual(recovered["evaluation_protocol"], STANDARD_PROTOCOL)
            self.assertEqual(recovered["metadata_status"], "recovered")
            provenance = recovered["reproducibility"]
            self.assertIsNone(provenance["model_revision"])
            self.assertIsNone(provenance["code"]["commit"])
            self.assertEqual(provenance["execution"]["batch_size"], 2)
            self.assertEqual(provenance["evidence"]["summary_sha256"], hashlib.sha256(raw).hexdigest())
            problems = []
            validate_row("test", 0, recovered, problems)
            self.assertEqual(problems, [])
            metadata["run_id"] = "other-run"
            path.write_text(json.dumps({"metadata": metadata}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "run_id does not match"):
                recover_run_metadata(row, path)

    def test_aggregate_keeps_individual_execution_settings(self):
        rows = [_row() for _ in range(4)]
        for index, row in enumerate(rows):
            row["run_id"] = f"run-{index}"
            row["reproducibility"]["execution"]["batch_size"] = index + 1
        combined = aggregate_metadata(rows)
        self.assertEqual([source["reproducibility"]["execution"]["batch_size"]
                          for source in combined["reproducibility"]["source_runs"]], [1, 2, 3, 4])
        rows[-1]["evaluation_protocol"] = "v2/kspon/cer>1.0"
        with self.assertRaisesRegex(ValueError, "Mixed evaluation_protocol"):
            aggregate_metadata(rows)

    def test_rejects_missing_inconsistent_or_forged_metadata(self):
        for change in (
            {"normalization_preset": None}, {"normalization_preset": "raw"},
            {"evaluation_protocol": "v2/kspon/cer>1.0"},
            {"outlier_policy": {"metric": "cer", "threshold": 9999}},
            {"reproducibility": {}}, {"metadata_status": "legacy"},
        ):
            with self.subTest(change=change):
                row = _row()
                row.update(change)
                problems = []
                validate_row("test", 0, row, problems)
                self.assertTrue(problems)
        row = _row()
        row["metadata_status"] = "recovered"
        row["reproducibility"]["evidence"] = {"run_id": "other-run", "summary_sha256": "a" * 64}
        problems = []
        validate_row("test", 0, row, problems)
        self.assertIn("summary evidence", " ".join(problems))

    def test_distinct_valid_protocols_can_be_published(self):
        first, second = _row(), _row()
        second.update(run_id="run-2", normalization_preset="raw", evaluation_protocol="v1/raw/cer>1.0")
        problems = []
        validate_rows("test", [first, second], problems)
        self.assertEqual(problems, [])

    def test_loaded_revision_and_allowlisted_settings_are_recorded(self):
        backend = SimpleNamespace(config=SimpleNamespace(_commit_hash="b" * 40),
                                  generation_config=SimpleNamespace(num_beams=5))
        config = SimpleNamespace(repo_name="org/model", language="ko", api_key="hidden", rootpath="private")
        for wrapped in (backend, SimpleNamespace(model=backend)):
            with self.subTest(wrapped=wrapped), patch(
                "openkoasr.evaluation.provenance.code_metadata", return_value={"commit": None}
            ):
                data = capture_reproducibility(SimpleNamespace(model=wrapped), config, {"batch_size": 1}, {})
                self.assertEqual(data["model_revision"], "b" * 40)
                self.assertEqual(data["generation_config"], {"num_beams": 5})
                self.assertEqual(data["model_config"], {"repo_name": "org/model", "language": "ko"})
                self.assertNotIn("hidden", json.dumps(data))
                self.assertNotIn("private", json.dumps(data))

    def test_source_hash_remains_available_without_git(self):
        with patch("openkoasr.evaluation.provenance.subprocess.check_output", side_effect=OSError):
            data = code_metadata()
        self.assertIsNone(data["commit"])
        self.assertIsNone(data["dirty"])
        self.assertRegex(data["source_sha256"], r"^[0-9a-f]{64}$")

    @patch("openkoasr.evaluation.provenance.code_metadata", return_value={})
    def test_observed_dtype_and_call_overrides_do_not_copy_requested_defaults(self, _code):
        backend = SimpleNamespace(
            config=SimpleNamespace(_commit_hash="b" * 40),
            generation_config=SimpleNamespace(max_new_tokens=20, do_sample=False, temperature=0.8, top_k=50),
            parameters=lambda: iter([SimpleNamespace(dtype="torch.bfloat16", numel=lambda: 12)]),
        )
        model = SimpleNamespace(
            model=SimpleNamespace(model=backend, max_inference_batch_size=4),
            processor_revision="c" * 40, generation_overrides={"max_new_tokens": 256, "secret": "hidden"},
        )
        config = SimpleNamespace(family="qwen3_asr", dtype="float16")
        data = capture_reproducibility(model, config, {"batch_size": 8}, {})
        self.assertEqual(data["model_config"]["dtype"], "float16")
        self.assertEqual(data["inference"]["effective_dtype"], "bfloat16")
        self.assertEqual(data["inference"]["backend_batch_size"], 4)
        self.assertEqual(data["processor_revision"], "c" * 40)
        decoding = data["inference"]["decoding"]
        self.assertEqual(decoding["resolved_parameters"]["max_new_tokens"], 256)
        self.assertNotIn("temperature", decoding["resolved_parameters"])
        self.assertNotIn("hidden", json.dumps(data))
        self.assertEqual(decoding["inactive_sampling_parameters"], ["temperature", "top_k"])

    @patch("openkoasr.evaluation.provenance.code_metadata", return_value={})
    def test_unknown_and_mixed_dtypes_are_not_reported_as_config_dtype(self, _code):
        config = SimpleNamespace(family="whisper", dtype="float16")
        model = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace()))
        self.assertIsNone(capture_reproducibility(model, config, {}, {})["inference"]["effective_dtype"])
        model.model.parameters = lambda: iter([
            SimpleNamespace(dtype="torch.float16", numel=lambda: 12),
            SimpleNamespace(dtype="torch.float32", numel=lambda: 3),
        ])
        observed = capture_reproducibility(model, config, {}, {})["inference"]
        self.assertEqual(observed["effective_dtype"], "mixed:float16+float32")
        self.assertEqual(observed["parameter_dtype_numel"], {"float16": 12, "float32": 3})


if __name__ == "__main__":
    unittest.main()
