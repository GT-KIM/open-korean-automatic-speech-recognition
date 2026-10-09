import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from metadata_fixtures import public_metadata

from scripts.generate_leaderboard import dedupe_rows, normalize_metric_schema, sanitize_public_command, render_markdown, format_cell


class GenerateLeaderboardTest(unittest.TestCase):
    def test_verified_report_does_not_revive_legacy_aliases(self):
        report_path = Path('doc/benchmarks/server_accuracy_results_20261008.json')
        report = json.loads(report_path.read_bytes())
        current = next(row for row in report['rows']
                       if row['model_repo'] == 'openai/whisper-tiny' and row['subset'] == 'clean')
        old = {**current, 'model': 'old_tiny_alias', 'run_id': 'old-documented',
               '_artifact': 'doc/submitted_results.json'}
        legacy = {**old, 'run_id': 'readme-legacy', 'metadata_status': 'legacy',
                  'reproducibility': {}, 'normalization_preset': None, 'evaluation_protocol': None}
        temp_root = Path.cwd() / '.tmp_tests'
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            submitted = root / 'submitted.json'
            submitted.write_text(json.dumps([old, legacy]), encoding='utf-8')
            subprocess.run([
                sys.executable, 'scripts/generate_leaderboard.py',
                '--results_dir', str(root / 'empty'), '--submitted_rows_path', str(submitted),
                '--verified_accuracy_path', str(report_path),
                '--data_path', str(root / 'data.json'), '--markdown_path', str(root / 'leaderboard.md'),
            ], check=True, capture_output=True)
            rows = json.loads((root / 'data.json').read_bytes())
            self.assertEqual({row['run_id'] for row in rows}, {row['run_id'] for row in report['rows']})

    def test_markdown_separates_published_slices_and_reference_protocols(self):
        rows = json.loads(Path("doc/leaderboard_data.json").read_text(encoding="utf-8"))
        original = json.dumps(rows)
        markdown = render_markdown(rows)
        standard, references = markdown.split("## Reference results", 1)
        clean = standard.split("### KsponSpeech · clean", 1)[1].split("###", 1)[0]
        tiny_clean = next(row for row in rows if row.get("model_repo") == "openai/whisper-tiny"
                          and row.get("subset") == "clean" and row.get("evaluation_protocol") == "v1/kspon/cer>1.0")
        self.assertIn(f"{tiny_clean['metrics']['macro']['cer']:.2%}", clean)
        self.assertNotIn("v1/punctuation_agnostic", clean)
        self.assertNotIn("readme-legacy", standard)
        self.assertIn("### KsponSpeech · other", standard)
        self.assertIn("### AIHubLowQualityTelephone · all", standard)
        self.assertIn("### AIHubLowQualityTelephone · D01", standard)
        self.assertIn("v1/punctuation_agnostic/cer>1.0", references)
        self.assertIn("Unverified protocol", references)
        self.assertIn("26.85%", references)
        self.assertIn("readme-legacy-whisper-large-v3-kspon-clean", references)
        for row in rows:
            self.assertEqual(markdown.count("| " + row["run_id"] + " |"), 1, row["run_id"])
        self.assertEqual(render_markdown(list(reversed(rows))), markdown)
        self.assertEqual(json.dumps(rows), original)

    def test_standard_markdown_requires_matching_protocol_fields(self):
        for change in ({"evaluation_protocol": None}, {"normalization_preset": "raw"},
                       {"outlier_policy": {"metric": "cer", "threshold": 2.0}}):
            with self.subTest(change=change):
                row = {**public_metadata(), "model": "model", "dataset": "KsponSpeech",
                       "subset": "clean", "metrics": {"macro": {"cer": 0.1}},
                       "outlier_policy": {"metric": "cer", "threshold": 1.0}, **change}
                markdown = render_markdown([row])
                self.assertNotIn("## Standard results", markdown)
                self.assertIn("## Reference results", markdown)

    def test_empty_markdown_has_no_misleading_result_sections(self):
        markdown = render_markdown([])
        self.assertIn("No runs yet", markdown)
        self.assertNotIn("## Standard results", markdown)
        self.assertNotIn("## Reference results", markdown)

    def test_markdown_distinguishes_main_outliers_and_all_samples_cer(self):
        row = {"model": "model", "metrics": {"macro": {"cer": 0.1},
               "all_samples_micro": {"cer": 0.4}}, "outlier_count": 2, "total_samples": 10}
        markdown = render_markdown([row])
        self.assertIn("Main CER", markdown)
        self.assertIn("All-sample CER", markdown)
        self.assertIn("outlier", markdown.lower())
        self.assertIn("| 10.00% | 20.00% (2 / 10) | 40.00% |", markdown)
        del row["metrics"]["all_samples_micro"]
        self.assertIn("N/A", render_markdown([row]))

    def test_display_units_preserve_raw_metrics(self):
        row = {"metrics": {"macro": {"cer": 1.25, "latency": 0.2365, "rtfx": 19.307}}}
        original = json.dumps(row)
        self.assertEqual(format_cell(row, "cer"), "125.00%")
        self.assertEqual(format_cell(row, "rtfx"), "19.31×")
        self.assertEqual(format_cell(row, "latency"), "236.5 ms")
        self.assertEqual(json.dumps(row), original)

    def test_invalid_legacy_rtf_is_preserved_for_validation(self):
        for value in (float("nan"), float("inf"), True, -1):
            with self.subTest(value=value):
                row = {"metrics": {"macro": {"rtf": value}}}
                normalized = normalize_metric_schema(row)
                self.assertIn("rtf", normalized["metrics"]["macro"])
                self.assertNotIn("rtfx", normalized["metrics"]["macro"])

    def test_invalid_public_result_does_not_overwrite_outputs(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as directory:
            root = Path(directory)
            submitted = root / "submitted.json"
            submitted.write_text(json.dumps([{
                "run_id": "bad", "model": "bad", "dataset": "KsponSpeech",
                "is_full_evaluation": True, "metrics": {"macro": {"cer": float("nan")}},
            }]), encoding="utf-8")
            markdown, data = root / "leaderboard.md", root / "data.json"
            for path in (markdown, data):
                path.write_text("previous output", encoding="utf-8")
            completed = subprocess.run([
                sys.executable, "scripts/generate_leaderboard.py",
                "--results_dir", str(root / "empty"), "--submitted_rows_path", str(submitted),
                "--markdown_path", str(markdown), "--data_path", str(data),
            ], capture_output=True, text=True)
            self.assertNotEqual(completed.returncode, 0)
            for path in (markdown, data):
                self.assertEqual(path.read_text(encoding="utf-8"), "previous output")

    def test_generates_markdown_and_data(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        root = temp_root / f"leaderboard-{uuid.uuid4().hex}"
        root.mkdir()
        run_dir = root / "results" / "run-1"
        run_dir.mkdir(parents=True)
        (run_dir / "leaderboard_row.json").write_text(
            json.dumps(
                {
                    "run_id": "run-1",
                    **public_metadata(),
                    "model": "whisper_tiny",
                    "model_repo": "openai/whisper-tiny",
                    "dataset": "KsponSpeech",
                    "subset": None,
                    "gpu": None,
                    "metrics": {"macro": {"wer": 0.0, "cer": 0.0, "mer": 0.0, "jer": 0.0,
                                          "rtf": 0.25, "latency": 0.1}},
                        "total_samples": 2,
                        "evaluated_samples": 2,
                        "dataset_total_samples": 2,
                        "is_full_evaluation": True,
                        "outlier_count": 0,
                        "outlier_policy": {"metric": "cer", "threshold": 1.0},
                }
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                "scripts/generate_leaderboard.py",
                "--results_dir",
                str(root / "results"),
                "--markdown_path",
                str(root / "leaderboard.md"),
                "--data_path",
                str(root / "leaderboard_data.json"),
                "--submitted_rows_path",
                str(root / "submitted_results.json"),
            ],
            check=True,
        )

        markdown = (root / "leaderboard.md").read_text(encoding="utf-8")
        data = json.loads((root / "leaderboard_data.json").read_text(encoding="utf-8"))
        self.assertIn("[whisper_tiny](https://huggingface.co/openai/whisper-tiny)", markdown)
        self.assertIn("| 4.00× |", markdown)
        self.assertIn("Live leaderboard", markdown)
        self.assertEqual(data[0]["model"], "whisper_tiny")
        self.assertNotIn("rtf", data[0]["metrics"]["macro"])
        self.assertEqual(data[0]["metrics"]["macro"]["rtfx"], 4.0)

    def test_merges_submitted_rows(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        root = temp_root / f"leaderboard-submitted-{uuid.uuid4().hex}"
        root.mkdir()
        submitted_path = root / "submitted_results.json"
        submitted_path.write_text(
            json.dumps(
                [
                    {
                        "run_id": "submitted-1",
                        **public_metadata(),
                        "model": "submitted",
                        "model_repo": "org/submitted",
                        "dataset": "KsponSpeech",
                        "metrics": {"macro": {"cer": 0.1, "wer": 0.2, "mer": 0.1, "jer": 0.1}},
                        "total_samples": 2,
                        "outlier_count": 0,
                        "outlier_policy": {"metric": "cer", "threshold": 1.0},
                        "evaluated_samples": 2,
                        "dataset_total_samples": 2,
                        "is_full_evaluation": True,
                    }
                ]
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                "scripts/generate_leaderboard.py",
                "--results_dir",
                str(root / "empty-results"),
                "--markdown_path",
                str(root / "leaderboard.md"),
                "--data_path",
                str(root / "leaderboard_data.json"),
                "--submitted_rows_path",
                str(submitted_path),
            ],
            check=True,
        )

        data = json.loads((root / "leaderboard_data.json").read_text(encoding="utf-8"))
        self.assertEqual(data[0]["model"], "submitted")

    def test_generated_rows_take_precedence_over_legacy_duplicates(self):
        temp_root = Path.cwd() / ".tmp_tests"
        temp_root.mkdir(exist_ok=True)
        root = temp_root / f"leaderboard-dedupe-{uuid.uuid4().hex}"
        run_dir = root / "results" / "run-2"
        run_dir.mkdir(parents=True)
        generated = {
            **public_metadata(),
            "run_id": "run-2",
            "model": "whisper_tiny",
            "model_repo": "openai/whisper-tiny",
            "dataset": "KsponSpeech",
            "subset": "clean",
            "metrics": {"macro": {"cer": 0.2, "wer": 0.3, "mer": 0.2, "jer": 0.2}},
            "total_samples": 3000,
            "outlier_count": 0,
            "outlier_policy": {"metric": "cer", "threshold": 1.0},
            "evaluated_samples": 3000,
            "dataset_total_samples": 3000,
            "is_full_evaluation": True,
        }
        submitted = dict(generated)
        submitted["run_id"] = "readme-legacy-whisper-tiny-kspon-clean"
        submitted["metrics"] = {"macro": {"cer": 0.3}}
        submitted.update(normalization_preset=None, evaluation_protocol=None,
                         metadata_status="legacy", reproducibility={}, source="README legacy table")
        (run_dir / "leaderboard_row.json").write_text(json.dumps(generated), encoding="utf-8")
        submitted_path = root / "submitted_results.json"
        submitted_path.write_text(json.dumps([submitted]), encoding="utf-8")

        subprocess.run(
            [
                sys.executable,
                "scripts/generate_leaderboard.py",
                "--results_dir",
                str(root / "results"),
                "--markdown_path",
                str(root / "leaderboard.md"),
                "--data_path",
                str(root / "leaderboard_data.json"),
                "--submitted_rows_path",
                str(submitted_path),
            ],
            check=True,
        )

        data = json.loads((root / "leaderboard_data.json").read_text(encoding="utf-8"))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["run_id"], "run-2")
        self.assertEqual(data[0]["metrics"]["macro"]["cer"], 0.2)

    def test_deduplication_keeps_distinct_protocols(self):
        base = {"model": "model", "dataset": "dataset", "subset": "clean"}
        rows = [{**base, **public_metadata(preset), "run_id": str(index)}
                for index, preset in enumerate(("kspon", "raw", "kspon"))]
        self.assertEqual([row["run_id"] for row in dedupe_rows(rows)], ["2", "1"])

    def test_sanitizes_local_paths_in_public_commands(self):
        command = (
            "/tmp/work/open-korean-automatic-speech-recognition/"
            "openkoasr/main.py --dataset_name KsponSpeech "
            "--dataset_rootpath /private/data/KsponSpeech --model_name whisper_tiny"
        )

        sanitized = sanitize_public_command(command, dataset="KsponSpeech")

        self.assertIn("python -m openkoasr.main", sanitized)
        self.assertIn("--dataset_rootpath $KSPON_ROOT", sanitized)
        self.assertNotIn("/tmp/work", sanitized)
        self.assertNotIn("/private/data", sanitized)


if __name__ == "__main__":
    unittest.main()
