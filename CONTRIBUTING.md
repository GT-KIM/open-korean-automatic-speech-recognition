# Contributing to OpenKoASR

Thanks for improving OpenKoASR.

## Development

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
python -m pytest
```

Use the mock smoke test when GPU or datasets are unavailable:

```bash
python -m openkoasr.main --dataset_name mock --model_name mock --limit 2 --save_predictions
```

## Adding a Model

Add a model wrapper that implements `transcribe(sample, sampling_rate)` and `metadata()`, then register it in `openkoasr.model`.

## Adding a Dataset

Prefer the manifest dataset for simple integrations. A manifest row should contain `audio_path` and `text`; optional fields are preserved as metadata.

## Submitting Results

Include the generated `summary.json`, `leaderboard_row.json`, sanitized command, hardware information, and the OpenKoASR commit. Do not attach raw audio, raw transcripts, or per-sample predictions unless the dataset terms explicitly permit public redistribution.

Read `doc/result_submission.md` and `doc/data_policy.md` before opening a result PR.
