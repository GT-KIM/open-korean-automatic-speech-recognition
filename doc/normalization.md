# Korean ASR Normalization Presets

OpenKoASR supports explicit normalization presets through `--normalization_preset`.

- `raw`: keep text unchanged.
- `strict`: normalize Unicode and whitespace while preserving punctuation and case.
- `punctuation_agnostic`: lowercase, remove punctuation, and normalize whitespace.
- `kspon`: apply the KsponSpeech transcription rules implemented in `openkoasr.dataset.KsponSpeech.utils`.

Use `raw` or `strict` when punctuation and casing should count as recognition errors. Use `punctuation_agnostic` for leaderboard-style ASR comparison where punctuation is not the main target. Use `kspon` for KsponSpeech label preparation and reproducibility checks.
