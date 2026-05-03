# AIHub Pipeline Check

This note records the maintainer sanity check for `AIHubLowQualityTelephone` before publishing leaderboard rows.

## What Was Checked

- Validation label ZIPs and source-audio ZIPs are paired by domain: `VL_D01`/`VS_D01`, `VL_D02`/`VS_D02`, `VL_D03`/`VS_D03`, and `VL_D04`/`VS_D04`.
- Every `.txt` label path is mapped to the `.wav` path with the same relative path.
- Sample IDs use the dataset-relative audio stem, for example `D03/J13/S002202/0001`.
- Source WAV files are mono 8 kHz PCM telephone audio in the checked package.
- The loader decodes with `soundfile`, `librosa`, or a standard-library WAV fallback, then resamples to the configured 16 kHz target.
- The default public evaluation command uses `--normalization_preset kspon`, which removes common Kspon-style speaker/noise tags before scoring.
- Whisper configs use Korean transcription with deterministic decoding and a bounded `max_new_tokens` value to reduce repeated hallucination on short/noisy telephone utterances.
- Qwen3-ASR configs force `language="Korean"` so telephone samples are not auto-detected as another language.
- The evaluation loop now flattens DataLoader batches, so `batch_size > 1` no longer drops all but the first sample in a batch.

## Validation Counts

| Subset | Samples |
| :-- | --: |
| D01 | 8,664 |
| D02 | 15,211 |
| D03 | 1,936 |
| D04 | 14,105 |
| all | 39,916 |

## Interpretation

If AIHub telephone scores are much worse than KsponSpeech, the checked causes are dataset/domain related rather than an obvious path-matching or sample-rate pipeline bug: this corpus is telephone-bandwidth, low-quality, consultation-domain speech, while many public ASR models are stronger on cleaner broadband speech.

Public rows should still be treated as full-evaluation rows only when `evaluated_samples == dataset_total_samples` and `is_full_evaluation` is `true`.
