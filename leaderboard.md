# OpenKoASR Leaderboard

This table includes only full evaluation runs generated from `results/**/leaderboard_row.json` or curated in `doc/submitted_results.json`.

| Model | Dataset | Subset | WER | CER | MER | JER | SER | RTFx | Latency(s) | GPU | Outliers | Run |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | :-- | --: | :-- |
| qwen3_asr_1_7b | KsponSpeech | clean | 0.2736 | 0.1454 | 0.1685 | 0.1093 |  | 0.2289 | 0.5607 | RTX3090ti | 55 / 3000 | readme-legacy-qwen3-asr-1-7b-kspon-clean |
| qwen3_asr_0_6b | KsponSpeech | clean | 0.3204 | 0.1740 | 0.2162 | 0.1317 |  | 0.2460 | 0.6072 | RTX3090ti | 43 / 3000 | readme-legacy-qwen3-asr-0-6b-kspon-clean |
| whisper-large-v3 | KsponSpeech | clean | 0.2952 | 0.2312 | 0.1835 | 0.1604 |  | 0.1740 | 0.3959 | RTX3090ti | 23 / 3000 | readme-legacy-whisper-large-v3-kspon-clean |
| whisper_medium | KsponSpeech | clean | 0.3231 | 0.2557 | 0.2094 | 0.1771 |  | 0.1038 | 0.2435 | RTX3090ti | 30 / 3000 | readme-legacy-whisper-medium-kspon-clean |
| whisper_base | KsponSpeech | clean | 0.4629 | 0.2685 | 0.3683 | 0.2098 | 0.8284 | 0.0475 | 0.1185 | NVIDIA GeForce RTX 3090 Ti | 51 / 3000 | 20260503T025048555494Z |
| whisper_small | KsponSpeech | clean | 0.3608 | 0.2730 | 0.2452 | 0.1945 |  | 0.0818 | 0.1898 | RTX3090ti | 69 / 3000 | readme-legacy-whisper-small-kspon-clean |
| whisper_base | KsponSpeech | clean | 0.4589 | 0.3347 | 0.3644 | 0.2483 |  | 0.0581 | 0.1309 | RTX3090ti | 64 / 3000 | readme-legacy-whisper-base-kspon-clean |
| whisper_tiny | KsponSpeech | clean | 0.5617 | 0.3465 | 0.4744 | 0.2728 | 0.8833 | 0.0330 | 0.0806 | NVIDIA GeForce RTX 3090 Ti | 94 / 3000 | 20260503T023237400589Z |
| whisper_tiny | KsponSpeech | clean | 0.5555 | 0.3977 | 0.4675 | 0.3051 |  | 0.0592 | 0.1364 | RTX3090ti | 120 / 3000 | readme-legacy-whisper-tiny-kspon-clean |
| whisper_tiny | AIHubLowQualityTelephone | D03 | 0.9030 | 0.6059 | 0.7846 | 0.5258 | 0.9983 | 0.0208 | 0.1496 | NVIDIA GeForce RTX 3090 Ti | 143 / 1936 | 20260503T024311660383Z |
