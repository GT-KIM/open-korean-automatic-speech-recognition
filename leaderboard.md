# OpenKoASR Leaderboard

These tables include only full evaluation runs generated from `results/**/leaderboard_row.json` or curated in `doc/submitted_results.json`.

[Live leaderboard](https://gt-kim.github.io/open-korean-automatic-speech-recognition/) | [Evaluation method](https://gt-kim.github.io/open-korean-automatic-speech-recognition/#evaluation-method) | [Leaderboard JSON](https://gt-kim.github.io/open-korean-automatic-speech-recognition/leaderboard_data.json) | [Submit a result](https://github.com/GT-KIM/open-korean-automatic-speech-recognition/issues/new?template=result_submission.md)

Error rates are shown in %, Macro RTFx in ×, and latency in ms. JSON retains ratios and seconds.

Macro RTFx is the mean of per-sample audio duration / processing time after outlier exclusion, not total audio duration / total processing time.

## Standard results

Kspon normalization · CER > 100% excluded.

### AIHubLowQualityTelephone · D01

| Model | Main CER | Outlier rate | All-sample CER | WER | MER | JER | SER | Macro RTFx | Latency | GPU | Run |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 9.39% | 0.37% (32 / 8664) | 9.05% | 25.47% | 13.02% | 7.14% | 74.27% | 33.69× | 152.3 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T163854554748Z |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 9.61% | 0.35% (30 / 8664) | 9.28% | 25.89% | 13.66% | 7.31% | 75.40% | 80.70× | 61.8 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T182125441958Z |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 16.71% | 0.33% (29 / 8664) | 14.56% | 40.23% | 24.26% | 12.60% | 90.92% | 26.99× | 200.1 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T222751453628Z |
| [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 19.79% | 0.50% (43 / 8664) | 17.58% | 45.73% | 28.99% | 15.08% | 92.40% | 28.40× | 191.1 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T194632367840Z |
| [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) | 27.89% | 0.87% (75 / 8664) | 28.38% | 51.84% | 33.77% | 22.87% | 95.04% | 51.64× | 101.8 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T145929950532Z |
| [openai/whisper-small](https://huggingface.co/openai/whisper-small) | 29.58% | 1.02% (88 / 8664) | 29.37% | 55.43% | 37.65% | 24.08% | 96.18% | 102.35× | 52.3 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T135922924753Z |
| [google_speech_recognition](https://pypi.org/project/SpeechRecognition/) | 30.48% | 0.18% (16 / 8664) | 28.53% | 57.08% | 36.02% | 25.66% | 98.17% | 4.14× | 1,243.9 ms |  | 20260611T112044282099Z |
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | 33.42% | 0.25% (22 / 8664) | 27.04% | 54.08% | 40.17% | 29.23% | 94.60% | 160.29× | 32.2 ms | NVIDIA GeForce RTX 3090 Ti | 20261010T050137920566Z |
| [openai/whisper-base](https://huggingface.co/openai/whisper-base) | 38.91% | 2.19% (190 / 8664) | 43.92% | 68.59% | 51.16% | 31.98% | 98.62% | 174.56× | 32.4 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T132225280646Z |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | 49.60% | 4.27% (370 / 8664) | 64.34% | 78.77% | 64.50% | 41.99% | 99.47% | 219.78× | 27.8 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T125219542222Z |

### AIHubLowQualityTelephone · D02

| Model | Main CER | Outlier rate | All-sample CER | WER | MER | JER | SER | Macro RTFx | Latency | GPU | Run |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 12.48% | 0.41% (62 / 15211) | 11.68% | 31.98% | 18.46% | 9.73% | 78.90% | 34.41× | 159.4 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T172057416308Z |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 12.70% | 0.38% (58 / 15211) | 11.92% | 32.49% | 18.99% | 9.89% | 80.00% | 84.79× | 62.8 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T183852745255Z |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 25.07% | 0.59% (90 / 15211) | 21.20% | 52.07% | 36.39% | 19.81% | 93.10% | 27.08× | 215.4 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T232411775431Z |
| [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 30.67% | 0.87% (132 / 15211) | 25.86% | 60.74% | 44.38% | 24.23% | 96.27% | 27.84× | 210.7 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T204140880955Z |
| [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) | 33.88% | 1.48% (225 / 15211) | 33.72% | 62.00% | 43.85% | 28.05% | 96.59% | 50.60× | 112.7 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T152956012783Z |
| [openai/whisper-small](https://huggingface.co/openai/whisper-small) | 39.03% | 2.35% (357 / 15211) | 39.82% | 68.56% | 51.56% | 32.42% | 97.85% | 99.82× | 58.9 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T141607815618Z |
| [openai/whisper-base](https://huggingface.co/openai/whisper-base) | 48.85% | 4.26% (648 / 15211) | 59.20% | 81.07% | 65.62% | 40.86% | 99.24% | 168.33× | 37.3 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T133354302307Z |
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | 51.20% | 0.16% (24 / 15211) | 41.52% | 71.86% | 59.15% | 46.62% | 98.16% | 169.08× | 33.1 ms | NVIDIA GeForce RTX 3090 Ti | 20261010T051116818126Z |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | 59.48% | 6.94% (1056 / 15211) | 81.85% | 90.46% | 77.85% | 50.90% | 99.73% | 214.03× | 31.7 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T130229382034Z |

### AIHubLowQualityTelephone · D03

| Model | Main CER | Outlier rate | All-sample CER | WER | MER | JER | SER | Macro RTFx | Latency | GPU | Run |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 9.00% | 0.10% (2 / 1936) | 8.05% | 23.74% | 11.16% | 7.39% | 76.16% | 33.22× | 200.9 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T172806179213Z |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 9.33% | 0.10% (2 / 1936) | 8.38% | 24.35% | 11.62% | 7.72% | 75.85% | 93.21× | 69.9 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T184143488436Z |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 16.55% | 0.05% (1 / 1936) | 13.81% | 39.67% | 22.14% | 13.20% | 90.80% | 24.85× | 276.8 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T233350350771Z |
| [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 20.10% | 0.15% (3 / 1936) | 17.20% | 46.10% | 27.99% | 16.19% | 92.91% | 25.51× | 270.7 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T205106763347Z |
| [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) | 25.18% | 0.57% (11 / 1936) | 23.76% | 52.15% | 32.54% | 20.55% | 96.99% | 48.52× | 139.4 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T153504145221Z |
| [openai/whisper-small](https://huggingface.co/openai/whisper-small) | 30.92% | 1.08% (21 / 1936) | 29.63% | 59.14% | 41.31% | 25.62% | 96.87% | 94.41× | 72.9 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T141905939595Z |
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | 40.12% | 0.00% (0 / 1936) | 30.90% | 64.86% | 46.82% | 35.35% | 97.11% | 156.95× | 42.5 ms | NVIDIA GeForce RTX 3090 Ti | 20261010T051302521970Z |
| [openai/whisper-base](https://huggingface.co/openai/whisper-base) | 46.35% | 4.08% (79 / 1936) | 54.20% | 78.48% | 62.77% | 38.78% | 99.78% | 155.68× | 47.6 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T133606680588Z |
| [google_speech_recognition](https://pypi.org/project/SpeechRecognition/) | 55.33% | 0.00% (0 / 1936) | 58.35% | 74.09% | 60.24% | 52.33% | 99.64% | 4.45× | 1,566 ms |  | 20260611T064319962916Z |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | 61.03% | 6.51% (126 / 1936) | 75.79% | 90.36% | 78.54% | 53.02% | 99.83% | 202.40× | 38.9 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T130425831103Z |

### AIHubLowQualityTelephone · D04

| Model | Main CER | Outlier rate | All-sample CER | WER | MER | JER | SER | Macro RTFx | Latency | GPU | Run |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 9.62% | 0.37% (52 / 14105) | 9.03% | 26.99% | 13.51% | 7.34% | 75.19% | 31.84× | 148.6 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T180421289678Z |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 9.84% | 0.37% (52 / 14105) | 9.47% | 27.23% | 13.80% | 7.55% | 75.76% | 74.45× | 61.5 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T185727806725Z |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 16.78% | 0.38% (54 / 14105) | 14.61% | 39.34% | 23.18% | 13.09% | 87.22% | 25.43× | 198 ms | NVIDIA GeForce RTX 3090 Ti | 20261009T002146998623Z |
| [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 21.10% | 0.41% (58 / 14105) | 18.26% | 47.79% | 29.66% | 16.37% | 92.28% | 27.08× | 186.6 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T213621171051Z |
| [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) | 28.02% | 0.96% (135 / 14105) | 28.58% | 53.36% | 34.23% | 22.95% | 94.72% | 48.10× | 101.6 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T160026149064Z |
| [openai/whisper-small](https://huggingface.co/openai/whisper-small) | 30.51% | 1.08% (153 / 14105) | 29.62% | 57.25% | 38.50% | 24.88% | 96.22% | 97.09× | 50.9 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T143224451403Z |
| [google_speech_recognition](https://pypi.org/project/SpeechRecognition/) | 32.24% | 0.13% (19 / 14105) | 29.84% | 61.73% | 38.47% | 26.79% | 99.43% | 4.18× | 1,133 ms |  | 20260611T163019846455Z |
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | 38.39% | 0.21% (30 / 14105) | 30.03% | 60.56% | 45.31% | 33.83% | 96.57% | 157.38× | 30.3 ms | NVIDIA GeForce RTX 3090 Ti | 20261010T052112975339Z |
| [openai/whisper-base](https://huggingface.co/openai/whisper-base) | 40.15% | 1.87% (264 / 14105) | 43.52% | 71.06% | 51.96% | 32.98% | 98.67% | 165.40× | 31.2 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T134453184449Z |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | 50.58% | 4.89% (690 / 14105) | 67.65% | 80.82% | 64.60% | 42.62% | 99.25% | 203.51× | 28.2 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T131246943166Z |

### AIHubLowQualityTelephone · all

| Model | Main CER | Outlier rate | All-sample CER | WER | MER | JER | SER | Macro RTFx | Latency | GPU | Run |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 10.63% | 0.37% (148 / 39916) | 10.00% | 28.40% | 15.17% | 8.21% | 76.45% | 33.29× | 156 ms | NVIDIA GeForce RTX 3090 Ti | aggregate-aihub-all-openai-whisper-large-v3-20261009T002230Z |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 10.85% | 0.36% (142 / 39916) | 10.31% | 28.80% | 15.64% | 8.40% | 77.30% | 80.66× | 62.5 ms | NVIDIA GeForce RTX 3090 Ti | aggregate-aihub-all-openai-whisper-large-v3-turbo-20261009T002232Z |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 19.91% | 0.44% (174 / 39916) | 17.11% | 44.39% | 28.39% | 15.55% | 90.43% | 26.36× | 208.9 ms | NVIDIA GeForce RTX 3090 Ti | aggregate-aihub-all-Qwen-Qwen3-ASR-1.7B-20261009T002235Z |
| [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 24.40% | 0.59% (236 / 39916) | 21.00% | 52.18% | 35.02% | 19.07% | 93.85% | 27.58× | 200.8 ms | NVIDIA GeForce RTX 3090 Ti | aggregate-aihub-all-Qwen-Qwen3-ASR-0.6B-20261009T002233Z |
| [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) | 30.08% | 1.12% (446 / 39916) | 30.22% | 56.25% | 37.70% | 24.75% | 95.61% | 49.84× | 107.7 ms | NVIDIA GeForce RTX 3090 Ti | aggregate-aihub-all-openai-whisper-medium-20261009T002229Z |
| [openai/whisper-small](https://huggingface.co/openai/whisper-small) | 33.55% | 1.55% (619 / 39916) | 33.54% | 61.22% | 43.39% | 27.59% | 96.86% | 99.14× | 55.3 ms | NVIDIA GeForce RTX 3090 Ti | aggregate-aihub-all-openai-whisper-small-20261009T002227Z |
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | 42.28% | 0.19% (76 / 39916) | 33.92% | 63.67% | 49.54% | 37.78% | 96.77% | 162.45× | 32.4 ms | NVIDIA GeForce RTX 3090 Ti | aggregate-aihub-all-nvidia-nemotron-3.5-asr-streaming-0.6b-20261010T054903Z |
| [openai/whisper-base](https://huggingface.co/openai/whisper-base) | 43.45% | 2.96% (1181 / 39916) | 50.42% | 74.64% | 57.44% | 36.00% | 98.93% | 168.04× | 34.6 ms | NVIDIA GeForce RTX 3090 Ti | aggregate-aihub-all-openai-whisper-base-20261009T002226Z |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | 54.21% | 5.62% (2242 / 39916) | 73.01% | 84.45% | 70.23% | 46.09% | 99.51% | 210.99× | 29.9 ms | NVIDIA GeForce RTX 3090 Ti | aggregate-aihub-all-openai-whisper-tiny-20261009T002224Z |

### KsponSpeech · clean

| Model | Main CER | Outlier rate | All-sample CER | WER | MER | JER | SER | Macro RTFx | Latency | GPU | Run |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 16.26% | 0.27% (8 / 3000) | 12.01% | 39.85% | 24.87% | 10.98% | 82.79% | 18.99× | 173.4 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T214611129760Z |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 18.00% | 0.60% (18 / 3000) | 15.01% | 39.41% | 22.82% | 13.15% | 79.91% | 24.96× | 124.7 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T160746007949Z |
| [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 18.58% | 0.40% (12 / 3000) | 13.99% | 43.92% | 28.35% | 12.80% | 85.68% | 19.84× | 166.5 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T190639451672Z |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 19.53% | 0.63% (19 / 3000) | 16.44% | 41.44% | 25.52% | 14.27% | 82.42% | 54.56× | 56.7 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T180757701105Z |
| [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) | 21.01% | 1.13% (34 / 3000) | 18.04% | 44.69% | 26.78% | 15.09% | 85.50% | 35.70× | 88.9 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T143753003298Z |
| [openai/whisper-small](https://huggingface.co/openai/whisper-small) | 23.22% | 1.63% (49 / 3000) | 20.86% | 47.90% | 30.36% | 17.12% | 87.22% | 74.18× | 43.6 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T134741121713Z |
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | 28.57% | 0.77% (23 / 3000) | 19.42% | 50.04% | 36.21% | 23.65% | 87.24% | 116.13× | 27 ms | NVIDIA GeForce RTX 3090 Ti | 20261010T045408662355Z |
| [openai/whisper-base](https://huggingface.co/openai/whisper-base) | 30.40% | 1.77% (53 / 3000) | 27.73% | 56.23% | 40.48% | 22.94% | 92.06% | 128.76× | 25.3 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T131433549498Z |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | 37.27% | 3.10% (93 / 3000) | 38.44% | 64.12% | 50.20% | 28.94% | 93.81% | 165.12× | 20.5 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T124521558407Z |
| [google_speech_recognition](https://pypi.org/project/SpeechRecognition/) | 38.16% | 0.03% (1 / 3000) | 27.14% | 57.25% | 44.10% | 33.85% | 95.37% | 2.58× | 1,122 ms |  | 20260610T021005900995Z |

### KsponSpeech · other

| Model | Main CER | Outlier rate | All-sample CER | WER | MER | JER | SER | Macro RTFx | Latency | GPU | Run |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- |
| [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) | 14.26% | 0.20% (6 / 3000) | 11.66% | 39.89% | 22.04% | 9.49% | 90.85% | 21.05× | 221.6 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T215752178738Z |
| [Qwen/Qwen3-ASR-0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | 16.32% | 0.17% (5 / 3000) | 13.60% | 42.96% | 25.33% | 11.10% | 90.95% | 22.03× | 212.3 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T191751866755Z |
| [openai/whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 16.46% | 0.17% (5 / 3000) | 15.21% | 39.50% | 21.76% | 12.14% | 88.05% | 29.77× | 151.5 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T161553648588Z |
| [openai/whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) | 17.13% | 0.27% (8 / 3000) | 15.73% | 40.74% | 22.92% | 12.55% | 88.00% | 72.24× | 62 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T181133199462Z |
| [openai/whisper-medium](https://huggingface.co/openai/whisper-medium) | 19.08% | 0.43% (13 / 3000) | 17.87% | 43.78% | 25.58% | 14.03% | 90.49% | 42.59× | 106.8 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T144344972239Z |
| [openai/whisper-small](https://huggingface.co/openai/whisper-small) | 21.35% | 0.43% (13 / 3000) | 19.21% | 47.74% | 29.72% | 15.77% | 92.67% | 85.87× | 53.1 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T135049881299Z |
| [nvidia/nemotron-3.5-asr-streaming-0.6b](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | 23.22% | 0.30% (9 / 3000) | 17.55% | 48.26% | 32.02% | 17.98% | 91.88% | 130.74× | 34.6 ms | NVIDIA GeForce RTX 3090 Ti | 20261010T045612281531Z |
| [openai/whisper-base](https://huggingface.co/openai/whisper-base) | 27.72% | 0.80% (24 / 3000) | 26.67% | 56.47% | 39.97% | 20.69% | 95.26% | 145.03× | 32.2 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T131639934999Z |
| [google_speech_recognition](https://pypi.org/project/SpeechRecognition/) | 31.40% | 0.00% (0 / 3000) | 25.80% | 54.07% | 37.99% | 27.06% | 96.83% | 3.74× | 1,115.6 ms |  | 20260611T233041686594Z |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | 35.03% | 1.60% (48 / 3000) | 35.06% | 65.64% | 49.92% | 26.89% | 97.15% | 188.69× | 25.4 ms | NVIDIA GeForce RTX 3090 Ti | 20261008T124707819833Z |

## Reference results

Different or unverified evaluation protocols; excluded from standard rankings.

### KsponSpeech · clean · Unverified protocol

| Model | Main CER | Outlier rate | All-sample CER | WER | MER | JER | SER | Macro RTFx | Latency | GPU | Run |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- |
| [whisper-large-v3](https://huggingface.co/openai/whisper-large-v3) | 23.12% | 0.77% (23 / 3000) | N/A | 29.52% | 18.35% | 16.04% |  | 5.75× | 395.9 ms | RTX3090ti | readme-legacy-whisper-large-v3-kspon-clean |

### KsponSpeech · clean · v1/punctuation_agnostic/cer>1.0

| Model | Main CER | Outlier rate | All-sample CER | WER | MER | JER | SER | Macro RTFx | Latency | GPU | Run |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- | :-- |
| [whisper_base](https://huggingface.co/openai/whisper-base) | 26.85% | 1.70% (51 / 3000) | 31.92% | 46.29% | 36.83% | 20.98% | 82.84% | 27.68× | 118.5 ms | NVIDIA GeForce RTX 3090 Ti | 20260503T025048555494Z |
| [whisper_tiny](https://huggingface.co/openai/whisper-tiny) | 34.65% | 3.13% (94 / 3000) | 56.08% | 56.17% | 47.44% | 27.28% | 88.33% | 37.46× | 80.6 ms | NVIDIA GeForce RTX 3090 Ti | 20260503T023237400589Z |
