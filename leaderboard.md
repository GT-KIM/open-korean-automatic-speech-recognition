# OpenKoASR Leaderboard

This table includes only full evaluation runs generated from `results/**/leaderboard_row.json` or curated in `doc/submitted_results.json`.

| Model | Dataset | Subset | WER | CER | MER | JER | SER | RTFx | Latency(s) | GPU | Outliers | Run |
| :-- | :-- | :-- | --: | --: | --: | --: | --: | --: | --: | :-- | --: | :-- |
| qwen3_asr_1_7b | KsponSpeech | other | 0.3999 | 0.1431 | 0.2220 | 0.0952 | 0.9085 | 0.0489 | 0.1244 | NVIDIA GeForce RTX 3090 Ti | 6 / 3000 | 20260503T151705877082Z |
| qwen3_asr_0_6b | KsponSpeech | other | 0.4295 | 0.1626 | 0.2517 | 0.1107 | 0.9105 | 0.1053 | 0.2739 | NVIDIA GeForce RTX 3090 Ti | 6 / 3000 | 20260503T125648560195Z |
| qwen3_asr_1_7b | KsponSpeech | clean | 0.3983 | 0.1628 | 0.2474 | 0.1098 | 0.8265 | 0.0537 | 0.0922 | NVIDIA GeForce RTX 3090 Ti | 8 / 3000 | 20260503T150932121213Z |
| qwen3_asr_1_7b | AIHubLowQualityTelephone | D03 | 0.3983 | 0.1664 | 0.2222 | 0.1326 | 0.9075 | 0.0255 | 0.1244 | NVIDIA GeForce RTX 3090 Ti | 1 / 1936 | 20260503T161924528431Z |
| qwen3_asr_1_7b | AIHubLowQualityTelephone | D01 | 0.4031 | 0.1673 | 0.2429 | 0.1262 | 0.9098 | 0.0263 | 0.1073 | NVIDIA GeForce RTX 3090 Ti | 29 / 8664 | 20260503T153438672098Z |
| qwen3_asr_1_7b | AIHubLowQualityTelephone | D04 | 0.3936 | 0.1680 | 0.2319 | 0.1309 | 0.8727 | 0.0394 | 0.1454 | NVIDIA GeForce RTX 3090 Ti | 55 / 14105 | 20260503T165637340335Z |
| qwen3_asr_0_6b | KsponSpeech | clean | 0.4397 | 0.1856 | 0.2838 | 0.1279 | 0.8588 | 0.1276 | 0.2253 | NVIDIA GeForce RTX 3090 Ti | 12 / 3000 | 20260503T124153468914Z |
| whisper_medium | KsponSpeech | other | 0.4380 | 0.1908 | 0.2554 | 0.1401 | 0.9056 | 0.0476 | 0.1244 | NVIDIA GeForce RTX 3090 Ti | 13 / 3000 | 20260503T104945732451Z |
| qwen3_asr_0_6b | AIHubLowQualityTelephone | D01 | 0.4562 | 0.1973 | 0.2889 | 0.1503 | 0.9238 | 0.0257 | 0.1050 | NVIDIA GeForce RTX 3090 Ti | 45 / 8664 | 20260503T134506305691Z |
| qwen3_asr_1_7b | AIHubLowQualityTelephone | all | 0.4445 | 0.1993 | 0.2843 | 0.1555 | 0.9049 | 0.0334 | 0.1338 | NVIDIA GeForce RTX 3090 Ti | 181 / 39916 | aggregate-aihub-all-qwen3-asr-1-7b-20260503T165809Z |
| qwen3_asr_0_6b | AIHubLowQualityTelephone | D03 | 0.4593 | 0.2006 | 0.2800 | 0.1619 | 0.9271 | 0.0327 | 0.1557 | NVIDIA GeForce RTX 3090 Ti | 2 / 1936 | 20260503T143033863265Z |
| whisper_medium | KsponSpeech | clean | 0.4472 | 0.2101 | 0.2682 | 0.1511 | 0.8551 | 0.0701 | 0.1236 | NVIDIA GeForce RTX 3090 Ti | 33 / 3000 | 20260503T104238810452Z |
| qwen3_asr_0_6b | AIHubLowQualityTelephone | D04 | 0.4775 | 0.2109 | 0.2961 | 0.1636 | 0.9234 | 0.0344 | 0.1277 | NVIDIA GeForce RTX 3090 Ti | 60 / 14105 | 20260503T150347041511Z |
| whisper_small | KsponSpeech | other | 0.4781 | 0.2144 | 0.2976 | 0.1587 | 0.9264 | 0.0302 | 0.0784 | NVIDIA GeForce RTX 3090 Ti | 11 / 3000 | 20260503T093806989032Z |
| whisper-large-v3 | KsponSpeech | clean | 0.2952 | 0.2312 | 0.1835 | 0.1604 |  | 0.1740 | 0.3959 | RTX3090ti | 23 / 3000 | readme-legacy-whisper-large-v3-kspon-clean |
| whisper_small | KsponSpeech | clean | 0.4794 | 0.2321 | 0.3040 | 0.1713 | 0.8739 | 0.0303 | 0.0544 | NVIDIA GeForce RTX 3090 Ti | 51 / 3000 | 20260503T093331782326Z |
| qwen3_asr_0_6b | AIHubLowQualityTelephone | all | 0.5212 | 0.2437 | 0.3496 | 0.1905 | 0.9383 | 0.0324 | 0.1298 | NVIDIA GeForce RTX 3090 Ti | 248 / 39916 | aggregate-aihub-all-qwen3-asr-0-6b-20260503T165808Z |
| qwen3_asr_1_7b | AIHubLowQualityTelephone | D02 | 0.5213 | 0.2510 | 0.3646 | 0.1982 | 0.9318 | 0.0328 | 0.1394 | NVIDIA GeForce RTX 3090 Ti | 96 / 15211 | 20260503T161427307493Z |
| whisper_medium | AIHubLowQualityTelephone | D03 | 0.5221 | 0.2522 | 0.3256 | 0.2056 | 0.9704 | 0.0310 | 0.1525 | NVIDIA GeForce RTX 3090 Ti | 10 / 1936 | 20260503T115558830261Z |
| whisper_base | KsponSpeech | other | 0.5663 | 0.2780 | 0.4011 | 0.2078 | 0.9540 | 0.0171 | 0.0446 | NVIDIA GeForce RTX 3090 Ti | 21 / 3000 | 20260503T084613049499Z |
| whisper_medium | AIHubLowQualityTelephone | D01 | 0.5193 | 0.2789 | 0.3378 | 0.2290 | 0.9499 | 0.0324 | 0.1335 | NVIDIA GeForce RTX 3090 Ti | 67 / 8664 | 20260503T111038213036Z |
| whisper_medium | AIHubLowQualityTelephone | D04 | 0.5361 | 0.2818 | 0.3437 | 0.2308 | 0.9484 | 0.0334 | 0.1253 | NVIDIA GeForce RTX 3090 Ti | 121 / 14105 | 20260503T122808569314Z |
| whisper_small | AIHubLowQualityTelephone | D01 | 0.5546 | 0.2960 | 0.3764 | 0.2411 | 0.9623 | 0.0165 | 0.0685 | NVIDIA GeForce RTX 3090 Ti | 86 / 8664 | 20260503T094920732351Z |
| whisper_medium | AIHubLowQualityTelephone | all | 0.5637 | 0.3017 | 0.3776 | 0.2483 | 0.9565 | 0.0331 | 0.1355 | NVIDIA GeForce RTX 3090 Ti | 420 / 39916 | aggregate-aihub-all-whisper-medium-20260503T165811Z |
| whisper_base | KsponSpeech | clean | 0.5633 | 0.3040 | 0.4049 | 0.2296 | 0.9206 | 0.0202 | 0.0355 | NVIDIA GeForce RTX 3090 Ti | 54 / 3000 | 20260503T084323817330Z |
| whisper_small | AIHubLowQualityTelephone | D04 | 0.5736 | 0.3056 | 0.3853 | 0.2493 | 0.9626 | 0.0178 | 0.0673 | NVIDIA GeForce RTX 3090 Ti | 139 / 14105 | 20260503T103445151160Z |
| qwen3_asr_0_6b | AIHubLowQualityTelephone | D02 | 0.6070 | 0.3063 | 0.4431 | 0.2422 | 0.9620 | 0.0345 | 0.1425 | NVIDIA GeForce RTX 3090 Ti | 141 / 15211 | 20260503T142438159660Z |
| whisper_small | AIHubLowQualityTelephone | D03 | 0.5933 | 0.3098 | 0.4146 | 0.2566 | 0.9708 | 0.0246 | 0.1207 | NVIDIA GeForce RTX 3090 Ti | 21 / 1936 | 20260503T101656776514Z |
| whisper_small | AIHubLowQualityTelephone | all | 0.6126 | 0.3358 | 0.4341 | 0.2762 | 0.9689 | 0.0181 | 0.0750 | NVIDIA GeForce RTX 3090 Ti | 594 / 39916 | aggregate-aihub-all-whisper-small-20260503T165812Z |
| whisper_medium | AIHubLowQualityTelephone | D02 | 0.6203 | 0.3396 | 0.4388 | 0.2813 | 0.9660 | 0.0334 | 0.1438 | NVIDIA GeForce RTX 3090 Ti | 222 / 15211 | 20260503T115020175977Z |
| whisper_tiny | KsponSpeech | other | 0.6570 | 0.3499 | 0.4984 | 0.2682 | 0.9712 | 0.0180 | 0.0466 | NVIDIA GeForce RTX 3090 Ti | 46 / 3000 | 20260503T075923023494Z |
| whisper_tiny | KsponSpeech | clean | 0.6417 | 0.3720 | 0.5002 | 0.2887 | 0.9378 | 0.0168 | 0.0303 | NVIDIA GeForce RTX 3090 Ti | 91 / 3000 | 20260503T075532897072Z |
| whisper_base | AIHubLowQualityTelephone | D01 | 0.6856 | 0.3884 | 0.5116 | 0.3190 | 0.9864 | 0.0131 | 0.0536 | NVIDIA GeForce RTX 3090 Ti | 189 / 8664 | 20260503T085510938250Z |
| whisper_small | AIHubLowQualityTelephone | D02 | 0.6851 | 0.3904 | 0.5157 | 0.3242 | 0.9785 | 0.0184 | 0.0801 | NVIDIA GeForce RTX 3090 Ti | 348 / 15211 | 20260503T101219688008Z |
| whisper_base | AIHubLowQualityTelephone | D04 | 0.7111 | 0.4019 | 0.5201 | 0.3303 | 0.9868 | 0.0131 | 0.0492 | NVIDIA GeForce RTX 3090 Ti | 262 / 14105 | 20260503T092947492745Z |
| whisper_base | AIHubLowQualityTelephone | all | 0.7466 | 0.4344 | 0.5744 | 0.3601 | 0.9893 | 0.0138 | 0.0563 | NVIDIA GeForce RTX 3090 Ti | 1168 / 39916 | aggregate-aihub-all-whisper-base-20260503T165810Z |
| whisper_base | AIHubLowQualityTelephone | D03 | 0.7853 | 0.4628 | 0.6263 | 0.3877 | 0.9978 | 0.0157 | 0.0784 | NVIDIA GeForce RTX 3090 Ti | 77 / 1936 | 20260503T091632517286Z |
| whisper_base | AIHubLowQualityTelephone | D02 | 0.8110 | 0.4884 | 0.6558 | 0.4087 | 0.9922 | 0.0145 | 0.0619 | NVIDIA GeForce RTX 3090 Ti | 640 / 15211 | 20260503T091313872634Z |
| whisper_tiny | AIHubLowQualityTelephone | D01 | 0.7837 | 0.4925 | 0.6404 | 0.4161 | 0.9947 | 0.0111 | 0.0453 | NVIDIA GeForce RTX 3090 Ti | 399 / 8664 | 20260503T080725775028Z |
| whisper_tiny | AIHubLowQualityTelephone | D04 | 0.8077 | 0.5046 | 0.6445 | 0.4250 | 0.9926 | 0.0127 | 0.0476 | NVIDIA GeForce RTX 3090 Ti | 706 / 14105 | 20260503T084105537832Z |
| whisper_tiny | AIHubLowQualityTelephone | all | 0.8429 | 0.5404 | 0.6998 | 0.4590 | 0.9950 | 0.0128 | 0.0522 | NVIDIA GeForce RTX 3090 Ti | 2332 / 39916 | aggregate-aihub-all-whisper-tiny-20260503T165814Z |
| whisper_tiny | AIHubLowQualityTelephone | D02 | 0.9032 | 0.5939 | 0.7760 | 0.5078 | 0.9970 | 0.0140 | 0.0598 | NVIDIA GeForce RTX 3090 Ti | 1091 / 15211 | 20260503T082519170241Z |
| whisper_tiny | AIHubLowQualityTelephone | D03 | 0.9035 | 0.6069 | 0.7863 | 0.5266 | 0.9983 | 0.0120 | 0.0590 | NVIDIA GeForce RTX 3090 Ti | 136 / 1936 | 20260503T082751480346Z |
