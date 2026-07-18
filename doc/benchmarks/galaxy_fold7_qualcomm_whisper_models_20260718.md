# Galaxy Z Fold7 Qualcomm Whisper 모델 비교

2026-07-18 기준 Qualcomm AI Hub가 Snapdragon 8 Elite for Galaxy용 QNN context binary로 제공하는 Whisper-Tiny, Whisper-Base, Whisper-Small(float), Whisper-Large-V3-Turbo를 Samsung Galaxy Z Fold7(`SM-F966N`)에서 측정했다. 기존 Whisper-Small-Quantized 결과도 같은 조건의 비교 기준으로 포함했다.

## 측정 조건

- SoC: Qualcomm `SM8750`, Snapdragon 8 Elite for Galaxy, HTP v79
- OS/ABI: Android 16, `arm64-v8a`
- 런타임: QAIRT 2.47.0
- 데이터셋: KsponSpeech `eval_clean.trn` 전체 3,000건
- 디코딩: Korean, transcribe, no timestamps, greedy
- 점수 정책: OpenKoASR `kspon` 정규화 후 `CER > 1.0`을 outlier로 제외
- 성능 범위: QNN encoder/decoder graph 실행 시간만 포함. feature extraction, USB 전송, tokenization, scoring은 제외

Distil-Whisper는 Qualcomm 배포 자산에 Fold7 전용 QNN context binary가 없어 이번 비교에서 제외했다.

## 전체 결과

| 모델 | 정밀도 | 유효 / outlier | Macro WER | Micro WER | Micro CER | 평균 latency | 전체 QNN RTFx |
|---|---|---:|---:|---:|---:|---:|---:|
| Whisper-Tiny | float | 2,902 / 98 | 0.639511 | 0.565155 | 0.314965 | 0.228 s | **13.1657×** |
| Whisper-Base | float | 2,936 / 64 | 0.562470 | 0.477049 | 0.246771 | 0.344 s | **8.9674×** |
| Whisper-Small | float | 2,949 / 51 | 0.480089 | 0.406259 | 0.192908 | 0.882 s | **3.5627×** |
| Whisper-Small-Quantized | w8a16 | 2,950 / 50 | 0.479028 | 0.408752 | 0.195797 | 1.713 s | **1.8438×** |
| Whisper-Large-V3-Turbo | float | 2,982 / 18 | **0.421072** | **0.352245** | **0.156386** | 2.739 s | **1.1566×** |

Small(float)은 Small-Quantized보다 micro WER가 0.00249 낮고 전체 QNN RTFx는 약 1.93배 높았다. Large-V3-Turbo는 가장 낮은 오류율을 기록하면서 전체 오디오 기준 1.1566× 실시간으로 동작했다. Tiny와 Base는 각각 13.17×, 8.97×로 빠르지만 모델 크기가 작아질수록 오류율이 증가했다.

## 실행 시간 분해

| 모델 | Encoder 합계 | Decoder 합계 | QNN 합계 | p50 latency | p95 latency |
|---|---:|---:|---:|---:|---:|
| Whisper-Tiny | 243.574 s | 479.398 s | 722.972 s | 0.197 s | 0.437 s |
| Whisper-Base | 394.802 s | 666.651 s | 1,061.453 s | 0.302 s | 0.634 s |
| Whisper-Small | 1,118.261 s | 1,553.475 s | 2,671.735 s | 0.778 s | 1.590 s |
| Whisper-Small-Quantized | 4,115.479 s | 1,047.017 s | 5,162.496 s | 1.671 s | 2.209 s |
| Whisper-Large-V3-Turbo | 6,988.911 s | 1,240.594 s | 8,229.505 s | 2.676 s | 3.351 s |

## 검증

- native persistent runner가 decoder layer 수와 float16/float32/quantized tensor 형식을 QNN tensor metadata에서 감지하도록 확장했다.
- Tiny, Base, Small(float)의 첫 발화는 공식 `qnn-net-run` 기준과 전체 token sequence가 각각 완전히 일치했다.
- 모델별 tokenizer에서 decoder start, EOS, Korean/transcribe/no-timestamps 강제 token을 가져오며 Large-V3-Turbo의 변경된 token ID도 자동 반영했다.
- 각 결과 캐시는 1부터 3,000까지 연속 인덱스를 검증하고, 최종 리더보드 행은 원본 전체 측정 JSON과 자동 대조한다.

## 원본 결과

- [Whisper-Tiny 전체 결과](galaxy_fold7_whisper_tiny_float_kspon_clean_full_20260718.json)
- [Whisper-Base 전체 결과](galaxy_fold7_whisper_base_float_kspon_clean_full_20260718.json)
- [Whisper-Small float 전체 결과](galaxy_fold7_whisper_small_float_kspon_clean_full_20260718.json)
- [Whisper-Small-Quantized 전체 결과](galaxy_fold7_whisper_small_quantized_kspon_clean_full_20260718.json)
- [Whisper-Large-V3-Turbo 전체 결과](galaxy_fold7_whisper_large_v3_turbo_float_kspon_clean_full_20260718.json)

## 모델 출처

- [Qualcomm Whisper-Tiny](https://aihub.qualcomm.com/models/whisper_tiny)
- [Qualcomm Whisper-Base](https://aihub.qualcomm.com/models/whisper_base)
- [Qualcomm Whisper-Small](https://aihub.qualcomm.com/models/whisper_small)
- [Qualcomm Whisper-Small-Quantized](https://aihub.qualcomm.com/models/whisper_small_quantized)
- [Qualcomm Whisper-Large-V3-Turbo](https://aihub.qualcomm.com/models/whisper_large_v3_turbo)
