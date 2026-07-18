# Galaxy Fold 7 Qualcomm Whisper benchmark

2026-07-18에 연결된 Samsung Galaxy Fold 7(`SM-F966N`)에서 Qualcomm이 배포한
`Whisper-Small-Quantized v0.58.0` QNN context binary를 실행했다.

## 환경

- SoC: Qualcomm `SM8750`, Snapdragon 8 Elite for Galaxy, HTP v79
- OS/ABI: Android 16, `arm64-v8a`
- 모델: `openai/whisper-small` 기반 w8a16 Qualcomm 최적화 모델
- 모델 빌드: QAIRT 2.45.0, 실행기: QAIRT 2.47.0
- 디코딩 설정: Korean, transcribe, no timestamps

## 단일 그래프 성능

| 그래프 | 반복 | 평균 지연시간 | 최소 | 최대 | 환산 성능 |
|---|---:|---:|---:|---:|---:|
| Encoder (30초 입력) | 25 | 1,274.252 ms | 783.994 ms | 1,570.912 ms | 23.543 RTFx |
| Decoder (토큰 1개) | 100 | 16.559 ms | 14.484 ms | 25.655 ms | 60.390 token/s |

표의 지연시간은 `qnn-net-run` QNN execute 구간이다. Encoder는 일반 버퍼,
decoder는 shared buffer 결과이며 두 그래프 모두 HTP에서 HVX thread 6개를 사용했다.
Encoder 초기화 270.007ms와 decoder 초기화 68.143ms는 반복 실행 지연시간에 포함하지 않았다.

## KsponSpeech clean 전체 점수

`eval_clean.trn`의 3,000개 발화를 모두 평가했다. PCM을 Whisper mel feature로 변환하고,
단말의 지속 실행 프로세스가 QNN encoder와 autoregressive decoder를 처리했다. 토큰을 호스트로
돌려받은 뒤 OpenKoASR `kspon` 정규화와 기존 리더보드의 `CER > 1.0` outlier 정책을 적용했다.

| 항목 | 결과 |
|---|---:|
| 전체 / 유효 / outlier | 3,000 / 2,950 / 50 |
| Macro WER | **0.4790275514** |
| Macro CER | **0.2341744740** |
| Micro WER | 0.4087523731 |
| Micro CER | 0.1957969818 |
| SER | 0.8728813559 |
| 정확 일치 (전체) | 375 / 3,000 |
| 평균 QNN 지연시간 (유효 샘플) | 1.712534 s |
| Macro QNN RTFx (유효 샘플) | 1.753995 |
| 전체 QNN RTFx | 1.843771 |

전체 음성 길이는 9,518.463초, QNN encoder 실행 합계는 4,115.479초, decoder 실행 합계는
1,047.017초였다. 전체 3,000건의 QNN 실행 합계는 5,162.496초다. 특징 추출, USB 전송,
토큰화, 정규화와 채점 시간은 QNN RTFx에 포함하지 않았다.

참고로 동일 저장소의 GPU `whisper_small` clean 전체 결과는 macro WER 0.4794057348,
macro CER 0.2321068990이다. 양자화 QNN 결과와 WER은 매우 가깝고 CER은 0.00207 높았다.

전체 샘플별 오류 수와 실행 시간은
[`galaxy_fold7_whisper_small_quantized_kspon_clean_full_20260718.json`](galaxy_fold7_whisper_small_quantized_kspon_clean_full_20260718.json)에 기록했다.
데이터 정책에 따라 원문, 예측문, 토큰 ID는 저장하지 않았다. 평가 transcript SHA-256도 JSON에
포함해 입력 목록을 식별할 수 있게 했다.

## 구현 및 검증

- 지속 실행기: [`native/qnn_whisper_runner`](../../native/qnn_whisper_runner)
- 전체 평가 오케스트레이터: [`scripts/run_qnn_whisper_kspon_full.py`](../../scripts/run_qnn_whisper_kspon_full.py)
- 기준 경로: [`scripts/validate_qnn_whisper_kspon.py`](../../scripts/validate_qnn_whisper_kspon.py)
- 1번 발화 회귀: 기존 `qnn-net-run` 기준과 43개 토큰, WER, CER가 모두 일치
- 10건 연속 회귀: 10/10 결과 수집 및 샘플 간 self-cache 초기화 확인
- 전체 결과 검증: 인덱스 1–3,000 연속, transcript 해시 일치, 민감 텍스트 미저장
- 자동 테스트: 21개 통과

## 출처

- [Qualcomm Whisper-Small-Quantized model card](https://huggingface.co/qualcomm/Whisper-Small-Quantized)
- [Qualcomm AI Hub model page](https://aihub.qualcomm.com/models/whisper_small_quantized)
