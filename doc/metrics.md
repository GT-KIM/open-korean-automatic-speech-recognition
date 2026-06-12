# OpenKoASR Metrics

본 문서는 OpenKoASR에서 사용하는 평가 지표의 의미와 해석 기준을 간단히 정리합니다.

## 1. 인식 품질 지표

OpenKoASR의 기본 품질 지표는 아래와 같습니다.

- **WER (Word Error Rate)**: 단어 단위 오류율
- **CER (Character Error Rate)**: 글자 단위 오류율
- **MER (Morpheme Error Rate)**: 형태소 단위 오류율
- **JER (Jamo Error Rate)**: 자모 단위 오류율
- **SER (Sentence Error Rate)**: 문장 단위 오류율

값이 낮을수록 성능이 좋습니다.

## 2. 성능 지표

- **RTFx (Real-Time Factor times, 실시간 배속)**
  - 계산: `총 오디오 길이 / 총 처리 시간`
  - **값이 클수록 빠릅니다.** 예: RTFx 20이면 실시간 대비 20배 빠른 처리에 해당
  - 샘플별로 계산한 뒤 outlier를 제외하고 macro 평균을 냅니다.
  - HuggingFace [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)와 동일한 정의입니다.
  - 참고: 전통적인 RTF(`처리 시간 / 오디오 길이`, 값이 작을수록 빠름)의 역수입니다.
- **Latency**
  - 샘플 1개 추론에 걸린 지연 시간(초), 값이 작을수록 좋습니다.

## 3. 모델 규모/연산량 지표

- **Params**: 모델 파라미터 수
- **FLOPS / MACS**: 모델 연산량 지표

> 참고: 모델 백엔드에 따라 FLOPS 계산이 제한될 수 있습니다.

## 4. Outlier 제외 평균

OpenKoASR는 outlier 기준을 넘는 샘플을 평균 집계에서 제외할 수 있습니다.

- 기본값
  - `--outlier_metric cer`
  - `--outlier_threshold 1.0`

즉, 기본 설정에서는 `CER > 1.0` 샘플이 평균 계산에서 제외됩니다.

## 5. 재현을 위한 기록 권장 항목

비교 실험 시 아래 항목을 함께 기록하는 것을 권장합니다.

1. `dataset_name`, `subset`
2. `model_name`
3. outlier 기준 (`outlier_metric`, `outlier_threshold`)
4. 실행 환경 (GPU, CUDA, torch 버전)
