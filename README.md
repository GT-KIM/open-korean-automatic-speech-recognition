# OpenKoASR

OpenKoASR은 한국어 자동 음성 인식(ASR) 모델을 동일한 평가 파이프라인으로 비교하기 위한 오픈소스 프로젝트입니다.

- 단일 실행 엔트리(`python -m openkoasr.main`) 기반 평가
- WER/CER/MER/JER/SER/RTF/Latency 정량 비교
- Whisper / Qwen3-ASR 모델군 지원

## 리더보드

| 모델 | 데이터셋 | GFLOPS | Params(M) | 평균 WER | 평균 CER | 평균 MER | 평균 JER | GPU | RTFx | 평균 Latency(s) | 아웃라이어 |
|:--|:--|--:|--:|--:|--:|--:|--:|:--|--:|--:|--:|
| Whisper-tiny | KsponSpeech | 55.56 | 37.18 | 0.5555 | 0.3977 | 0.4675 | 0.3051 | RTX3090ti | 0.0592 | 0.1364 | 120 / 3000 |
| Whisper-base | KsponSpeech | 142.6 | 71.83 | 0.4589 | 0.3347 | 0.3644 | 0.2483 | RTX3090ti | 0.0581 | 0.1309 | 64 / 3000 |
| Whisper-small | KsponSpeech | 619.9 | 240.58 | 0.3608 | 0.2730 | 0.2452 | 0.1945 | RTX3090ti | 0.0818 | 0.1898 | 69 / 3000 |
| Whisper-medium | KsponSpeech | 2170 | 762.32 | 0.3231 | 0.2557 | 0.2094 | 0.1771 | RTX3090ti | 0.1038 | 0.2435 | 30 / 3000 |
| Whisper-large-v3 | KsponSpeech | 4510 | 1540 | 0.2952 | 0.2312 | 0.1835 | 0.1604 | RTX3090ti | 0.1740 | 0.3959 | 23 / 3000 |
| Qwen3-ASR-0.6B | KsponSpeech | - | - | 0.3204 | 0.1740 | 0.2162 | 0.1317 | RTX3090ti | 0.7995 | 0.6072 | 43 / 3000 |

> 평균 지표는 outlier 제외 샘플 기반으로 집계합니다. 기본 outlier 기준은 `CER > 1.0` 입니다.

## 서론 
국내 학계 및 산업계에서 한국어 음성 인식에 대한 연구 및 개발이 활발히 이루어지고 있으나, 이를 정리한 문서가 없어 연구자 개개인이 직접 실험을 통해 성능을 비교해야 했습니다. OpenKoASR 프로젝트는 Huggingface의 [ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)([Github](https://github.com/huggingface/open_asr_leaderboard))에서 English ASR 모델에 대한 리더보드를 제공하고 있는 것처럼 동일한 evaluation metrics와 validation set을 활용하여 여러 한국어 음성인식 모델을 비교하고자 합니다.

기존 한국어 음성 인식 프로젝트로는 [KoSpeech](https://github.com/sooftware/kospeech)와 [Awesome Korean Speech Recognition](https://github.com/rtzr/Awesome-Korean-Speech-Recognition) 등이 있습니다. KoSpeech의 경우 LAS ~ Conformer까지의 baseline을 지원하고 있으나, Whisper 이전에 업데이트가 중단되어 Whisper 이후 최신 한국어 음성 인식 모델과 비교가 어렵습니다. Awesome Korean Speech Recognitionnn 프로젝트는 최근까지 업데이트가 진행되고 있으나, 소스코드가 제공되지 않으며 Whisper를 제외하면 모델이 공개되지 않은 상업용 음성 인식 API에 대한 실험 결과만 제공하고 있습니다.

OpenKoASR은
1. Whisper 등 Huggingface에 올라온 한국어 음성 인식 모델에 대한 실험 결과 정리
2. 상업용 한국어 음성 인식 API에 대한 실험 결과 정리
3. 공개된 모델 구조에 대한 학습 (scratch and fine-tune) 및 테스트
를 목표로 합니다.

## 프로젝트 목표

1. 공개 ASR 모델(Whisper, Qwen3-ASR 등)의 한국어 평가 결과 정리
2. 동일 지표/동일 데이터셋 기반 비교 벤치마크 제공
3. 재현 가능한 평가 코드/문서 제공

## Quick Start

### 설치

```bash
pip install -r requirements.txt
```

### 기본 실행

```bash
python -m openkoasr.main --dataset_name KsponSpeech --model_name whisper_tiny
```

Qwen3-ASR 예시:

```bash
python -m openkoasr.main --dataset_name KsponSpeech --model_name qwen3_asr_0_6b
```

스크립트 실행:

```bash
bash scripts/run.sh
```

## CLI 인자

- `--dataset_name` (default: `KsponSpeech`)
- `--model_name` (default: `whisper_tiny`)
- `--outlier_metric` (default: `cer`)
- `--outlier_threshold` (default: `1.0`)

## 지원 범위

### 데이터셋

- `KsponSpeech` (`openkoasr/configs/dataset/KsponSpeech.py`)
  - `subset`: `clean`, `other`
  - `rootpath` 기본값: `/app/data/KsponSpeech/`

### 모델

- 고정 config
  - Whisper: `whisper_tiny`, `whisper_base`, `whisper_small`, `whisper_medium`
  - Qwen3-ASR: `qwen3_asr_0_6b`, `qwen3_asr_1_7b`
- 동적 모델명 추론 지원
  - `whisper-large-v3` → `openai/whisper-large-v3`
  - `qwen3-asr-0.6b` → `Qwen/Qwen3-ASR-0.6B`

## 평가 지표

- 인식 품질: WER, CER, MER, JER, SER
- 성능: RTF, Latency
- 모델 규모/연산량: Params, FLOPS (백엔드에 따라 계산 제한 가능)

상세 내용은 `doc/metrics.md` 참고.

## 데이터 준비

KsponSpeech 데이터/정규화 가이드는 아래 문서를 참고하세요.

- `doc/data/KsponSpeech.md`
- `doc/README.md`

분할 압축 병합/해제 스크립트:

```bash
bash scripts/extract_KsponSpeech_data.sh
```

## 저장소 구조

```text
openkoasr/
  configs/     # 데이터셋/모델 설정 및 모델명 추론
  dataset/     # 데이터 로더 및 정규화
  model/       # 모델별 추론 래퍼
  metrics/     # 오류율/성능 지표 계산
  utils/       # 로깅 등 유틸
doc/
  data/        # 데이터셋 문서
scripts/       # 실행/데이터 준비 스크립트
docker/        # 도커 환경 스크립트
```

## Docker (선택)

```bash
bash docker/build.sh
bash docker/run.sh
```

`docker/run.sh`의 `DATASET_PATH` 값은 사용자 환경에 맞게 변경해서 사용하세요.

## Future Works

- [ ] 데이터셋 확장
- [ ] 모델 확장
- [ ] 리더보드 자동 업데이트

## Contribution

Issue / PR 모두 환영합니다.

## References

- https://huggingface.co/spaces/hf-audio/open_asr_leaderboard