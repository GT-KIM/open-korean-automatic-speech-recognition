# OpenKoASR

OpenKoASR은 한국어 자동 음성 인식(ASR) 모델을 동일한 평가 파이프라인으로 비교하기 위한 오픈소스 프로젝트입니다.

- Live leaderboard: https://gt-kim.github.io/open-korean-automatic-speech-recognition/
- 단일 실행 엔트리(`python -m openkoasr.main`) 기반 평가
- WER/CER/MER/JER/SER/RTF/Latency(지연 시간) 정량 비교
- Whisper / Qwen3-ASR 모델군 지원

## 리더보드

정적 리더보드는 `doc/leaderboard_data.json`을 데이터 소스로 사용하며, GitHub Pages에서 다음 보기를 제공합니다.

- 공개 사이트: https://gt-kim.github.io/open-korean-automatic-speech-recognition/

- `Overall`: 모델별 평균 성능 리더보드
- 데이터셋 탭: `KsponSpeech`, `AIHubLowQualityTelephone` 등 데이터셋별 전체 평가 결과

Markdown 표는 `leaderboard.md`에서 확인할 수 있습니다. README에는 결과 표를 직접 복사하지 않고, 재현 가능한 결과 아티팩트와 GitHub Pages 데이터 파일을 기준으로 공개합니다.

> 기본 리더보드는 `--limit` 없이 전체 평가 세트를 끝까지 실행한 결과만 포함합니다. 기존 수동 정리 결과는 `doc/submitted_results.json`에 `README legacy table` 출처로 보관되어 있으며, 실제 전체 평가 아티팩트로 재평가되면 교체할 예정입니다.

새 결과를 추가하거나 외부 contributor의 결과를 검토하는 절차는 `doc/leaderboard_update_workflow.md`를 참고하세요.

## 서론
국내 학계와 산업계에서 한국어 음성 인식 연구가 활발히 이루어지고 있지만, 공개 모델을 같은 조건에서 재현 가능하게 비교할 수 있는 자료는 아직 부족합니다. OpenKoASR은 Hugging Face의 [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)([GitHub](https://github.com/huggingface/open_asr_leaderboard))처럼 동일한 평가 지표와 검증 세트를 기준으로 한국어 ASR 모델을 비교하는 것을 목표로 합니다.

관련 프로젝트로는 [KoSpeech](https://github.com/sooftware/kospeech)와 [Awesome Korean Speech Recognition](https://github.com/rtzr/Awesome-Korean-Speech-Recognition) 등이 있습니다. OpenKoASR은 이들과 경쟁하기보다, 최신 공개 ASR 모델을 같은 평가 파이프라인으로 실행하고 결과 아티팩트를 남기는 데 초점을 둡니다.

OpenKoASR은
1. Whisper, Qwen3-ASR 등 공개 ASR 모델의 한국어 평가 결과 정리
2. 동일 데이터셋, 동일 정규화, 동일 지표 기반의 비교 결과 제공
3. 재현 가능한 실행 명령, 결과 아티팩트, GitHub Pages 리더보드 제공
를 목표로 합니다.

## 프로젝트 목표

1. 공개 ASR 모델의 한국어 평가 결과 정리
2. 동일 지표와 동일 데이터셋 기반의 비교 벤치마크 제공
3. 재현 가능한 평가 코드, 결과 아티팩트, 문서 제공

## 빠른 시작

### 설치

```bash
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 기본 실행

```bash
python -m openkoasr.main --dataset_name KsponSpeech --model_name whisper_tiny
```

평가 결과는 기본적으로 `results/<run_id>/` 아래에 저장됩니다.

Qwen3-ASR 예시:

```bash
python -m openkoasr.main --dataset_name KsponSpeech --model_name qwen3_asr_0_6b
```

GPU나 실제 데이터셋 없이 평가 루프를 확인하는 스모크 테스트입니다.

```bash
python -m openkoasr.main --dataset_name mock --model_name mock --limit 2 --save_predictions
```

스크립트 실행:

```bash
bash scripts/run.sh
```

## CLI 인자

- `--dataset_name` (기본값: `KsponSpeech`)
- `--model_name` (기본값: `whisper_tiny`)
- `--outlier_metric` (기본값: `cer`)
- `--outlier_threshold` (기본값: `1.0`)
- `--output_dir` (기본값: `results`)
- `--limit` (기본값: 전체 샘플)
- `--batch_size` (기본값: `1`)
- `--num_workers` (기본값: `0`)
- `--warmup_samples` (기본값: `0`)
- `--log_interval` (기본값: `1`)
- `--normalization_preset` (`raw`, `strict`, `punctuation_agnostic`, `kspon`)
- `--save_predictions` (예측 CSV와 오류 분석 저장)
- `--manifest_path` (`--dataset_name manifest` 사용 시 입력)
- `--dataset_rootpath` (데이터셋 루트 경로 덮어쓰기)
- `--dataset_subset` (예: KsponSpeech `clean`, `other` 덮어쓰기)

## 결과 아티팩트

각 실행은 `results/<run_id>/`에 다음 파일을 생성합니다.

- `summary.json`: 실행 메타데이터, 환경, 평균 지표, 모델 지표
- `samples.jsonl`: 샘플별 지표
- `outliers.jsonl`: 이상치로 제외된 샘플
- `leaderboard_row.json`: 리더보드 생성용 요약 행
- `predictions.csv`, `error_analysis.jsonl`: `--save_predictions` 사용 시 생성

저장된 결과를 모아 리더보드 파일을 생성할 수 있습니다. 기본값은 `--limit` 없이 전체 평가 세트를 끝까지 실행한 결과만 포함합니다.

```bash
python scripts/generate_leaderboard.py --results_dir results --markdown_path leaderboard.md
```

개발용 스모크 테스트나 부분 실행 결과를 확인할 때만 `--include_partial`을 사용하세요.

GitHub Pages용 정적 리더보드는 같은 `doc/leaderboard_data.json`을 데이터 소스로 사용합니다.
사이트는 모델별 평균 성능을 보여주는 `Overall` 탭과 데이터셋별 전체 평가 탭을 함께 제공합니다.

```bash
python scripts/build_pages.py --output_dir _site
```

`main` 또는 `master` 브랜치에 push하면 `.github/workflows/pages.yml`이 `_site/`를 빌드해 GitHub Pages로 배포합니다. 저장소 Settings에서 Pages 소스를 GitHub Actions로 설정하세요.

## 지원 범위

### 데이터셋

- `KsponSpeech` (`openkoasr/configs/dataset/KsponSpeech.py`)
  - `subset`: `clean`, `other`
  - `rootpath` 기본값: `/app/data/KsponSpeech/`
- `manifest`
  - `.jsonl` 또는 `.csv` manifest 기반 로더
  - 각 row는 `audio_path`와 `text`를 포함해야 함
- `AIHubLowQualityTelephone`
  - AI-Hub `007.저음질 전화망 음성인식 데이터` 검증용 ZIP을 직접 읽는 로더
  - `subset`: `D01`, `D02`, `D03`, `D04`, `all`
- `mock`
  - CPU 스모크 테스트 및 CI용 작은 데이터셋

### 모델

- 고정 설정
  - Whisper: `whisper_tiny`, `whisper_base`, `whisper_small`, `whisper_medium`
  - Qwen3-ASR: `qwen3_asr_0_6b`, `qwen3_asr_1_7b`
- 동적 모델명 추론
  - `whisper-large-v3` → `openai/whisper-large-v3`
  - `qwen3-asr-0.6b` → `Qwen/Qwen3-ASR-0.6B`

## 평가 지표

- 인식 품질: WER, CER, MER, JER, SER
- 성능: RTF, 지연 시간
- 모델 규모/연산량: Params, FLOPS (백엔드에 따라 계산 제한 가능)

상세 내용은 `doc/metrics.md` 참고.

## 데이터 준비

KsponSpeech 데이터/정규화 가이드는 아래 문서를 참고하세요.

- `doc/data/KsponSpeech.md`
- `doc/README.md`
- `doc/data_policy.md`
- `doc/result_submission.md`

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

## Docker 실행 (선택)

```bash
bash docker/build.sh
bash docker/run.sh
```

`docker/run.sh`의 `DATASET_PATH` 값은 사용자 환경에 맞게 변경해서 사용하세요.

## 향후 작업

- [ ] README의 legacy 행을 실제 전체 평가 아티팩트로 교체
- [ ] KsponSpeech other 및 AIHub D01/D02/D04/all 전체 평가
- [ ] Overall 리더보드에 CER/RTFx 파레토 차트 추가

## 기여

Issue와 PR 모두 환영합니다. 결과 제출 전에는 `doc/result_submission.md`와 `doc/data_policy.md`를 확인해 주세요.

## 참고

- https://huggingface.co/spaces/hf-audio/open_asr_leaderboard
