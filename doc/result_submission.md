# 결과 제출 가이드

OpenKoASR 공개 리더보드는 전체 평가 실행 결과만 받습니다. 스모크 테스트, 디버깅 실행, `--limit`을 사용한 부분 실행은 개발에는 유용하지만 순위 결과로 제출하지 않습니다.

리더보드가 GitHub Pages에 반영되는 전체 흐름은 `doc/leaderboard_update_workflow.md`를 함께 참고하세요.

## 필수 기준

- `--limit` 없이 실행합니다.
- 문서화된 데이터셋 split/subset을 사용합니다.
- 정확한 모델 이름 또는 repository id를 기록합니다.
- 하드웨어와 소프트웨어 메타데이터를 포함합니다.
- 로컬 경로가 제거된 공개 재현 명령을 제공합니다.
- normalization preset과 outlier 정책을 명시합니다.
- `normalization_preset`, `outlier_policy`, `evaluation_protocol`, `metadata_status`, `reproducibility`를 기록하고, 샘플 수·유효 샘플 수·이상치 수가 일치하는지 확인합니다.
- 제한된 오디오, 원본 전사, 비공개 샘플별 예측 아티팩트를 첨부하지 않았는지 확인합니다.

## 권장 실행 명령

```bash
python -m openkoasr.main \
  --dataset_name KsponSpeech \
  --dataset_rootpath $KSPON_ROOT \
  --dataset_subset clean \
  --model_name whisper_tiny \
  --normalization_preset kspon \
  --output_dir results/full \
  --save_predictions \
  --log_interval 500
```

공개 제출 전에는 데이터셋 약관상 재배포가 허용되는 경우가 아니라면 `predictions.csv`와 `error_analysis.jsonl`을 공유하지 마세요.

## 검증 수준

평가 규약 ID `v1/{preset}/{metric}>{threshold}`는 정규화·outlier 비교 규칙을 식별합니다. 표준 규약은 `v1/kspon/cer>1.0`이며, Overall의 세 구간은 모두 이 규약이어야 합니다. 다른 규약의 결과는 별도 행으로 보존하고 데이터셋 화면의 접힌 ‘참고 결과’ 영역에 순위 없이 표시합니다. ID가 같아도 모델 가중치·정규화 구현·패키지 버전까지 같다는 뜻은 아닙니다.

새 평가기는 다음 정보를 `summary.json`과 `leaderboard_row.json`에 함께 기록합니다.

- `reproducibility.model_revision`: 로딩된 모델 config의 실제 commit hash. API나 확인 불가능한 모델은 `null`입니다.
- `reproducibility.code`: Git commit, `openkoasr` 코드 변경 여부, 패키지 내 Python 소스 SHA-256. Git이 없어도 소스 해시는 기록합니다.
- `model_config`, `generation_config`: 허용된 요청 설정과 로딩된 생성 기본값. 인증 정보·로컬 데이터 경로는 수집하지 않습니다.
- `processor_revision`, `processor_revision_source`: processor/tokenizer에 지정한 commit과 그 근거. 로더가 실제 hash를 제공하면 `loaded_metadata`, 제공하지 않으면 불변 revision 인자를 사용한 `pinned_loader_argument`로 구분합니다. 모델은 요청 commit과 로딩된 config hash가 다르거나 확인되지 않으면 중단합니다. processor가 다른 hash를 보고해도 중단합니다.
- `inference`: 실제 파라미터를 순회한 dtype별 원소 수와 `effective_dtype`, backend, 내부 batch 한도, 측정 범위. 혼합 dtype은 `mixed:...`, 관측 불가는 `null`이며 요청 dtype으로 대신 채우지 않습니다. activation dtype 전체를 관측한 값은 아닙니다.
- `inference.decoding`: 어댑터가 실제 호출에 전달한 인자와 로딩된 생성 기본값의 병합. `do_sample=false`일 때 temperature/top-k/top-p는 비활성 항목으로 분리합니다. backend 내부의 모든 자동 분기를 추적하는 로그는 아니며, Qwen은 고정한 qwen-asr Transformers 경로의 max_new_tokens 호출을 기록합니다.
- `execution`: batch size, worker 수, warmup 수, limit. 공개 전체 평가는 `limit: null`이어야 합니다.
- `environment`: Python·하드웨어·주요 패키지 버전.

Whisper·Qwen·HF CTC는 `--model_revision <40자리 commit>`과 선택적인 `--processor_revision <40자리 commit>`으로 버전을 고정할 수 있습니다. processor 인자를 생략하면 model commit을 사용합니다. 가변 branch/tag는 이 옵션으로 받지 않습니다. 옵션 없는 기존 실행은 계속 가능하지만 processor revision을 확인한 것으로 취급하지 않습니다. `--dtype`과 Qwen의 `--max_inference_batch_size`도 실행별로 지정할 수 있으며 전역 config는 변경하지 않습니다.

로컬 모델의 `asr_transcribe_call`은 feature 추출·기기 전송·추론·텍스트 디코딩을 포함하고, 파일 읽기·모델 로딩·warmup·채점을 제외합니다. API의 `api_request_reported`와 온디바이스 QNN 그래프 시간은 다른 범위입니다. Warmup은 현재 평가기의 샘플 단위 경로를 쓰며 `execution.warmup_mode=single_sample`로 명시합니다.

`metadata_status`는 새 실행의 `recorded`, 원본 summary에서 복원한 `recovered`, 네 도메인별 출처를 보관하는 `aggregate`, 근거가 없는 기존 README 자료의 `legacy`로 구분합니다. 복원 시 run ID·모델·평가 구간·샘플 수·정책을 대조하고 원본 summary SHA-256을 남깁니다. 과거 코드 버전이나 revision을 현재 값으로 채우지 않습니다. `legacy` 예외는 이미 등록된 README 행에만 적용하며 신규 제출의 누락을 허용하지 않습니다.

아래 source label은 출처 분류입니다. 재현 정보의 완전성은 위 `metadata_status` 및 개별 필드로 확인합니다.

공개 상세 화면은 보조 지표와 평가 조건만 표시하며, 전체 메타데이터는 원본 JSON 링크로 제공합니다. 화면 표시 축소는 제출 필드나 검증 요건을 변경하지 않습니다.

- `verified`: OpenKoASR 전체 평가 아티팩트에서 생성했고, 데이터 정책 준수를 검토한 결과입니다.
- `submitted`: 과거 결과나 외부 보고를 바탕으로 `doc/submitted_results.json`에 정리한 결과입니다.
- `README legacy table`: 이전 README 표에서 옮긴 curated full-evaluation 결과입니다. 부분 실행이 아니라 전체 평가로 확인된 행만 보관합니다.

기본 리더보드는 전체 평가 결과만 포함합니다. UI는 독자가 실행 아티팩트 기반 행과 curated row의 출처를 구분할 수 있도록 source label을 함께 표시할 수 있습니다.

## Pull Request 체크리스트

1. 관련 `leaderboard_row.json` 소스 또는 `doc/submitted_results.json` 행을 추가하거나 수정합니다.
2. `python scripts/generate_leaderboard.py --results_dir results --markdown_path leaderboard.md`를 실행합니다.
3. `python scripts/build_pages.py --output_dir _site`를 실행합니다.
4. `python -m unittest discover -s tests`를 실행합니다.
5. `python scripts/public_readiness_check.py`를 실행합니다.
6. `python scripts/validate_leaderboard_data.py`를 실행합니다.
7. `doc/leaderboard_data.json`에 로컬 절대 경로나 secret이 없는지 확인합니다.

생성과 Pages 빌드도 서버 리더보드 검증을 자동으로 실행합니다. 비유한 값, 음수 지표, 중복 행 또는 샘플 수 불일치가 있으면 공개 출력을 갱신하지 않습니다.
