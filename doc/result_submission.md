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
