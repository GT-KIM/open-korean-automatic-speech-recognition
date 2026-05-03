# 리더보드 아티팩트

평가 결과는 `results/<run_id>/` 아래에 저장됩니다.

- `summary.json`: 실행 메타데이터, 환경 정보, 집계 지표, 모델 단위 지표
- `samples.jsonl`: 샘플별 지표. 예측 텍스트는 `--save_predictions`를 사용한 경우에만 포함됩니다.
- `outliers.jsonl`: 선택한 outlier 정책에 따라 제외된 샘플
- `leaderboard_row.json`: `scripts/generate_leaderboard.py`가 읽는 리더보드용 요약 행
- `predictions.csv`, `error_analysis.jsonl`: `--save_predictions` 사용 시 생성

직접 실행한 결과 외에 검토된 과거 결과나 외부 제출 결과는 `doc/submitted_results.json`에 저장합니다. 이 파일의 행은 `leaderboard_row.json`과 같은 요약 스키마를 사용하며, 저장소 리더보드를 생성할 때 자동으로 합쳐집니다.

업데이트 권한과 검토 흐름은 `doc/leaderboard_update_workflow.md`를 따릅니다. 외부 contributor의 결과도 maintainer 승인과 PR review를 거쳐 병합하면 GitHub Pages에 반영됩니다.

저장소 단위 리더보드를 생성하려면 아래 명령을 실행합니다.

```bash
python scripts/generate_leaderboard.py --results_dir results --markdown_path leaderboard.md
```

기본값에서는 `--limit`을 사용한 부분 실행 결과를 제외합니다. `--include_partial`은 디버깅이나 로컬 스모크 테스트에서만 사용하고, 부분 실행 결과를 리더보드 순위로 공개하지 마세요.

GitHub Pages용 정적 사이트는 생성된 JSON 데이터를 사용합니다.

```bash
python scripts/build_pages.py --output_dir _site
```

정적 사이트는 모델별 평균 성능을 보여주는 `Overall` 리더보드와 데이터셋별 탭을 함께 렌더링합니다. `Overall` 화면은 모델/데이터셋 조합마다 가장 좋은 전체 평가 결과를 하나씩 남긴 뒤, 데이터셋 조합 전체의 평균을 계산합니다.

공개 명령은 로컬 경로가 노출되지 않도록 정리합니다. 공개 아티팩트에는 `$KSPON_ROOT`, `$AIHUB_TELEPHONE_ROOT` 같은 placeholder를 사용하세요.

Pages 워크플로는 생성된 `_site/` 디렉터리를 `actions/configure-pages`, `actions/upload-pages-artifact`, `actions/deploy-pages`로 배포합니다. 배포 전에 저장소 Pages 소스를 GitHub Actions로 설정해야 합니다.
