# 리더보드 업데이트 절차

GitHub Pages 리더보드는 저장소에 커밋된 `doc/leaderboard_data.json`을 그대로 읽어 렌더링합니다. GitHub Actions는 Pages 배포 시 평가를 다시 실행하지 않습니다. 따라서 리더보드 변경은 결과 행을 검토한 뒤 `leaderboard.md`와 `doc/leaderboard_data.json`을 재생성하는 Pull Request로 반영합니다.

## 기본 원칙

- 리더보드 순위에는 `--limit` 없이 실행한 전체 평가 결과만 올립니다.
- `results/`, `_site/`, 원본 오디오, 원본 전사, 비공개 샘플별 예측 파일은 커밋하지 않습니다.
- 공개 명령은 `$KSPON_ROOT`, `$AIHUB_TELEPHONE_ROOT` 같은 placeholder를 사용합니다.
- `doc/leaderboard_data.json`은 직접 편집하지 않고 `scripts/generate_leaderboard.py`로 생성합니다.
- `main`은 보호된 브랜치이므로 리더보드 업데이트는 PR, CI, maintainer review를 거쳐 병합합니다.

## 내가 직접 업데이트하는 경우

1. 최신 `main`에서 새 브랜치를 만듭니다.

   ```bash
   git fetch origin main
   git switch -c result/<model>-<dataset> origin/main
   ```

2. 전체 평가를 실행합니다. `--limit`은 사용하지 않습니다.

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

3. 생성된 `results/full/<run_id>/leaderboard_row.json`을 확인합니다.

   - `is_full_evaluation`이 `true`인지 확인합니다.
   - `evaluated_samples`와 `dataset_total_samples`가 전체 평가를 나타내는지 확인합니다.
   - `command`에 로컬 절대 경로나 사용자 이름이 없는지 확인합니다.
   - `source`는 새 재현 가능 결과라면 `verified` 또는 명확한 출처 문자열을 사용합니다.

4. 공개 리더보드 파일을 재생성합니다.

   ```bash
   python scripts/generate_leaderboard.py --results_dir results --markdown_path leaderboard.md
   python scripts/build_pages.py --output_dir _site
   ```

   `results/`와 `_site/`는 `.gitignore` 대상입니다. 커밋 대상은 보통 `leaderboard.md`, `doc/leaderboard_data.json`, 필요 시 `doc/submitted_results.json`입니다.

5. 공개 전 검사를 실행합니다.

   ```bash
   python scripts/public_readiness_check.py
   python -m unittest discover -s tests
   ```

6. PR을 열고 CI가 통과하면 merge합니다. `main`에 merge되면 `.github/workflows/pages.yml`이 Pages 사이트를 자동 배포합니다.

## 다른 사람이 허가를 받아 업데이트하는 경우

외부 contributor에게 `main` push 권한을 줄 필요는 없습니다. 아래 흐름을 권장합니다.

1. contributor가 Issue 또는 PR 설명에 평가하려는 모델, 데이터셋, subset, GPU, 예상 실행 명령을 적습니다.
2. maintainer가 댓글로 범위를 승인합니다. 예: `Approved for KsponSpeech clean / whisper-large-v3 full evaluation.`
3. contributor는 fork 또는 브랜치에서 전체 평가를 실행하고 PR을 엽니다.
4. PR에는 아래 항목을 포함합니다.

   - `summary.json`
   - `leaderboard_row.json` 내용 또는 그 내용을 반영한 `doc/submitted_results.json` 변경
   - 재현 명령
   - GPU/CPU, torch/CUDA, 모델 repository id
   - 데이터셋 약관상 공개 가능한 경우에만 예측 아티팩트 링크

5. maintainer는 PR에서 아래를 확인합니다.

   - 사전 승인 범위와 일치하는 결과인지
   - `--limit` 없는 전체 평가인지
   - `doc/data_policy.md`를 위반하는 파일이 없는지
   - 로컬 경로, username, API key, model token이 없는지
   - `python scripts/public_readiness_check.py`와 CI가 통과하는지
   - `leaderboard.md`와 `doc/leaderboard_data.json`이 스크립트로 재생성되었는지

6. 승인되면 maintainer가 PR을 merge합니다. GitHub Pages는 merge 후 자동으로 최신 리더보드를 배포합니다.

## 과거 결과나 논문/외부 보고 결과를 추가하는 경우

직접 재현 가능한 `leaderboard_row.json`이 없고, 과거 README 표나 외부 보고 결과를 정리해야 할 때는 `doc/submitted_results.json`에 curated row로 추가합니다.

- `source`에는 `README legacy table`, 논문명, issue 번호처럼 출처를 명확히 적습니다.
- `_artifact`에는 `README.md#리더보드`, `issues/<number>`, 외부 URL 등 확인 가능한 위치를 적습니다.
- `verification` 또는 `source` 표현으로 `verified`, `submitted`, `legacy`를 구분합니다.
- 재현 가능한 전체 평가를 나중에 확보하면 legacy/submitted row를 verified run 기반 row로 교체합니다.

## 병합 후 확인

병합 후에는 아래를 확인합니다.

- CI workflow가 성공했는지
- GitHub Pages workflow가 성공했는지
- https://gt-kim.github.io/open-korean-automatic-speech-recognition/ 에 새 행이 보이는지
- `Overall` 탭과 데이터셋별 탭에서 정렬과 metric 값이 예상대로 표시되는지
