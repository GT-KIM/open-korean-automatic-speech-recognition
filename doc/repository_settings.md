# 공개 전 저장소 설정

일부 GitHub 보안 설정은 저장소 파일만으로 강제할 수 없습니다. 저장소를 공개하기 전에 GitHub UI에서 아래 설정을 적용하세요.

## `main` 브랜치 규칙

`main` 브랜치에는 ruleset 또는 branch protection rule을 설정합니다.

- Pull request를 통해서만 병합하도록 설정합니다.
- 최소 1명 이상의 승인 리뷰를 요구합니다.
- 새 커밋이 push되면 기존 승인이 만료되도록 설정합니다.
- 가장 최근 push에 대한 승인을 요구합니다.
- 모든 대화가 resolved 상태여야 병합할 수 있게 설정합니다.
- 필수 status check를 통과해야 병합할 수 있게 설정합니다.
  - 필수 check: `.github/workflows/ci.yml`의 `test`
- force push를 차단합니다.
- 브랜치 삭제를 차단합니다.
- 히스토리를 단순하게 유지하려면 squash merge를 기본 병합 방식으로 사용합니다.

## Actions와 Pages

- repository Actions 설정에서 기본 `GITHUB_TOKEN` 권한을 read-only로 설정합니다.
- workflow YAML에는 필요한 권한만 명시합니다.
- GitHub Pages source는 GitHub Actions로 설정합니다.
- workflow에서 꼭 필요하지 않다면 repository secret을 만들지 않습니다.

## Secret과 데이터 보호

- 가능하면 GitHub secret scanning과 push protection을 켭니다.
- 공개 push 전 `python scripts/public_readiness_check.py`를 실행합니다.
- `results/`, `_site/`, 원본 데이터셋, 원본 전사, 비공개 예측 아티팩트는 커밋하지 않습니다.
- 결과 PR을 받을 때는 `doc/data_policy.md`를 먼저 확인합니다.

## 공개 전 체크리스트

1. `git status --short`가 깨끗한지 확인합니다.
2. `python -m unittest discover -s tests`를 실행합니다.
3. `python scripts/public_readiness_check.py`를 실행합니다.
4. `python scripts/build_pages.py --output_dir _site`를 실행합니다.
5. `.github/workflows/pages.yml`로 GitHub Pages가 배포되는지 확인합니다.
