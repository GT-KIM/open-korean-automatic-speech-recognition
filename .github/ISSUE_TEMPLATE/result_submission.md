---
name: Result submission
about: 새 ASR 리더보드 결과를 제출합니다
title: "[Result] "
labels: result
---

## 결과

- Model:
- Dataset/subset:
- Command:
- Hardware:
- OpenKoASR commit:

## 아티팩트

첨부 또는 링크:

- `summary.json`
- `leaderboard_row.json`
- 데이터셋 약관상 재배포가 허용되는 경우에만 공개 예측 아티팩트

## 체크리스트

- [ ] `--limit` 없이 실행한 전체 평가 결과입니다.
- [ ] 실행 명령은 로컬 절대 경로가 아니라 `$KSPON_ROOT` 같은 공개 placeholder를 사용합니다.
- [ ] 제한된 오디오, 전사, 인증 정보, 비공개 샘플별 아티팩트를 첨부하지 않았습니다.
- [ ] 결과는 `doc/result_submission.md`와 `doc/data_policy.md`를 따릅니다.
