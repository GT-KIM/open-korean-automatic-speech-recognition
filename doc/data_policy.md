# 데이터 정책

OpenKoASR은 평가 코드, 집계 지표, 재현 메타데이터를 공개합니다. 데이터셋 라이선스와 개인정보 검토상 명확히 허용되는 경우가 아니라면 벤치마크 오디오, 원본 전사, 샘플별 예측 아티팩트는 재배포하지 않습니다.

## 지원 데이터셋

### KsponSpeech

KsponSpeech는 ETRI/NIA가 구축한 한국어 자유 대화 음성 말뭉치입니다. Applied Sciences 논문 "KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition"은 이 데이터셋을 969시간 규모의 한국어 open-domain dialog speech corpus로 설명하며, 한국 정부 공개 데이터 허브를 통해 제공된다고 밝힙니다.

- 논문: https://www.mdpi.com/2076-3417/10/19/6936
- ETRI 기록: https://ksp.etri.re.kr/ksp/article/read?id=62525

사용자는 공식 배포 경로에서 데이터셋을 직접 받아야 하며, 적용되는 이용 약관을 따라야 합니다. OpenKoASR은 데이터셋 이름, subset 이름, 집계 지표, 재현 명령만 저장합니다.

### AIHub Low-Quality Telephone ASR

`AIHubLowQualityTelephone`은 AI-Hub `저음질 전화망 음성인식 데이터`(`dataSetSn=571`)를 가리킵니다. AI-Hub는 이 데이터셋을 실제 상담 환경에서 수집한 저음질 전화망 음성과 전사 데이터로 설명하며, 총 6,500시간 규모라고 안내합니다.

- 데이터셋 페이지: https://www.aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=571&topMenu=

사용자는 AI-Hub를 통해 직접 신청하고 다운로드해야 합니다. 원본 오디오, 원본 라벨, 샘플 전사, 비공개 split에서 파생된 prediction 파일은 커밋하지 마세요.

## 커밋 가능한 항목

- `leaderboard_row.json` 형식의 집계 결과 행
- 환경 정보와 집계 지표 메타데이터를 담은 `summary.json`
- `$KSPON_ROOT` 같은 placeholder를 사용한 공개 재현 명령
- 데이터셋 준비 및 평가 정책 문서

## 커밋하면 안 되는 항목

- 원본 오디오 또는 원본 전사
- 개인정보 또는 재배포 제한이 있는 데이터셋의 샘플별 예측 결과
- 로컬 절대 경로, 사용자 이름, 마운트 경로, 머신 이름, 토큰, 인증 정보
- API key, 모델 접근 토큰, cookie, 다운로드한 private 아티팩트

## 공개 명령 정책

공개 명령은 환경에 종속되지 않는 경로를 사용해야 합니다.

```bash
python -m openkoasr.main --dataset_name KsponSpeech --dataset_rootpath $KSPON_ROOT --model_name whisper_tiny
python -m openkoasr.main --dataset_name AIHubLowQualityTelephone --dataset_rootpath $AIHUB_TELEPHONE_ROOT --model_name whisper_tiny
```

`scripts/generate_leaderboard.py`는 `doc/leaderboard_data.json`을 작성할 때 흔한 로컬 경로를 자동으로 정리합니다.
