# KsponSpeech

본 문서는 [KsponSpeech](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123) 데이터셋의 평가 방법을 설명합니다.

## 데이터 전처리

### 텍스트 정규화

데이터셋과 함께 제공된 [전사규칙](https://aihub.or.kr/aihubnews/notice/view.do?pageIndex=1&nttSn=9746&currMenu=132&topMenu=103)을 기반으로 텍스트를 정규화합니다. 기본 정책은 아래와 같습니다.

#### 영어/숫자

라벨의 영어 및 숫자 표기는 가능한 한 전사 규칙에 맞춰 한글 표기로 변환합니다. 영어는 알파벳 단위로 읽는 경우와 단어로 읽는 경우가 섞여 있을 수 있고, 숫자도 하나의 표기에 여러 발음이 가능합니다. 예를 들어 `111`은 문맥에 따라 `일일일`, `하나하나하나`, `백십일`처럼 달라질 수 있습니다.

#### 이중전사

이중전사는 원문에 `(철자전사)/(발음전사)` 형태로 들어 있습니다. OpenKoASR의 `kspon` preset은 영어 표기처럼 철자전사가 더 적합한 경우 앞쪽 표기를 사용하고, 한국어 발음 변이가 표시된 경우 뒤쪽 발음전사를 사용합니다.

#### 잡음

잡음은 `/b`(숨소리), `/l`(웃음소리), `/o`(다른 사람의 말소리가 포함됨), `/n`(주변 잡음) 등으로 라벨링되어 있습니다. 대부분의 범용 ASR 모델은 잡음 라벨을 직접 예측하지 않으므로, 공개 리더보드 평가에서는 잡음 라벨을 제거합니다.

#### 기타

1. 간투어는 `아/`, `어/`, `그/`처럼 전사되며, OpenKoASR은 `/`만 제거합니다.
2. 문장 부호는 쉼표, 마침표, 물음표, 느낌표를 허용합니다.
3. 알아들을 수 없는 발화를 나타내는 `u/`는 잡음 라벨과 함께 제거합니다.

### 오디오 처리

KsponSpeech는 `.pcm` 형식의 16 kHz 오디오로 구성됩니다.

## OpenKoASR 설정 연동

OpenKoASR에서는 `openkoasr/configs/dataset/KsponSpeech.py` 설정을 통해 데이터셋을 불러옵니다.

```python
kspon_speech_config = {
    "name": "KsponSpeech",
    "subset": "clean",  # [clean, other]
    "rootpath": "/app/data/KsponSpeech/"
}
```

- `subset` 값(`clean`/`other`)에 따라 `eval_clean.trn`, `eval_other.trn`을 선택합니다.
- `rootpath`는 데이터셋 루트 경로이며, Docker 환경에서는 `/app/data/KsponSpeech/`를 기본으로 사용합니다.

## 데이터 준비 스크립트

분할 압축 파일 병합/해제는 아래 스크립트를 사용할 수 있습니다.

```bash
bash scripts/extract_KsponSpeech_data.sh
```

스크립트는 기본적으로 `/app/data/KsponSpeech/` 경로를 사용하므로 환경에 따라 경로를 조정합니다.

## 평가 파이프라인

KsponSpeech 평가는 다음 흐름으로 진행됩니다.

1. `DatasetFactory.load_dataset(config)`
2. `generate_dataloader(batch_size=1, shuffle=False, num_workers=0)`
3. 모델 추론 (`model.inference_sample`)
4. `Evaluator.evaluate()`로 WER/CER/MER/JER/SER/RTF/Latency 계산
5. Outlier 기준(`--outlier_metric`, `--outlier_threshold`) 초과 샘플 제외 후 평균 집계

## 재현 체크리스트

- `subset`, `model_name`, outlier 기준을 함께 기록합니다.
- Docker 사용 시 데이터 마운트 경로(`/app/data`)와 `rootpath`를 일치시킵니다.
- 지표 정의는 `doc/metrics.md` 문서를 참고합니다.
