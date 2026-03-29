# KSponSpeech
본 문서에서는 [KsponSpeech](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=123) 데이터셋의 평가 방법에 대해 설명한다.

## Data Preprocessing

### Text Normalization
데이터셋과 같이 제공된 [전사규칙](https://aihub.or.kr/aihubnews/notice/view.do?pageIndex=1&nttSn=9746&currMenu=132&topMenu=103)을 기반으로 Text Normalization을 진행한다.
기본적인 규칙은 아래와 같다.

#### 영어/숫자  
   가능한 한 label 내 영어 및 숫자 표기를 한글 표기로 바꿔서 사용한다.  
   
   영어의 경우 알파벳 단위로 읽는 경우와 영단어로 읽는 경우가 혼재되어 있는데, 전사 규칙에 따라 표현한다.  
   
   
   숫자 역시 하나의 숫자에 여러 발음이 가능한데, (e.g. 111: 일일일, 하나하나하나, 백십일) 마찬가지로 전사 규칙에 따라 한국어 발음에 대응되는 한글로 변환한다. 

#### 이중전사  
   이중전사는 발음이 표준 발음에서 벗어나거나 두 가지 이상 발음이 가능한 경우로 두 경우를 raw data에 (철자전사)/(발음전사) 형태로 표현한다.  
   
   예를 들어 (컴퓨터)/(컴퓨타)와 같은 형태로 이중전사가 표현되면, 실제 음성은 "컴퓨타"에 가까울 것이고, 인식 결과는 "컴퓨터"로 표현되는 게 유리할 것이다.  
   본 프로젝트에서는 철자전사를 활용한다. 

#### 잡음
   잡음은 /b (숨소리), /l (웃음소리), /o (다른 사람의 말소리가 포함됨) /n (주변 잡음)이 labeling 되어 있다.  
   
   현재 대부분의 ASR 모델은 잡음 label을 추정하지 않기 때문에, 본 프로젝트에서는 잡음 레이블은 제거하고 평가한다. 다만, 함수에서는 잡음 레이블을 추가할 수 있게 구현하였다.   
   
   (추후 공개할 모델 학습 프로젝트에서는 잡음 label을 추가할 예정)
   
#### 기타
   1. 간투어는 아-, 어- 그- 와 같이 의미가 없는 발성을 뜻한다. 간투어는 아/와 같이 전사되어 있고, 본 프로젝트에서는 /만 제거한다.
   2. 문장 부호는 쉼표, 마침표, 물음표, 느낌표가 허용된다.
   3. 전사 과정에서 알아들을 수 없는 발화를 u/로 표현하고 있다. 본 프로젝트에서는 노이즈와 더불어 unknown 레이블을 제거하고 평가하며 함수 상으로면 구현한다.

### Audio Processing
KSponSpeech는 .pcm 형식의 오디오로 이루어져 있으며, sampling rate은 16kHz이다. 

## OpenKoASR 설정 연동

OpenKoASR에서는 `openkoasr/configs/dataset/KsponSpeech.py` 설정을 통해 데이터셋을 불러온다.

```python
kspon_speech_config = {
    "name": "KsponSpeech",
    "subset": "clean",  # [clean, other]
    "rootpath": "/app/data/KsponSpeech/"
}
```

- `subset` 값(`clean`/`other`)에 따라 `eval_clean.trn`, `eval_other.trn`을 선택한다.
- `rootpath`는 데이터셋 루트 경로이며, Docker 환경에서는 `/app/data/KsponSpeech/`를 기본으로 사용한다.

## 데이터 준비 스크립트

분할 압축 파일 병합/해제는 아래 스크립트를 사용할 수 있다.

```bash
bash scripts/extract_KsponSpeech_data.sh
```

스크립트는 기본적으로 `/app/data/KsponSpeech/` 경로를 사용하므로 환경에 따라 경로를 조정한다.

## 평가 파이프라인

KsponSpeech 평가는 다음 흐름으로 진행된다.

1. `DatasetFactory.load_dataset(config)`
2. `generate_dataloader(batch_size=1, shuffle=False, num_workers=0)`
3. 모델 추론 (`model.inference_sample`)
4. `Evaluator.evaluate()`로 WER/CER/MER/JER/SER/RTF/Latency 계산
5. Outlier 기준(`--outlier_metric`, `--outlier_threshold`) 초과 샘플 제외 후 평균 집계

## 재현 체크리스트

- `subset`, `model_name`, outlier 기준을 함께 기록한다.
- Docker 사용 시 데이터 마운트 경로(`/app/data`)와 `rootpath`를 일치시킨다.
- 지표 정의는 `doc/metrics.md` 문서를 참고한다.