# OpenKoASR Metrics

본 문서는 OpenKoASR에서 사용하는 평가 지표의 의미와 해석 기준을 간단히 정리합니다.

## 1. 인식 품질 지표

OpenKoASR의 기본 품질 지표는 아래와 같습니다.

- **WER (Word Error Rate)**: 단어 단위 오류율
- **CER (Character Error Rate)**: 글자 단위 오류율
- **MER (Morpheme Error Rate)**: 형태소 단위 오류율
- **JER (Jamo Error Rate)**: 자모 단위 오류율
- **SER (Sentence Error Rate)**: 문장 단위 오류율

값이 낮을수록 성능이 좋습니다.

WER의 편집 통계는 정답 단어열을 예측 단어열로 바꾸는 방향으로 계산합니다.
정답에만 있는 단어는 삭제(`D`), 예측에만 있는 단어는 삽입(`I`), 바뀐 단어는 대체(`S`)입니다.
정답 단어 수는 `N = hits + S + D`이며 WER는 `(S + D + I) / N`입니다.
Macro WER는 유효 샘플별 WER의 평균이고, Micro WER는 유효 샘플 전체의 오류 수 합을 정답 단어 수 합으로 나눕니다.
예를 들어 정답이 `a b`, 예측이 `a`이면 삭제 1개로 WER는 `0.5`입니다.
예측이 비어 있고 정답이 있으면 WER는 `1.0`입니다.
정답이 비어 있으면 예측도 비었을 때 `0.0`, 예측이 있을 때 무한대를 반환합니다.
Micro 집계는 전체 정답 단어 수가 0이면 WER를 생략합니다.

수정 전 WER 구현에서 생성된 결과는 삽입·삭제 통계가 뒤바뀌어 Micro WER가 부정확할 수 있습니다.
샘플별 WER와 Macro WER는 이 오류의 영향을 받지 않지만, 기존 `samples.jsonl`을 그대로 재집계해도 잘못된 통계는 복구되지 않습니다.
해당 결과는 정규화된 정답·예측으로 편집 통계를 다시 계산하거나 평가를 재실행한 뒤 집계 아티팩트를 갱신해야 합니다.

2026-10-03 재계산에서는 기존 카운터가 뒤바뀐 방향임을 425,063개 샘플의 저장 WER와 대조해 검증한 뒤 복구했습니다.
정규화된 정답·예측이 남은 57,015개 샘플은 수정된 함수의 직접 계산과도 일치했습니다.
모델 추론을 재실행하지 않고 74개 실행과 AIHub 통합 결과 9개를 재계산했으며,
공개 리더보드의 Micro WER 63개를 갱신했습니다. 상세 전후 수치는 [재계산 기록](benchmarks/wer_recalculation_20261003.json)에 있습니다.

## 2. 성능 지표

- **RTFx (Inverse Real-Time Factor, 실시간 배속)**
  - 샘플별 계산: `RTFx_i = 오디오 길이_i / 처리 시간_i`
  - 순위표의 **Macro RTFx**: outlier를 제외한 유효 샘플 집합 `V`에 대해 `sum(RTFx_i) / |V|`로 계산한 **macro 평균**입니다. 값이 클수록 빠릅니다.
  - Curated 그래프의 **RTFx**: 고정 입력에서 `sum(오디오 길이_i) / sum(처리 시간_i)`를 계산한 뒤 3회 반복의 중앙값을 표시합니다. 전체 평가 속도를 선택하면 그래프에도 **Macro RTFx**를 표시합니다.
  - Macro RTFx는 전체 처리량인 `sum(오디오 길이_i) / sum(처리 시간_i)`과 다릅니다. 동일한 유효 샘플에서도 Whisper small의 KsponSpeech clean 결과는 macro **61.00×**, 합계 비율 **59.09×**입니다.
  - 개별 샘플의 RTFx는 RTF(`처리 시간 / 오디오 길이`)의 역수입니다. 평균 RTFx가 평균 RTF의 역수인 것은 아닙니다.
  - Overall은 세 평가 구간의 macro RTFx를 각각 1/3로 평균냅니다. On-device 표의 QNN Macro RTFx도 outlier 제외 macro이며, 원본의 `performance.qnn_rtfx_all_samples`는 전체 샘플의 합계 비율입니다.
- **Latency**
  - Outlier를 제외한 샘플별 처리 시간의 평균입니다. JSON은 초, 화면은 ms로 표시합니다. 배치 실행에서는 배치 시간을 샘플 수로 나누므로 요청 1개의 응답 지연과 다를 수 있습니다.

평가기는 비어 있는 오디오, 잘못된 샘플레이트, 유한한 양수가 아닌 처리 시간을 거부합니다. 모델이 별도로 보고한 처리 시간이 있으면 이를 검증해 사용하고, 없으면 실제 경과 시간을 사용합니다. 배치 처리 시간도 같은 기준으로 검증합니다.

API 캐시를 재사용할 때는 저장된 `request_latency`로 속도를 계산합니다. 이 값이 없거나 잘못된 캐시는 파일을 읽은 시간을 추론 시간으로 대신 쓰지 않고 오류를 보고합니다. 오류에 표시된 캐시를 제거한 뒤 다시 평가하면 새 API 요청이 발생합니다.

2026-10-03 점검에서는 Google clean 첫 샘플의 캐시 읽기 시간(0.0116초)을 동일 샘플의 앞선 실행에 저장된 실제 추론 측정값(2.6117초)으로 복구했습니다. 정답·전사·오디오 길이·샘플레이트와 평가 조건을 대조했으며, RTFx는 3.0396에서 2.5805로 수정됐습니다. 인식 품질 지표는 동일합니다. 상세 내용은 [오디오·시간 무결성 점검 기록](benchmarks/audio_timing_integrity_20261003.json)에 있습니다.

## 3. 모델 규모/연산량 지표

- **Params**: 모델 파라미터 수
- **FLOPS / MACS**: 모델 연산량 지표

> 참고: 모델 백엔드에 따라 FLOPS 계산이 제한될 수 있습니다.

## 4. Main score와 outlier를 함께 해석하기

Main CER는 outlier를 제외한 샘플별 CER의 평균입니다. 일반 구간의 인식 품질을 평가하는 주 점수이며 기본 순위는 이 값을 사용합니다. outlier 판정 및 기존 메인 점수 계산은 유지합니다.

- 기본값
  - `--outlier_metric cer`
  - `--outlier_threshold 1.0`

즉, 기본 설정에서는 `CER > 1.0` 샘플이 평균 계산에서 제외됩니다.

메인 점수는 다음 두 지표와 함께 해석해야 합니다.

- **Outlier rate**: `outlier_count / evaluated_samples`. 메인 점수에서 제외한 샘플의 비율입니다. 일반적인 실패율 전체를 의미하지는 않습니다. 예를 들어 CER가 정확히 1.0인 빈 예측은 기본 정책에서 outlier가 아닙니다.
- **All-sample CER**: outlier를 포함한 전체 샘플의 `(S + D + I)` 합을 전체 정답 글자 수 합으로 나눈 corpus CER입니다. 샘플별 CER의 단순 평균이 아닙니다. 정답이 빈 샘플의 삽입 오류도 분자에 포함하며, 전체 정답 글자 수가 0이면 점수를 제공하지 않습니다.

메인 CER와 전체 corpus CER의 차이는 outlier 포함 여부와 macro/micro 가중 방식 모두의 영향을 받습니다. outlier의 영향만 비교하려면 상세 정보의 outlier 제외 `CER · corpus`와 All-sample CER를 비교합니다.

`metrics.macro`와 `metrics.micro`는 기존 outlier 제외 점수입니다. `metrics.all_samples_micro`는 전체 샘플의 CER/WER/MER/JER/SER corpus 통계이며, 실행 summary에서는 `aggregate.all_samples_micro_average`에 저장합니다. 원본 통계가 없는 과거 제출 결과는 값을 추정하지 않고 N/A로 표시합니다.

Overall의 Main CER, Outlier rate, All-sample CER는 KsponSpeech clean 3,000개·other 3,000개·AIHub all 39,916개를 모두 완료한 모델에 대해 세 구간별 값을 각각 1/3로 가중한 평균입니다. 저장소 ID로 별칭을 묶고 구간마다 고정된 대표 결과를 하나씩 사용합니다. D01–D04를 추가로 합산하지 않습니다. All-sample CER는 세 구간 모두에 통계가 있을 때만 제공하며, 전체 구간의 편집 카운터를 합친 corpus CER와는 다릅니다. 평가 구간 미완료 모델은 Overall 순위를 부여하지 않지만 데이터셋별 결과를 유지합니다. outlier 비율에 따른 순위 제한은 적용하지 않습니다.

Overall과 서버 데이터셋 순위는 `v1/kspon/cer>1.0` 규약(`kspon` 정규화, `CER > 1` outlier)을 사용합니다. 다른 규약 또는 미확인 결과는 접힌 ‘참고 결과’ 영역에서 확인합니다. 상세 화면에는 비교에 필요한 평가 조건을 표시하며, 전체 재현 정보는 원본 JSON으로 제공합니다.

2026-10-03에는 기존 74개 실행의 425,063개 샘플과 통합 결과 9개, 온디바이스 결과 5개에 전체 통계를 추가했습니다. 모델 추론은 재실행하지 않았고 메인 점수·속도·outlier 판정은 보존했습니다. 서버/API 공개 결과 62개 중 61개와 온디바이스 5개에 전체 CER를 제공하며, 원본 통계가 없는 과거 제출 1개는 N/A입니다. [재집계 기록](benchmarks/all_samples_metrics_20261003.json)에 계산 방법과 원본 해시를 기록했습니다.

## 5. 재현을 위한 기록 권장 항목

비교 실험 시 아래 항목을 함께 기록하는 것을 권장합니다.

1. `dataset_name`, `subset`
2. `model_name`
3. outlier 기준 (`outlier_metric`, `outlier_threshold`)
4. 실행 환경 (GPU, CUDA, torch 버전)
