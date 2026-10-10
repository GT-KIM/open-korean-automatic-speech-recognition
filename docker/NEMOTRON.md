# Nemotron 3.5 한국어 평가

모델 revision은 `ea30d66debe3740a08b573244286791d423d6b3e`입니다.
기본 설정은 BF16, `ko-KR`(prompt 14), lookahead 3, greedy RNN-T입니다.
전체 발화를 한 번에 입력하며 배치 내부에서 순차 추론으로 대체하지 않습니다.
코드가 생성 토큰 수 상한을 별도로 부과하지 않고, Transformers의 네이티브 프레임 종료와
프레임당 최대 10개 토큰 진행 정책을 사용합니다.

## 런타임

기존 봉인 이미지 `sha256:f9995a10077eb8bbd739388ff97c54f68e7656e6524183d1f76553c6f03c7e14`를
로컬에 준비한 환경에서 실행합니다. 이미지 준비 방법은 [기존 재현 문서](README.md)를 참고하세요.
아래 빌드는 별도 이미지를 만들며 Qwen SDK를 제외합니다. 새 SDK를 기존 이미지에 설치하지 않습니다.

```bash
docker tag sha256:f9995a10077eb8bbd739388ff97c54f68e7656e6524183d1f76553c6f03c7e14 \
  openkoasr-nemotron-parent:f9995a10077e
docker build -f docker/Dockerfile.nemotron -t openkoasr-nemotron:20261010 .
```

실측 이미지 ID와 전체 패키지 버전은 [inventory](nemotron-package-inventory.json)에 있습니다.
2026-10-10 실측 소스는 `4d381b8caf6503010fae8f05efa41c7948982148` 커밋입니다.
재현 시 해당 커밋의 소스를 사용합니다. 이후 RNNT 동적 길이의 메타데이터 표기만 보완했으며,
실측 원시 아티팩트와 추론 코드는 보존했습니다.
이 이미지는 소스를 `/app:ro`로 마운트해 사용하는 전용 런타임입니다.
`openkoasr` 전체 의존성을 다시 설치하면 구버전 Transformers를 요구하는 Qwen SDK와 충돌합니다.
컨테이너 작업 디렉터리는 `/tmp`로 지정해 로그가 소스 마운트에 기록되지 않도록 합니다.

가중치는 Hugging Face에서 해당 revision의 `model.safetensors`, JSON 설정과 tokenizer만 미리 내려받습니다.
추론 컨테이너에서는 네트워크를 끄고, 모델 캐시·입력·소스를 읽기 전용으로 마운트합니다.
GPU가 비었는지 확인하고 단일 GPU, CPU 8개, 메모리 32 GiB 조건을 유지합니다.

## 실행

`$SOURCE_SHA`는 해당 소스에서 `openkoasr.evaluation.provenance.code_metadata()`가 반환한
`source_sha256`입니다. `$IMAGE_ID`는 `docker image inspect`로 얻은 불변 ID이며
컨테이너 환경변수 `ASR_IMAGE_ID`에도 전달합니다.
`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`는 모두 8,
`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`은 모두 1입니다.

다음은 컨테이너 내부 명령입니다. 기존 `speed-768-v1` 입력을 `/speed-inputs`에 마운트합니다.

```bash
python /app/scripts/run_nemotron_benchmark.py \
  --purpose pilot --batch-size 4 --inputs /speed-inputs \
  --expected-source "$SOURCE_SHA" --output /results/pilot-b4
```

파일럿은 B1과 B4를 각각 실행합니다. 입력 순서·개수·Korean prompt·전체 프레임 소진·출력 차이를
점검한 후 `--purpose formal`로 768개 입력을 각각 3회 측정합니다.
각 그룹·반복 앞의 16개 B1 warmup은 시간 집계에서 제외합니다.

전체 정확도는 기존 입력 inventory를 `/accuracy-inputs`, 원본 데이터셋을 `/sources/kspon`,
`/sources/telephone`에 마운트합니다. 원본 파일마다 봉인한 파형·정답 해시를 대조합니다.

```bash
python /app/scripts/run_nemotron_accuracy.py \
  --dataset kspon-clean --inputs /accuracy-inputs \
  --input-protocol /app/doc/benchmarks/server_accuracy_protocol_20261008.json \
  --expected-source "$SOURCE_SHA" --output /results/accuracy/kspon-clean
```

대상은 `kspon-clean`, `kspon-other`, `telephone-D01`–`telephone-D04` 총 45,916개입니다.
전체 평가의 배치는 4, 반복은 1입니다. 빈 출력과 CER 초과 출력도 원시 결과에 보존합니다.
파일럿·정식 속도·정확도는 서로 다른 출력 디렉터리를 사용하며 기존 결과를 덮어쓰지 않습니다.

## 검증과 해석

```bash
python scripts/validate_nemotron_results.py \
  --root "$PRIVATE_RESULTS" --speed-manifest "$SPEED_INPUTS/manifest.jsonl" \
  --accuracy-inputs "$ACCURACY_INPUTS" \
  --protocol doc/benchmarks/nemotron_protocol_20261010.json \
  --output "$PRIVATE_RESULTS/validation.json"
```

검증기는 입력 순서·개수, 모델·소스·실행기·패키지, RNNT 프레임 종료, 원시 출력과 저장 결과의 일치,
CER 및 집계를 다시 확인합니다. 원시 오디오·정답·예측은 비공개로 유지합니다.

파일럿에서 BF16 B1/B4 출력은 33/192개, FP32 B1/B4는 8/192개가 달랐습니다.
특징 추출은 모두 같았고 실제 인코더 유효 길이도 감사 길이와 일치했습니다.
따라서 B1 속도 결과에 B4 정확도가 그대로 적용된다고 주장하지 않습니다.
새 Transformers 런타임의 속도는 기존 Whisper/Qwen 측정과 별도 조건으로 표시해야 합니다.
이 절차의 완료는 공개 순위표 반영이나 배포를 자동으로 수행하지 않습니다.
