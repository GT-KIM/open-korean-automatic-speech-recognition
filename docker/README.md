# 재측정 환경

## 2026-10-08 측정 코드 재현

저장소 루트에서 `python scripts/verify_benchmark_release.py`를 실행하면 공개 규약의
소스 해시, 정확도 실행 스크립트의 원본 바이트 해시, 172개 패키지 목록 해시를 대조합니다.
Python 표준 라이브러리만 필요합니다. Windows/Linux checkout 모두 같은 검사를 사용합니다.
`run_full_accuracy.py`는 실행 당시 CRLF 바이트를 보존하므로 자동 줄바꿈 변환을 하지 마세요.

- [정확도 규약](../doc/benchmarks/server_accuracy_protocol_20261008.json): 모델 revision·디코딩·입력 수·환경.
- [속도 규약](../doc/benchmarks/server_speed_protocol_20261008.json): 고정 768개·B1/B4·3회 반복.
- [전체 패키지 목록](benchmark-package-inventory.json): 실제 봉인 이미지에서 읽은 버전 목록.
- [순위 민감도 분석](../doc/benchmarks/outlier_rank_sensitivity_20261009.md): 보존 출력 재집계 결과.

일반 개발 설치는 루트 `requirements.txt`를 사용하고, benchmark 빌드는
`benchmark-requirements.txt`와 `benchmark-constraints.txt`로 측정 당시 PyTorch 2.13 환경을 선택합니다.
패키지 메타데이터는 개발용 2.14와 benchmark 2.13을 허용합니다.

최종 측정 이미지 ID는
`sha256:f9995a10077eb8bbd739388ff97c54f68e7656e6524183d1f76553c6f03c7e14`입니다.
이 저장소에는 이미지 archive·모델 캐시·평가 원본을 포함하지 않습니다. 현재 공개 레지스트리 다운로드 주소도 없습니다.
따라서 해당 archive를 가진 환경에서는 정확한 이미지를 재사용할 수 있지만, 소스만 받은 환경에서
동일 image ID를 확보했다고 주장할 수는 없습니다. 재빌드 결과는 새 이미지·preflight로 별도 봉인해야 하며,
기존 규약의 image ID를 임의로 대체해 기존 측정과 동일하다고 처리하지 않습니다.

로컬 Windows/WSL2와 Linux 서버에서 **한 번 빌드한 동일한 Linux 이미지**를 사용합니다.
각 호스트의 GPU·드라이버·커널은 달라질 수 있으므로 결과를 환경별로 분리합니다.
모델 commit, BF16, batch 1/4, curated 768개와 입력 순서는 공통 측정 규약을 따릅니다.

## 호스트 준비

- 로컬: Docker Desktop의 WSL2 backend 또는 Ubuntu WSL2의 Docker Engine을 사용합니다.
  Windows NVIDIA 드라이버를 사용하며, WSL 내부에 Linux GPU 드라이버를 설치하지 않습니다.
- Linux 서버: 기존 NVIDIA 드라이버와 Docker 설치를 확인하고 NVIDIA Container Toolkit을 구성합니다.
  서버 OS·GPU·드라이버를 먼저 조사하고 기존 작업과 공유하는 Docker daemon을 임의로 재시작하지 않습니다.
- 컨테이너마다 GPU 한 개를 선택합니다. 각 장비에서 평가 작업은 한 개씩 실행합니다.

설치 절차는 [Docker Engine](https://docs.docker.com/engine/install/ubuntu/),
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
[CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/) 공식 문서를 따릅니다.
CUDA toolkit을 호스트에 별도로 설치할 필요는 없으며 컨테이너 runtime을 사용합니다.

현재 CUDA 13 이미지는 [NVIDIA 드라이버 580 이상](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)이 필요합니다.
Ubuntu 22.04/24.04에서는 호스트 상태를 확인한 뒤 `sudo bash docker/install-ubuntu-host.sh`로 Docker와 NVIDIA Container Toolkit을 설치할 수 있습니다.
이 스크립트는 GPU 드라이버를 교체하거나 재부팅하지 않습니다. 실행 중인 컨테이너가 있으면 중단하며, 설치 후 Docker daemon을 재시작합니다.

## 이미지 빌드와 이동

저장소 루트에서 실행합니다. 빌드는 GPU를 사용하지 않지만 CUDA 패키지 다운로드와 디스크 공간이 필요합니다.

```bash
python docker/benchmark.py build
docker image inspect openkoasr-benchmark:20261006 --format '{{.Id}}'
docker save -o openkoasr-benchmark-20261006.tar openkoasr-benchmark:20261006
```

tar 파일을 다른 장비로 전달한 후 불러옵니다. 두 장비에서 **image ID가 같아야** 같은 실행 환경입니다.
태그가 같거나 Dockerfile이 같은 것만으로 확인하지 않습니다. 레지스트리를 사용한다면 같은 digest를 받습니다.

```bash
docker load -i openkoasr-benchmark-20261006.tar
docker image inspect openkoasr-benchmark:20261006 --format '{{.Id}}'
```

`Dockerfile.benchmark`의 base image는 digest로 고정돼 있습니다.
베이스의 CUDA 패키지를 활용하는 컨테이너 내부 venv에 의존성을 설치합니다.
`benchmark-constraints.txt`는 핵심 패키지의 요구 버전이며 완전한 Linux dependency lock은 아닙니다.
이미지 빌드 시 해결된 나머지 패키지와 Python은 점검 보고서에 기록됩니다.
별도 재빌드의 재현성을 보장하지 않으므로, 본 측정에는 봉인한 최종 이미지를 재사용합니다.
소스 코드를 컨테이너에 bind mount하지 않아 장비별 checkout 차이가 실행 코드를 바꾸지 않습니다.
`.dockerignore`는 데이터·결과·캐시·개인 설정을 빌드 context에서 제외합니다.

## 실행 환경 점검

Windows PowerShell 또는 WSL shell에서는:

```bash
python docker/benchmark.py check --environment local-wsl2 --host-kind wsl2 --gpu 0
```

Linux 서버에서는:

```bash
python3 docker/benchmark.py check --environment linux-server --host-kind native-linux --gpu 0
```

사용할 GPU 번호 또는 UUID를 `--gpu`에 지정합니다. `--host-kind`는 컨테이너가 아니라 호스트의 종류입니다.
WSL은 `--gpus all`로 노출한 뒤 CUDA visibility로 한 개를 선택하며,
일반 Linux는 Docker에도 선택한 한 개만 전달합니다.
로컬 Docker daemon을 대상으로 실행하세요. 원격 Docker context에서는 bind mount 경로가 달라집니다.

매번 `results/environments/<environment>/<UTC 시각>/`에 새 기록을 만듭니다.
`image.json`은 호스트에서 확인한 image ID, `environment.json`은 컨테이너의 패키지·코드 해시·GPU·드라이버·BF16 점검 결과입니다.
네트워크를 차단한 채 오디오 decode/resample, 백엔드 import, GPU BF16 연산만 수행합니다.
실패는 종료 코드 1로 반환하며 기존 보고서를 덮어쓰지 않습니다.

환경 점검 통과는 실제 ASR 측정 준비 완료를 뜻하지 않습니다. 다음 순서로 확인합니다.

1. 두 장비의 이미지·Python·패키지·코드 해시 일치와 개별 GPU/드라이버 기록.
2. 데이터 경로 바인딩과 curated 원본 오디오/정답 해시 검증.
3. 모델 재사용·warmup 규약을 지키는 측정 실행기 연결.
4. 8개 모델의 192개 pilot에서 BF16 및 batch 1/4 동작 확인.

본평가 결과 루트는 `results/remeasure/<environment>/<track>/<repeat>/`처럼 환경·트랙·반복을 분리합니다.
데이터는 이미지에 넣지 않고 읽기 전용 mount로 연결합니다. 모델 cache는 다운로드 때만 쓰고 측정 중에는 읽기 전용으로 연결하며, 결과는 별도 쓰기 mount를 사용합니다.
기존 `run.sh`/`run.bat`은 개발용 예전 컨테이너를 재사용하므로 재측정에는 사용하지 않습니다.

## 고정 입력과 pilot

`scripts/seal_speed_inputs.py`는 고정 선택 목록의 SHA-256을 먼저 확인한 뒤 기존 데이터셋 로더로 원본을 읽습니다.
ID·16 kHz 길이·kspon 정규화 정답 글자 수가 다르면 중단합니다. 기존 로더의 float32 파형을 손실 없이 NPY로 저장하고,
원본 오디오·라벨, 파형, 정답, 전체 manifest의 해시를 남깁니다. 192개 pilot manifest는 같은 768개 입력의 부분집합입니다.
출력 폴더는 새 경로여야 하며 데이터와 전사는 `results/` 아래 비공개로 보관합니다.

```bash
python scripts/seal_speed_inputs.py \
  --protocol doc/benchmarks/server_speed_protocol_20261006.json \
  --selection results/speed-curation-20261006/selection.jsonl \
  --pilot-selection results/speed-curation-20261006/pilot_selection.jsonl \
  --kspon-root /path/to/KsponSpeech --telephone-root /path/to/telephone \
  --output results/speed-inputs/speed-768-v1
```

측정 실행기를 추가한 이미지는 기존에 검증한 패키지 이미지를 확장합니다. 기반 image ID를 확인하고 빌드용 로컬 태그를 붙이므로,
먼저 이전 공통 이미지 archive를 적재해야 합니다. `Dockerfile.speed`를 직접 빌드하는 대신 다음 명령을 사용합니다.

```bash
python docker/benchmark.py build-speed
python docker/benchmark.py check --image openkoasr-speed:20261007 --environment linux-server --host-kind native-linux --gpu 0
```

새 image ID·소스 해시·통과 preflight와 입력 manifest 해시를 규약에 기록한 뒤 실행합니다.
`scripts/download_speed_models.py --protocol <규약>`으로 먼저 공개 고정 revision의 모델 cache를 준비합니다.
Hugging Face snapshot과 generation config 해시를 확인하며 다운로드 과정과 실제 측정은 분리합니다.

```bash
python docker/run_speed.py \
  --protocol doc/benchmarks/server_speed_protocol_20261006.json \
  --inputs results/speed-inputs/speed-768-v1 \
  --cache /path/to/huggingface \
  --preflight results/environments/linux-server/<UTC>/environment.json \
  --environment linux-server --host-kind native-linux --gpu 0 \
  --purpose pilot --output-root results/speed-pilot
```

각 모델·트랙을 별도 offline 컨테이너에서 순차 실행합니다. 모델은 트랙당 한 번 로드하며 각 그룹/반복의 첫 16개를 단건 warmup한 뒤 본 측정에서 다시 포함합니다.
파일 읽기·내용 검증·warmup·채점·결과 기록은 타이머 밖에 두고 transcribe 호출 전후 CUDA를 동기화합니다.
모든 모델 파라미터의 BF16, 고정 revision, 전체 출력 수와 EOS 종료를 검사합니다.
pilot은 EOS가 없는 출력도 192개 입력에 포함해 기록하고 해당 조합을 실패로 표시합니다.
기본값은 실패한 조합에서 종료하며, `--continue-on-failure`를 붙이면 나머지 모델·트랙도 진단합니다. 전체 종료 코드는 실패를 유지합니다.
이 옵션은 pilot에만 허용합니다. 종료 정책이 없는 기존 규약은 `require_eos`를 적용해 formal의 EOS 미종료 시 즉시 중단합니다.
v4의 `termination.policy=retain_budget_terminated`는 생성 토큰 수가 정확히 고정 상한에 도달한 출력만 유지하고,
측정·warmup 및 그룹·반복별 상한 종료 건수와 비율을 별도 기록합니다. 상한 이전의 EOS 없는 종료나 출력·audit 수 불일치는 계속 오류입니다.
실행할 때 규약 원문을 결과 루트의 `protocol.json`에 복사해 고정하고 SHA-256을 기록합니다.
단건 p50/p95와 batch 4 처리량을 구분하며, batch 시간은 한 번만 합산합니다. pilot 결과는 공식 순위에 게시하지 않습니다.
서버 정식 측정의 image ID·환경 보고서·종료 정책은 [v4 규약](../doc/benchmarks/server_speed_protocol_20261008.json)을 기준으로 합니다.
앞선 공통 이미지와 두 호스트 pilot 준비 이력은 [v3 규약](../doc/benchmarks/server_speed_protocol_20261006.json)에 보존합니다.

## 전체 정확도 평가

[전체 정확도 규약](../doc/benchmarks/server_accuracy_protocol_20261008.json)은 v4 이미지의 기존 평가기를 재사용합니다.
`scripts/run_full_accuracy.py`는 이미지 밖에서 읽기 전용으로 마운트하고 파일 SHA-256을 규약과 대조합니다.
컨테이너의 `/sources/kspon`, `/sources/telephone`에 원본을 읽기 전용으로 연결한 뒤 전체 입력을 먼저 봉인합니다.

호스트에서 다음 명령으로 새 결과 폴더를 지정합니다. 데이터 경로와 모델 캐시는 사전에 준비해야 합니다.
캐시는 `hub/`를 포함하는 Hugging Face home입니다. 전체 48개 조합을 실행하므로 GPU가 비어 있을 때 사용합니다.

```bash
python docker/run_accuracy.py \
  --protocol doc/benchmarks/server_accuracy_protocol_20261008.json \
  --kspon-root /path/to/KsponSpeech --telephone-root /path/to/telephone \
  --cache /path/to/huggingface --output results/full-accuracy-new \
  --environment linux-server --host-kind native-linux --gpu 0
python scripts/validate_full_accuracy.py \
  --results-root results/full-accuracy-new --output results/full-accuracy-new/validated.json
```

호스트 실행기는 당시 사용한 순서를 경로 인자로 옮긴 것입니다. 동일 이미지 preflight → 입력 봉인 →
tiny 8개 smoke → 48개 전체 평가 순서이며, `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`을 사용합니다.
프로토콜·preflight·matrix·종료 코드·로그를 결과 폴더에 보존합니다. 데이터·캐시·harness는 읽기 전용 mount,
평가는 offline 컨테이너에서 실행합니다. WSL2에서는 `--host-kind wsl2`와 별도 환경 이름을 지정합니다.
새 호스트의 관측 GPU·스레드 수는 별도 cohort 근거이므로 기존 속도 순위에 합치지 않습니다.
`validate_full_accuracy.py`는 평가 패키지 의존성이 설치된 환경에서 실행해야 합니다.

batch 4·workers 0·16개 단건 warmup·1회를 사용합니다. 본 평가에는 `--limit`을 사용하지 않습니다.
각 모델은 구간별로 한 번 로드하며, FLOP 프로파일링은 제외합니다. 종료 감사는 기존 평가기 타이머 밖에서 실행합니다.
상한 종료 출력도 원본과 all-sample 지표에 보존하고 기존 메인 규약 `v1/kspon/cer>1.0`을 유지합니다.
배치별 `journal.jsonl`과 구간별 상태·완료 파일로 중간 실패를 추적합니다.

`scripts/validate_full_accuracy.py --results-root $ACCURACY_RESULTS --output $REPORT`는 48개 전체 실행의
입력 순서·출력·환경·종료·집계와 전화음성 `all` 8개를 검증합니다. `--partial`은 완료 구간만 점검하며 게시 자격을 부여하지 않습니다.
전체 정확도 실행의 시간 통계는 별도 curated 속도 결과를 대체하지 않습니다.

## 보존 결과만으로 순위 분석

아래 명령은 표준 라이브러리만 사용하며 모델 로드와 추론을 하지 않습니다. 원본 결과 폴더에는
봉인된 protocol, inputs, samples.jsonl이 있어야 합니다. 공개 저장소의 집계 JSON만으로는 실행할 수 없습니다.

```bash
python scripts/analyze_outlier_rank_sensitivity.py \
  --results-root /path/to/accuracy-full-20261008 \
  --verified-report doc/benchmarks/server_accuracy_results_20261008.json \
  --output results/outlier-rank-sensitivity.json
```

48개 출력 파일과 입력 manifest 해시를 검증한 뒤 56개 기존 filtered CER를 재현합니다.
빈 정답의 정의 불가능한 CER는 삭제하거나 0으로 치환하지 않으며, 전체 점수·순위를 null로 기록합니다.
모든 모델에 동일한 빈 정답 제외 집합을 적용한 보조 분석을 별도로 제공합니다. 공식 순위는 수정하지 않습니다.
