# Persistent Qualcomm QNN Whisper runner

This Android executable keeps the Qualcomm Whisper encoder and decoder context binaries loaded
while processing a fixed-size stream of native mel features. It implements the autoregressive
self-cache loop on-device and emits token IDs plus QNN execution times.

It builds against Qualcomm's QAIRT `QNN/SampleApp` sources; those SDK sources and runtime binaries
are not vendored in this repository.

## Build

Set `QNN_SDK_ROOT` to an extracted QAIRT 2.47 SDK and invoke Android NDK r26c:

```powershell
$env:QNN_SDK_ROOT = "C:/path/to/qairt/2.47.0.260601"
C:/path/to/android-ndk-r26c/ndk-build.cmd `
  NDK_PROJECT_PATH=native/qnn_whisper_runner `
  APP_BUILD_SCRIPT=native/qnn_whisper_runner/jni/Android.mk `
  NDK_APPLICATION_MK=native/qnn_whisper_runner/jni/Application.mk
```

The executable is written to `native/qnn_whisper_runner/libs/arm64-v8a/`.

The host-side deployment, resumable execution, decoding, and scoring workflow is implemented in
[`scripts/run_qnn_whisper_kspon_full.py`](../../scripts/run_qnn_whisper_kspon_full.py).
