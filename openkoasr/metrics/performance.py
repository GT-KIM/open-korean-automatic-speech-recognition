# Copyright (c) 2025 Gwantae Kim. All Rights Reserved.
# Licensed under the MIT License.

try:
    import torch
except Exception:  # pragma: no cover - optional in mock-only runs.
    torch = None

try:
    from calflops import calculate_flops
except Exception:  # pragma: no cover - optional in mock-only runs.
    calculate_flops = None

def real_time_factor(total_processing_time: float, total_audio_length: float) -> dict:
    """
    실시간 배속 (Real-Time Factor times, RTFx)를 계산합니다.

    RTFx = 오디오 길이 / 처리 시간 (값이 클수록 빠름).
    예: RTFx 20이면 실시간 대비 20배 빠른 처리에 해당합니다.
    (HuggingFace Open ASR Leaderboard와 동일한 정의)
    """
    if total_processing_time == 0:
        rtfx = float('inf')
    else:
        rtfx = total_audio_length / total_processing_time
    return {"rtfx": rtfx}

def latency(total_processing_time: float) -> dict:
    """
    지연 시간 (Latency)을 반환합니다.
    """
    return {"latency": total_processing_time}

def get_flops(model, dummy_input) -> dict:
    """
    모델의 FLOPS (Floating Point Operations Per Second)를 계산합니다.
    """
    if calculate_flops is None or torch is None:
        raise RuntimeError("FLOPS calculation requires torch and calflops.")
    flops, macs, params = calculate_flops(model=model,
                                          args=[dummy_input],
                                          kwargs={"attention_mask": torch.ones_like(dummy_input),},
                                          forward_mode='generate',
                                          print_results=False)
    return {"flops": flops, "macs": macs, "params": params}

def get_num_parameters(model) -> dict:
    """
    모델의 파라미터 수를 계산합니다.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }
