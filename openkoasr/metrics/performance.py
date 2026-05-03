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
    실시간 계수 (Real-Time Factor, RTF)를 계산합니다.
    """
    if total_audio_length == 0:
        rtf = float('inf')
    else:
        rtf = total_processing_time / total_audio_length
    return {"rtf": rtf}

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
