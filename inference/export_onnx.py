# ============ inference/export_onnx.py ============
"""
ONNX 模型导出 + INT8 量化 + CPU 推理测速

流程：PyTorch → ONNX FP32 → INT8 动态量化 → CPU Benchmark
输出：results/weather_model.onnx, results/weather_model_int8.onnx
"""
import torch
import onnx
import onnxruntime as ort
import numpy as np
import time

from config import CONFIG
from models.weather_efficientnet import WeatherEfficientNet


def export_to_onnx(model_path, onnx_path=None):
    """
    将 PyTorch 模型导出为 ONNX FP32 格式

    Args:
        model_path: str — .pth 权重文件路径
        onnx_path: str — 输出 .onnx 路径

    Returns:
        str: ONNX 文件路径
    """
    cfg = CONFIG
    device = torch.device("cpu")  # 导出时用 CPU，避免 CUDA 图捕获问题
    onnx_path = onnx_path or cfg["onnx_path"]

    model = WeatherEfficientNet(
        model_name=cfg["student_model"],
        num_classes=cfg["num_classes"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()

    dummy_input = torch.randn(1, 3, cfg["img_size"], cfg["img_size"]).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model exported: {onnx_path}")

    # 简化图结构（可选）
    try:
        import onnxsim
        model_simp, check = onnxsim.simplify(onnx_path)
        if check:
            onnx.save(model_simp, onnx_path)
            print("✓ ONNX model simplified")
    except ImportError:
        pass

    return onnx_path


def quantize_to_int8(onnx_path=None, int8_path=None):
    """
    对 FP32 ONNX 模型进行 INT8 动态量化（CPU 推理加速 2-4×）

    Args:
        onnx_path: str — FP32 ONNX 路径
        int8_path: str — INT8 ONNX 输出路径

    Returns:
        str: INT8 ONNX 文件路径
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    cfg = CONFIG
    onnx_path = onnx_path or cfg["onnx_path"]
    int8_path = int8_path or cfg["onnx_int8_path"]

    print("Quantizing to INT8 ...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )
    print(f"✓ INT8 model saved: {int8_path}")

    # 打印压缩效果
    import os
    fp32_size = os.path.getsize(onnx_path) / 1024 / 1024
    int8_size = os.path.getsize(int8_path) / 1024 / 1024
    print(f"  FP32 size: {fp32_size:.1f} MB → INT8 size: {int8_size:.1f} MB ({int8_size/fp32_size*100:.0f}%)")

    return int8_path


def benchmark_cpu(onnx_path=None, int8_path=None, warmup=20, runs=100):
    """
    CPU 推理测速：对比 ONNX FP32 vs INT8

    Args:
        onnx_path: str — FP32 ONNX 路径
        int8_path: str — INT8 ONNX 路径
        warmup: int — 预热次数
        runs: int — 测量次数

    Returns:
        dict: {fp32_ms, int8_ms, speedup}
    """
    cfg = CONFIG
    onnx_path = onnx_path or cfg["onnx_path"]
    int8_path = int8_path or cfg["onnx_int8_path"]

    dummy = np.random.randn(1, 3, cfg["img_size"], cfg["img_size"]).astype(np.float32)
    providers = ['CPUExecutionProvider']
    results = {}

    # --- FP32 ---
    session_fp32 = ort.InferenceSession(onnx_path, providers=providers)
    for _ in range(warmup):
        _ = session_fp32.run(None, {'input': dummy})

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = session_fp32.run(None, {'input': dummy})
        times.append(time.perf_counter() - start)

    fp32_ms = np.mean(times) * 1000
    fp32_std = np.std(times) * 1000
    print(f"ONNX FP32 (CPU): {fp32_ms:.2f} ± {fp32_std:.2f} ms")
    results["fp32_ms"] = fp32_ms

    # --- INT8 ---
    if int8_path:
        session_int8 = ort.InferenceSession(int8_path, providers=providers)
        for _ in range(warmup):
            _ = session_int8.run(None, {'input': dummy})

        times = []
        for _ in range(runs):
            start = time.perf_counter()
            _ = session_int8.run(None, {'input': dummy})
            times.append(time.perf_counter() - start)

        int8_ms = np.mean(times) * 1000
        int8_std = np.std(times) * 1000
        speedup = fp32_ms / int8_ms
        print(f"ONNX INT8 (CPU): {int8_ms:.2f} ± {int8_std:.2f} ms  (Speedup: {speedup:.2f}×)")
        results["int8_ms"] = int8_ms
        results["speedup"] = speedup

    return results


if __name__ == "__main__":
    onnx_path = export_to_onnx(CONFIG["pruned_ckpt"])
    int8_path = quantize_to_int8(onnx_path)
    benchmark_cpu(onnx_path, int8_path)
