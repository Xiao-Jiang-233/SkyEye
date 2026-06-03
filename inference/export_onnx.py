# ============ inference/export_onnx.py ============
"""
ONNX 模型导出 + 图简化 + PyTorch vs ONNX Runtime 推理测速

输出：results/weather_model.onnx
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
    将 PyTorch 模型导出为 ONNX 格式

    Args:
        model_path: str — .pth 权重文件路径
        onnx_path: str — 输出 .onnx 路径

    Returns:
        str: ONNX 文件路径
    """
    cfg = CONFIG
    device = torch.device(cfg["device"])
    onnx_path = onnx_path or cfg["onnx_path"]

    # 加载模型
    model = WeatherEfficientNet(
        model_name=cfg["student_model"],
        num_classes=cfg["num_classes"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 创建 dummy input
    dummy_input = torch.randn(1, 3, cfg["img_size"], cfg["img_size"]).to(device)

    # 导出 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )

    # 验证 ONNX 模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX model exported and validated: {onnx_path}")

    # 尝试简化图结构
    try:
        import onnxsim
        model_simp, check = onnxsim.simplify(onnx_path)
        if check:
            onnx.save(model_simp, onnx_path)
            print("✓ ONNX model simplified")
        else:
            print("! onnxsim simplification check failed, keeping original")
    except ImportError:
        print("! onnxsim not installed, skipping simplification (pip install onnxsim)")

    return onnx_path


def benchmark_inference(model_path, onnx_path=None):
    """
    对比 PyTorch 原生 vs ONNX Runtime 推理速度

    Args:
        model_path: str — .pth 权重文件
        onnx_path: str — .onnx 文件（如已有则跳过导出）
    """
    cfg = CONFIG
    device = torch.device(cfg["device"])

    # --- PyTorch Inference ---
    model = WeatherEfficientNet(
        model_name=cfg["student_model"],
        num_classes=cfg["num_classes"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dummy = torch.randn(1, 3, cfg["img_size"], cfg["img_size"]).to(device)

    # Warmup
    for _ in range(20):
        _ = model(dummy)

    # Benchmark
    torch_times = []
    with torch.no_grad():
        for _ in range(100):
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            torch_times.append(time.perf_counter() - start)

    torch_avg = np.mean(torch_times) * 1000  # ms
    torch_std = np.std(torch_times) * 1000
    print(f"PyTorch inference: {torch_avg:.3f} ± {torch_std:.3f} ms")

    # --- ONNX Runtime Inference ---
    if onnx_path is None:
        onnx_path = cfg["onnx_path"]

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    dummy_np = dummy.cpu().numpy()

    # Warmup
    for _ in range(20):
        _ = session.run(None, {'input': dummy_np})

    # Benchmark
    onnx_times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = session.run(None, {'input': dummy_np})
        onnx_times.append(time.perf_counter() - start)

    onnx_avg = np.mean(onnx_times) * 1000
    onnx_std = np.std(onnx_times) * 1000
    speedup = torch_avg / onnx_avg
    print(f"ONNX Runtime inference: {onnx_avg:.3f} ± {onnx_std:.3f} ms")
    print(f"Speedup: {speedup:.2f}×")

    return {"torch_ms": torch_avg, "onnx_ms": onnx_avg, "speedup": speedup}


if __name__ == "__main__":
    onnx_path = export_to_onnx(CONFIG["pruned_ckpt"])
    benchmark_inference(CONFIG["pruned_ckpt"], onnx_path)
