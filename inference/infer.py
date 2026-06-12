# ============ inference/infer.py ============
"""单张/批量推理脚本

加载优先顺序：INT8 ONNX → FP32 ONNX → PyTorch 权重
Logit bias 已烘焙在 ONNX 模型中（导出时通过 LogitBiasWrapper），
PyTorch 回退路径手动应用 bias，与训练时验证行为一致。
"""
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from config import CONFIG


def _build_bias_tensor(class_names, cfg=None):
    """构建 per-class logit bias 张量（logits - bias）"""
    bias = torch.zeros(len(class_names))
    bias_cfg = (cfg or CONFIG).get("logit_bias", {})
    if bias_cfg:
        for cls_name, val in bias_cfg.items():
            if cls_name in class_names:
                bias[class_names.index(cls_name)] = val
    return bias


def _get_transform(img_size=380):
    """获取推理用 transform"""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class ONNXPredictor:
    """ONNX Runtime 推理包装器，提供与 nn.Module 兼容的调用接口。

    输出 logits 已通过烘焙在 ONNX 图中的 bias 调整（logits - bias），
    与训练时验证行为一致。
    """

    def __init__(self, session):
        self.session = session

    def __call__(self, tensor):
        """tensor: (1, 3, H, W) on CPU → logits: (1, C)"""
        np_input = tensor.cpu().numpy().astype(np.float32)
        outputs = self.session.run(None, {'input': np_input})
        return torch.from_numpy(outputs[0])

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter([torch.tensor(0.0)])


class PyTorchPredictor:
    """PyTorch 推理包装器（兜底路径），手动应用 logit bias。

    注意：ONNX 路径中 bias 已烘焙在模型图中，此处仅用于 PyTorch 回退。
    """

    def __init__(self, module, device, bias_tensor):
        self.module = module
        self._device = device
        self.bias = bias_tensor.to(device) if bias_tensor is not None else None

    def __call__(self, tensor):
        logits = self.module(tensor.to(self._device))
        if self.bias is not None:
            logits = logits - self.bias.unsqueeze(0)
        return logits.cpu()

    def eval(self):
        self.module.eval()
        return self

    def to(self, device):
        self._device = device
        self.module.to(device)
        if self.bias is not None:
            self.bias = self.bias.to(device)
        return self

    def parameters(self):
        return self.module.parameters()


def load_model(model_path=None, device=None):
    """
    加载推理模型

    优先级：INT8 ONNX → FP32 ONNX → PyTorch 权重
    返回一个可调用的 predictor（ONNXPredictor 或 PyTorchPredictor）。

    Args:
        model_path: str — 权重文件路径（仅 PyTorch 回退时使用）
        device: str — 设备（仅 PyTorch 回退时使用）

    Returns:
        predictor, device_str
    """
    cfg = CONFIG
    device = device or cfg.get("inference_device", "cpu")
    onnx_path = cfg["onnx_path"]
    int8_path = cfg["onnx_int8_path"]
    use_int8 = cfg.get("use_int8_quantization", False)

    # 显式指定 model_path → 跳过 ONNX，直接走 PyTorch 路径
    explicit_path = model_path is not None

    # 1) 尝试 INT8 ONNX（仅 model_path 未显式指定时）
    if not explicit_path and use_int8 and os.path.exists(int8_path):
        import onnxruntime as ort
        session = ort.InferenceSession(
            int8_path, providers=['CPUExecutionProvider'])
        print(f"Model loaded: {int8_path}  (ONNX INT8)")
        return ONNXPredictor(session), "cpu"

    # 2) 尝试 FP32 ONNX（仅 model_path 未显式指定时）
    if not explicit_path and os.path.exists(onnx_path):
        import onnxruntime as ort
        session = ort.InferenceSession(
            onnx_path, providers=['CPUExecutionProvider'])
        print(f"Model loaded: {onnx_path}  (ONNX FP32)")
        return ONNXPredictor(session), "cpu"

    # 3) PyTorch 回退（显式路径或 ONNX 不存在时）
    from models.weather_efficientnet import WeatherEfficientNet

    model_path = model_path or cfg["distilled_ckpt"]
    model = WeatherEfficientNet(
        model_name=cfg["student_model"],
        num_classes=cfg["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    bias = _build_bias_tensor(cfg["active_class_names"])
    print(f"Model loaded: {model_path}  (PyTorch fallback)")
    return PyTorchPredictor(model, device, bias), device


@torch.no_grad()
def predict_image(image_path, model=None, device=None):
    """
    对单张图片进行预测

    Args:
        image_path: str — 图片路径
        model: nn.Module/ONNXPredictor/PyTorchPredictor — 如为 None 则自动加载
        device: str

    Returns:
        dict: {prediction, confidence, top_k}
    """
    if model is None:
        model, device = load_model()
    elif device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    img = Image.open(image_path).convert('RGB')
    transform = _get_transform(CONFIG["img_size"])
    tensor = transform(img).unsqueeze(0)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)

    top_k = 3
    top_probs, top_indices = torch.topk(probs, top_k, dim=1)

    return {
        "prediction": CONFIG["active_class_names"][top_indices[0][0].item()],
        "confidence": top_probs[0][0].item(),
        "top_k": [
            (CONFIG["active_class_names"][idx.item()], prob.item())
            for idx, prob in zip(top_indices[0], top_probs[0])
        ],
    }


def predict_batch(image_paths, model=None, device=None):
    """
    批量预测

    Args:
        image_paths: list[str] — 图片路径列表
        model: nn.Module/ONNXPredictor/PyTorchPredictor
        device: str

    Returns:
        list[dict]: 每张图片的预测结果
    """
    if model is None:
        model, device = load_model()
    elif device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cpu"

    results = []
    for path in image_paths:
        result = predict_image(path, model, device)
        result["path"] = path
        results.append(result)

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = predict_image(image_path)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print("Top 3:")
        for name, prob in result['top_k']:
            print(f"  {name}: {prob:.4f}")
    else:
        print("Usage: python -m inference.infer <image_path>")
