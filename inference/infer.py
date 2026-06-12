# ============ inference/infer.py ============
"""单张/批量推理脚本"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

from config import CONFIG
from models.weather_efficientnet import WeatherEfficientNet


def load_model(model_path=None, device=None):
    """
    加载训练好的模型

    Args:
        model_path: str — 权重文件路径
        device: str — 设备

    Returns:
        tuple: (model, device)
    """
    cfg = CONFIG
    model_path = model_path or cfg["distilled_ckpt"]
    device = device or cfg.get("inference_device", "cpu")

    model = WeatherEfficientNet(
        model_name=cfg["student_model"],
        num_classes=cfg["num_classes"],
        pretrained=False,
    )
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    return model, device


def get_transform(img_size=380):
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


@torch.no_grad()
def predict_image(image_path, model=None, device=None):
    """
    对单张图片进行预测

    Args:
        image_path: str — 图片路径
        model: nn.Module — 如为 None 则自动加载
        device: str

    Returns:
        dict: {prediction, confidence, top_k}
    """
    if model is None:
        model, device = load_model()
    elif device is None:
        device = next(model.parameters()).device

    img = Image.open(image_path).convert('RGB')
    transform = get_transform(CONFIG["img_size"])
    tensor = transform(img).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)

    top_k = 3
    top_probs, top_indices = torch.topk(probs, top_k, dim=1)

    return {
        "prediction": CONFIG["class_names"][top_indices[0][0].item()],
        "confidence": top_probs[0][0].item(),
        "top_k": [
            (CONFIG["class_names"][idx.item()], prob.item())
            for idx, prob in zip(top_indices[0], top_probs[0])
        ],
    }


def predict_batch(image_paths, model=None, device=None):
    """
    批量预测

    Args:
        image_paths: list[str] — 图片路径列表
        model: nn.Module
        device: str

    Returns:
        list[dict]: 每张图片的预测结果
    """
    if model is None:
        model, device = load_model()
    elif device is None:
        device = next(model.parameters()).device

    transform = get_transform(CONFIG["img_size"])
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
