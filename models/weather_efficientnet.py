# ============ models/weather_efficientnet.py ============
"""EfficientNet 天气分类模型，支持中间层特征提取用于知识蒸馏"""
import timm
import torch.nn as nn
import torch.nn.functional as F


class WeatherEfficientNet(nn.Module):
    """
    EfficientNet 天气分类器

    使用 timm 库加载预训练 EfficientNet，去除原始分类头，
    替换为自定义分类器。通过 forward hook 捕获中间层特征，
    用于知识蒸馏中的特征对齐。

    Args:
        model_name: str — timm 模型名 (e.g. "efficientnet_b0", "efficientnet_b5")
        num_classes: int — 分类类别数
        pretrained: bool — 是否加载 ImageNet 预训练权重
    """

    def __init__(self, model_name="efficientnet_b0", num_classes=6, pretrained=True):
        super().__init__()
        # 加载 timm EfficientNet，去掉分类头，保留空间特征
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # 去掉分类头
            global_pool='',      # 保留空间特征图，不做全局池化
            features_only=False,
        )

        # 获取 backbone 输出通道数 (B0=1280, B5=2048)
        self.num_features = self.backbone.num_features

        # 自定义分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.num_features, 512),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        # 中间层特征存储
        self.intermediate_features = {}

        # 注册 hook 捕获各 stage 输出
        self._register_hooks()

    def _register_hooks(self):
        """注册 forward hook 收集中间层特征"""

        def hook_fn(name):
            def fn(_, __, output):
                self.intermediate_features[name] = output
            return fn

        for i, block in enumerate(self.backbone.blocks):
            block.register_forward_hook(hook_fn(f"stage_{i}"))

    def forward(self, x, return_features=False):
        """
        前向传播

        Args:
            x: Tensor (B,3,H,W)
            return_features: bool — 是否返回中间层特征（KD 时需要）

        Returns:
            if return_features:
                (logits, intermediate_features_dict)
            else:
                logits
        """
        self.intermediate_features = {}

        # Backbone 前向
        x = self.backbone.forward_features(x)

        # 全局平均池化
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # 分类
        logits = self.classifier(x)

        if return_features:
            return logits, self.intermediate_features
        return logits

    def get_stage_channels(self):
        """
        返回各 stage 的输出通道数，用于特征蒸馏时的投影层配置

        优先使用 timm 内置的 feature_info（兼容各版本 timm），
        回退到手动遍历 blocks 的 conv_pwl/conv_pw。

        Returns:
            list[int]: 各 stage 的输出通道数（去重后）
        """
        # 优先使用 timm 内置元数据（timm 1.0.27+ features_only=False 时为 list）
        if hasattr(self.backbone, 'feature_info') and self.backbone.feature_info:
            info = self.backbone.feature_info
            if hasattr(info, 'channels'):
                return info.channels()
            if isinstance(info, list) and 'num_chs' in info[0]:
                channels = []
                for entry in info:
                    if entry['num_chs'] not in channels:
                        channels.append(entry['num_chs'])
                return channels

        # 回退：手动遍历 blocks（兼容旧版 timm）
        channels = []
        prev_channels = None

        for block in self.backbone.blocks:
            ch = None
            # timm 1.0.27 中 blocks 是 Sequential，需进入内部找 DepthwiseSeparableConv
            if hasattr(block, '__iter__') and not hasattr(block, 'conv_pwl'):
                for inner in block:
                    if hasattr(inner, 'conv_pwl'):
                        ch = inner.conv_pwl.out_channels
                    elif hasattr(inner, 'conv_pw'):
                        ch = inner.conv_pw.out_channels
                    if ch is not None:
                        break
            else:
                if hasattr(block, 'conv_pwl'):
                    ch = block.conv_pwl.out_channels
                elif hasattr(block, 'conv_pw'):
                    ch = block.conv_pw.out_channels

            if ch is None:
                continue
            if ch != prev_channels:
                channels.append(ch)
                prev_channels = ch

        return channels
