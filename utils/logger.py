"""训练日志工具：控制台输出 + TensorBoard（可选）"""
import os
import time
from datetime import datetime


class TrainLogger:
    """简单的训练日志记录器，支持 TensorBoard SummaryWriter"""

    def __init__(self, log_dir=None, use_tb=False):
        self.use_tb = use_tb
        self.writer = None

        if use_tb:
            from torch.utils.tensorboard import SummaryWriter
            if log_dir is None:
                log_dir = f"results/tb_results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)

        self.start_time = time.time()
        self.metrics_history = {}

    def log_scalar(self, tag, value, step):
        """记录标量值（TensorBoard）"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_metrics(self, phase, metrics, epoch):
        """
        记录一个 epoch 的指标集合

        Args:
            phase: str — 'train' / 'val' / 'test'
            metrics: dict — 指标字典
            epoch: int — 当前 epoch
        """
        key = f"{phase}_{epoch}"
        self.metrics_history[key] = metrics

        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"{phase}/{name}", value, epoch)

    def elapsed(self):
        """返回已用时间（秒）"""
        return time.time() - self.start_time

    def close(self):
        if self.writer:
            self.writer.close()
