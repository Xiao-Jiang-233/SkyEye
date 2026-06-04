"""训练日志工具：控制台输出 + TensorBoard（可选）"""
import os
import time
import warnings
from datetime import datetime


class TrainLogger:
    """简单的训练日志记录器，支持 TensorBoard SummaryWriter

    当 tensorboard 包未安装时，use_tb=True 会优雅降级（打印警告并禁用 TB），
    不会中断训练。
    """

    def __init__(self, log_dir=None, use_tb=False):
        self.use_tb = use_tb
        self.writer = None
        self._tb_available = False

        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_available = True
            except ImportError:
                warnings.warn(
                    "TensorBoard 日志已启用，但 tensorboard 包未安装。"
                    "请运行: pip install tensorboard。"
                    "本次训练将跳过 TensorBoard 记录。"
                )
                self.use_tb = False

        if self._tb_available:
            if log_dir is None:
                log_dir = "results/tb_results"
            # 在指定目录下创建时间戳子目录，避免重复运行日志混叠
            log_dir = os.path.join(log_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
            os.makedirs(log_dir, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard 日志目录: {log_dir}")

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

    def flush(self):
        """强制将缓冲区写入磁盘（防止数据丢失）"""
        if self.writer:
            self.writer.flush()

    def close(self):
        if self.writer:
            self.writer.flush()
            self.writer.close()
