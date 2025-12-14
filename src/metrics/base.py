import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseMetric(nn.Module, ABC):
    """
    评价指标基类，继承自 nn.Module 以支持 GPU 和梯度传播
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or {}
        # 默认设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = self.__class__.__name__

    @abstractmethod
    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        核心计算逻辑
        :param preds: 仿真图像/预测图像 (B, C, H, W), 值域 [0, 1]
        :param target: 真实图像/参考图像 (B, C, H, W), 值域 [0, 1]
        :return: scalar or tensor of shape (B,)
        """
        pass

    def update(self, *args, **kwargs):
        """
        用于累积批次结果，适用于全局指标（如 FID）
        默认不实现，子类可重写
        """
        pass

    def prepare_data(self, x):
        """
        统一的数据预处理，例如确保维度是 BCHW，值域是 0-1
        """
        if x.dim() == 3: # CHW -> BCHW
            x = x.unsqueeze(0)
        # 这里假设输入已经是 Tensor 且归一化到了 [0,1]
        # 为了确保数据被移动到 metric 实际所在的设备（在外部可能调用了 .to(device)），
        # 优先尝试从 module 的参数或 buffers 获取 device；如果没有参数/缓冲区，则回退到
        # 初始化时的 self.device 配置。
        try:
            dev = next(self.parameters()).device
        except StopIteration:
            try:
                dev = next(self.buffers()).device
            except StopIteration:
                dev = self.device

        return x.to(dev)
    
    @abstractmethod
    def _normalize(self, x):
        """
        将输出最后指标归一化到 [0, 1] 范围内，便于比较
        """
        pass
        