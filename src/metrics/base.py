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

    def prepare_data(self, x):
        """
        统一的数据预处理，例如确保维度是 BCHW，值域是 0-1
        """
        if x.dim() == 3: # CHW -> BCHW
            x = x.unsqueeze(0)
        # 这里假设输入已经是 Tensor 且归一化到了 [0,1]
        # 如果需要特定算法的特殊归一化（如 [-1,1]），可在子类重写
        return x.to(self.device)
    
    @abstractmethod
    def _normalize(self, x):
        """
        将输出最后指标归一化到 [0, 1] 范围内，便于比较
        """
        pass
        