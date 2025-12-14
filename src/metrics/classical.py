import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from piq import fsim as piq_fsim
import numpy as np
import math

from .base import BaseMetric


class PSNR(BaseMetric):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        # data_range=1.0 表示输入图像是 [0, 1]
        # 使用 reduction='none' 以返回每张图的 PSNR 而不是 batch-level average
        self.metric = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=[1,2,3]).to(self.device)
        self.need_normalize_result = self.cfg.get("need_normalize_result", True)
        self.psnr_max = cfg.get("psnr_max", 50.0)  # 用于归一化的最大 PSNR 值
    def forward(self, preds, target):
        preds = self.prepare_data(preds)
        target = self.prepare_data(target)

        ori_result = self.metric(preds, target)  # (B,)
        if self.need_normalize_result:
            return ori_result, self._normalize(ori_result)

        return ori_result
    
    def _normalize(self, x):
        # PSNR 理论上无上限，通常 30-50dB 之间较常见
        return torch.clamp(x / self.psnr_max, 0.0, 1.0)


class SSIM(BaseMetric):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        # kernel_size=11 是 SSIM 论文的标准设置
        # 使用 reduction='none' 以返回每张图的 SSIM
        self.metric = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11, reduction='none').to(self.device)
        self.need_normalize_result = self.cfg.get("need_normalize_result", True)

    def forward(self, preds, target):
        preds = self.prepare_data(preds)
        target = self.prepare_data(target)
        ori_result = self.metric(preds, target)
        if self.need_normalize_result:
            return ori_result, self._normalize(ori_result)
        return ori_result
    
    def _normalize(self, x):
        # SSIM 本身输出即在 [0,1] 范围
        return x
    

class MS_SSIM(BaseMetric):
    def __init__(self, cfg=None):
        super().__init__(cfg)

        self.metric = MultiScaleStructuralSimilarityIndexMeasure(
            data_range=1.0,
            kernel_size=11,
            reduction='none'
        ).to(self.device)

        self.scales = int(self.cfg.get("scales", 5))
        self.need_normalize_result = self.cfg.get("need_normalize_result", True)

    def forward(self, preds, target):
        preds = self.prepare_data(preds)
        target = self.prepare_data(target)

        ms_ssim_score = self.metric(preds, target)

        if self.need_normalize_result:
            return ms_ssim_score, self._normalize(ms_ssim_score)
        return ms_ssim_score
    
    def _normalize(self, x):
        # MS-SSIM 本身输出即在 [0,1] 范围
        return x


class FSIM(BaseMetric):
    """
    FSIM / FSIMc 实现（使用 piq 库）

    使用 PyTorch Image Quality (piq) 库提供的 FSIM 实现，这是一个经过充分测试和
    验证的研究级实现，与原始 MATLAB 参考实现保持一致。

    参数说明：
    - chromatic: 是否使用 FSIMc（亮度+色度）模式，默认 True
    - scales: 频率带数，默认 4
    - orientations: 方向数，默认 4
    - min_length: 最小波长，默认 6
    - mult: 波长乘数，默认 2
    - sigma_f: 对数-Gabor 带宽，默认 0.55
    - delta_theta: 方向传播函数的标准差因子，默认 1.2
    - k: 相位一致性阈值参数，默认 2.0
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.chromatic = self.cfg.get("chromatic", True)
        self.scales = int(self.cfg.get("scales", 4))
        self.orientations = int(self.cfg.get("orientations", 4))
        self.min_length = int(self.cfg.get("min_length", 6))
        self.mult = int(self.cfg.get("mult", 2))
        self.sigma_f = float(self.cfg.get("sigma_f", 0.55))
        self.delta_theta = float(self.cfg.get("delta_theta", 1.2))
        self.k = float(self.cfg.get("k", 2.0))
        
        self.need_normalize_result = self.cfg.get("need_normalize_result", False)

    def forward(self, preds, target):
        preds = self.prepare_data(preds)
        target = self.prepare_data(target)

        # Clamp to [0, 1] to handle floating point precision issues
        preds = torch.clamp(preds, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        # piq.fsim 接受 [0, 1] 范围的输入，使用 reduction='none' 返回每个样本的分数
        fsim_score = piq_fsim(
            preds, 
            target,
            reduction='none',
            data_range=1.0,
            chromatic=self.chromatic,
            scales=self.scales,
            orientations=self.orientations,
            min_length=self.min_length,
            mult=self.mult,
            sigma_f=self.sigma_f,
            delta_theta=self.delta_theta,
            k=self.k
        )

        if self.need_normalize_result:
            return fsim_score, self._normalize(fsim_score)
        return fsim_score
    
    def _normalize(self, x):
        # FSIM 本身输出即在 [0,1] 范围
        return x