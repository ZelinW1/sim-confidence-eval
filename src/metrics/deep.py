import torch
import torch.nn as nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance

from .base import BaseMetric

class LPIPS(BaseMetric):
    """
    Learned Perceptual Image Patch Similarity (LPIPS)
    Lower is better (0.0 = same).
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)
        # 获取配置参数
        net_type = self.cfg.get("net", "alex")  # 可选: alex, vgg, squeeze
        # 默认返回 per-sample 分数；如果需要全局平均可以在 cfg 中指定 reduction='mean'
        reduction = self.cfg.get("reduction", "none")
        normalize = self.cfg.get("normalize", True)

        # 初始化 torchmetrics 的 LPIPS
        # 参数包括 net_type, reduction, normalize 等。这里通过 cfg 控制这些参数。
        self.metric = LearnedPerceptualImagePatchSimilarity(
            net_type=net_type,
            reduction=reduction,
            normalize=normalize,
        )
        # 将内部 metric 移到与 BaseMetric 相同的 device，避免 device mismatch
        try:
            self.metric = self.metric.to(self.device)
        except Exception:
            # 某些 torchmetrics 版本的 metric 可能不是 nn.Module 或者不支持 to(); ignore if so
            pass
        
        # 标记为局部指标（每张图都能算出一个分）
        self.is_global = False 
        self.lpips_max = self.cfg.get("lpips_max", 1.0)  # 用于归一化的最大 LPIPS 值

    def forward(self, preds, target):
        """
        :param preds: (B, C, H, W) range [0, 1]
        :param target: (B, C, H, W) range [0, 1]
        """
        preds, target = self.prepare_data(preds), self.prepare_data(target)
        
        # LPIPS 计算量大，确保在 eval 模式且无梯度，节省显存
        with torch.no_grad():
            score = self.metric(preds, target)
            
        return score
    
    def _normalize(self, x):
        # LPIPS 理论上无上限，根据经验值做缩放反转
        return 1.0 - torch.clamp(x / self.lpips_max, 0.0, 1.0)


class FID(BaseMetric):
    """
    Fréchet Inception Distance (FID)
    衡量生成数据(Sim)与真实数据(Real)的分布距离。
    Lower is better.
    
    注意：FID 是 Global Metric，不能对单张图片评分，必须跑完整个 Batch/Dataset 才能调用 compute()。
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)
        feature = self.cfg.get("feature", 2048) # InceptionV3 特征维度: 64, 192, 768, 2048
        
        # reset_real_features=False 表示 Real 数据的统计量可以累积
        # normalize=True: torchmetrics 期望 float 输入为 [0, 1]
        self.metric = FrechetInceptionDistance(
            feature=feature, 
            normalize=True,
            reset_real_features=True
        )
        try:
            self.metric = self.metric.to(self.device)
        except Exception:
            pass
        
        # 标记为全局指标（Evaluator 需要在循环结束后调用 .compute()）
        self.is_global = True 

    def forward(self, preds, target=None):
        """
        FID 的 forward 仅用于更新统计量 (Update)，不返回具体分数。
        为了兼容 Pipeline 接口，返回 -1.0 或 None。
        
        :param preds: Sim 图像 batch, [0, 1]
        :param target: Real 图像 batch, [0, 1]。如果是 Unpaired 模式，Target 可以是一批随机的 Real 图。
        """
        preds = self.prepare_data(preds)
        
        # 更新 Sim 分布统计
        self.metric.update(preds, real=False)
        
        # 更新 Real 分布统计
        # 如果是 Paired 数据集，每次这里都会传 target；
        # 如果是 Unpaired，这里也会接收到一批 Real 数据。
        if target is not None:
            target = self.prepare_data(target)
            self.metric.update(target, real=True)
        
        return torch.tensor(-1.0) # Dummy value

    def compute(self):
        """
        计算最终的 FID 分数
        """
        return self.metric.compute()
    
    def reset(self):
        self.metric.reset()

    def _normalize(self, x):
        # FID 理论上无上限，根据经验值做缩放反转
        # 暂不实现
        return -1.0