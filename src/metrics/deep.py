import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance

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
        
        # 标记为非局部指标
        self.is_global = False 
        self.need_normalize_result = self.cfg.get("need_normalize_result", True)
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
        
        if self.need_normalize_result:
            return score, self._normalize(score)
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
        FID 不支持单次计算，需要累积特征后计算。
        """
        raise NotImplementedError("FID metric does not support single forward computation. Use update() and compute() instead.")       
    
    def update(self, preds, real=False):
        """
        用于累积批次结果，适用于全局指标（如 FID）
        :param preds: 图像 batch, [0, 1]
        :param real: 是否为真实数据
        """
        preds = self.prepare_data(preds)
        # 确保内部 metric（例如 torchmetrics 的 Inception 特征提取器）被迁移到与输入相同的设备
        # 某些 torchmetrics 实现会延迟构建内部模型或在 to() 调用后未正确迁移子模块，
        # 因此在 update 前显式将 metric 移到 preds.device
        try:
            self.metric = self.metric.to(preds.device)
        except Exception:
            # 忽略不能 to() 的情况
            pass

        self.metric.update(preds, real=real)

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


class KID(BaseMetric):
    """
    Kernel Inception Distance (KID)
    衡量生成数据(Sim)与真实数据(Real)的分布距离。
    Lower is better.

    注意：KID 是 Global Metric，不能对单张图片评分，必须跑完整个 Batch/Dataset 才能调用 compute()。
    torchmetrics 的 KID 默认返回 (mean, std)，本封装的 compute() 默认返回 mean，
    并将 std 缓存在 self.last_kid_std 中，便于日志或后处理使用。
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)

        feature = self.cfg.get("feature", 2048) # InceptionV3 特征维度: 64, 192, 768, 2048
        subsets = self.cfg.get("subsets", 100)
        subset_size = self.cfg.get("subset_size", 1000)
        degree = self.cfg.get("degree", 3)
        gamma = self.cfg.get("gamma", None)
        coef = self.cfg.get("coef", 1.0)

        # normalize=True: torchmetrics 期望 float 输入为 [0, 1]
        # reset_real_features=True: 每次评估 run 都重新统计 real 特征
        self.metric = KernelInceptionDistance(
            feature=feature,
            subsets=subsets,
            subset_size=subset_size,
            degree=degree,
            gamma=gamma,
            coef=coef,
            reset_real_features=True,
            normalize=True,
        )
        try:
            self.metric = self.metric.to(self.device)
        except Exception:
            pass

        self.is_global = True
        self.last_kid_std = None

    def forward(self, preds, target=None):
        """
        KID 不支持单次计算，需要累积特征后计算。
        """
        raise NotImplementedError("KID metric does not support single forward computation. Use update() and compute() instead.")

    def update(self, preds, real=False):
        """
        用于累积批次结果，适用于全局指标（如 KID）
        :param preds: 图像 batch, [0, 1]
        :param real: 是否为真实数据
        """
        preds = self.prepare_data(preds)
        try:
            self.metric = self.metric.to(preds.device)
        except Exception:
            pass

        self.metric.update(preds, real=real)

    def compute(self):
        """
        计算最终的 KID 分数（返回 mean）
        """
        kid_mean, kid_std = self.metric.compute()
        self.last_kid_std = kid_std
        return kid_mean

    def compute_with_std(self):
        """
        计算最终的 KID 分数并返回 (mean, std)
        """
        kid_mean, kid_std = self.metric.compute()
        self.last_kid_std = kid_std
        return kid_mean, kid_std

    def reset(self):
        self.metric.reset()
        self.last_kid_std = None

    def _normalize(self, x):
        # KID 理论上无上限，根据经验值做缩放反转
        # 暂不实现
        return -1.0


class MKMMD(BaseMetric):
    """
    Multi-Kernel Maximum Mean Discrepancy (MK-MMD, RBF kernels)
    使用 Inception-v3 特征并基于多带宽 RBF 核估计分布差异。
    Lower is better.

    注意：MKMMD 是 Global Metric，不能对单张图片评分，必须先 update() 累积再 compute()。
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)

        # 与 DA 论文常见配置一致的多核 RBF 参数
        self.kernel_mul = float(self.cfg.get("kernel_mul", 2.0))
        self.kernel_num = int(self.cfg.get("kernel_num", 5))
        self.unbiased = bool(self.cfg.get("unbiased", True))
        self.fix_sigma = self.cfg.get("fix_sigma", None)
        self.max_features = self.cfg.get("max_features", None)

        self.inception = self._build_inception_backbone()
        try:
            self.inception = self.inception.to(self.device)
        except Exception:
            pass

        self._real_features = []
        self._sim_features = []
        self.is_global = True

    def _build_inception_backbone(self):
        # 采用标准 ImageNet 预训练权重，提取 pool 前 2048-d 特征
        weights = Inception_V3_Weights.IMAGENET1K_V1
        model = inception_v3(weights=weights, aux_logits=False)
        model.fc = nn.Identity()
        model.eval()
        return model

    def forward(self, preds, target=None):
        """
        MKMMD 不支持单次计算，需要累积特征后计算。
        """
        raise NotImplementedError("MKMMD metric does not support single forward computation. Use update() and compute() instead.")

    def _extract_features(self, imgs):
        """
        输入 imgs: [B, C, H, W], range [0, 1]
        输出特征: [B, 2048]
        """
        if imgs.size(1) == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        elif imgs.size(1) > 3:
            imgs = imgs[:, :3, :, :]

        # Inception-v3 标准输入是 299x299 + ImageNet 归一化
        imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 3, 1, 1)
        imgs = (imgs - mean) / std

        with torch.no_grad():
            feats = self.inception(imgs)

        if feats.dim() > 2:
            feats = torch.flatten(feats, 1)
        return feats

    def update(self, preds, real=False):
        """
        累积批次特征，适用于全局指标。
        :param preds: 图像 batch, [0, 1]
        :param real: 是否为真实数据
        """
        preds = self.prepare_data(preds)
        try:
            self.inception = self.inception.to(preds.device)
        except Exception:
            pass

        feats = self._extract_features(preds).detach().cpu()

        if real:
            self._real_features.append(feats)
        else:
            self._sim_features.append(feats)

    def _pairwise_sqdist(self, x, y):
        x2 = (x ** 2).sum(dim=1, keepdim=True)
        y2 = (y ** 2).sum(dim=1, keepdim=True).transpose(0, 1)
        dist2 = x2 + y2 - 2.0 * (x @ y.transpose(0, 1))
        return torch.clamp(dist2, min=0.0)

    def _build_bandwidths(self, x, y):
        if self.fix_sigma is not None:
            if isinstance(self.fix_sigma, (list, tuple)):
                vals = [float(v) for v in self.fix_sigma]
                return x.new_tensor(vals)
            return x.new_tensor([float(self.fix_sigma)])

        dist_xy = self._pairwise_sqdist(x, y)
        base_sigma = torch.median(dist_xy)
        base_sigma = torch.clamp(base_sigma, min=1e-12)

        center = self.kernel_num // 2
        sigmas = [base_sigma * (self.kernel_mul ** (i - center)) for i in range(self.kernel_num)]
        return torch.stack(sigmas)

    def _rbf_kernel(self, dist2, sigmas):
        # K = mean_i exp(-d^2 / (2 * sigma_i))
        sigma = sigmas.view(-1, 1, 1)
        kernels = torch.exp(-dist2.unsqueeze(0) / (2.0 * sigma + 1e-12))
        return kernels.mean(dim=0)

    def _mkmmd(self, x, y):
        n, m = x.size(0), y.size(0)
        if n < 2 or m < 2:
            raise ValueError("MKMMD requires at least 2 samples in both sim and real feature sets.")

        d_xx = self._pairwise_sqdist(x, x)
        d_yy = self._pairwise_sqdist(y, y)
        d_xy = self._pairwise_sqdist(x, y)

        sigmas = self._build_bandwidths(x, y)
        k_xx = self._rbf_kernel(d_xx, sigmas)
        k_yy = self._rbf_kernel(d_yy, sigmas)
        k_xy = self._rbf_kernel(d_xy, sigmas)

        if self.unbiased:
            sum_xx = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
            sum_yy = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (m * (m - 1))
        else:
            sum_xx = k_xx.mean()
            sum_yy = k_yy.mean()

        sum_xy = k_xy.mean()
        return sum_xx + sum_yy - 2.0 * sum_xy

    def compute(self):
        """
        计算最终 MK-MMD 分数。
        """
        if len(self._sim_features) == 0 or len(self._real_features) == 0:
            raise ValueError("MKMMD has no accumulated features. Call update() for both sim and real batches before compute().")

        sim = torch.cat(self._sim_features, dim=0).to(self.device)
        real = torch.cat(self._real_features, dim=0).to(self.device)

        if self.max_features is not None:
            max_n = int(self.max_features)
            if sim.size(0) > max_n:
                idx = torch.randperm(sim.size(0), device=sim.device)[:max_n]
                sim = sim[idx]
            if real.size(0) > max_n:
                idx = torch.randperm(real.size(0), device=real.device)[:max_n]
                real = real[idx]

        return self._mkmmd(sim, real)

    def reset(self):
        self._sim_features = []
        self._real_features = []

    def _normalize(self, x):
        # MK-MMD 理论上无上限，根据经验值做缩放反转
        # 暂不实现
        return -1.0