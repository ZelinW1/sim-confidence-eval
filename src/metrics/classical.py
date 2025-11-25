import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import kornia.color as k_color
import kornia.filters as k_filters

from .base import BaseMetric


def _rgb_to_yiq(x: torch.Tensor) -> torch.Tensor:
    """Convert RGB tensor to YIQ.

    Accepts tensor in BCHW (or CHW) format with channels in RGB order.
    Returns tensor in same shape with channels Y, I, Q.

    If kornia provides rgb_to_yiq, prefer that. Otherwise use the
    standard linear transform (suitable when input is in 0..255 or 0..1).
    """
    # Prefer kornia implementation if available
    if hasattr(k_color, "rgb_to_yiq"):
        return k_color.rgb_to_yiq(x)

    single = False
    if x.dim() == 3:  # CHW -> BCHW
        x = x.unsqueeze(0)
        single = True

    # Expect BCHW and 3 channels
    if x.shape[1] != 3:
        raise ValueError("rgb_to_yiq expects input with 3 channels (RGB)")

    # coefficients (ITU-R BT.601 / standard YIQ conversion)
    # Y = 0.299 R + 0.587 G + 0.114 B
    # I = 0.596 R -0.274 G -0.322 B
    # Q = 0.211 R -0.523 G +0.312 B
    r = x[:, 0:1, ...]
    g = x[:, 1:2, ...]
    b = x[:, 2:3, ...]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.274 * g - 0.322 * b
    q = 0.211 * r - 0.523 * g + 0.312 * b

    out = torch.cat([y, i, q], dim=1)
    if single:
        out = out.squeeze(0)
    return out

class PSNR(BaseMetric):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        # data_range=1.0 表示输入图像是 [0, 1]
        # 使用 reduction='none' 以返回每张图的 PSNR 而不是 batch-level average
        self.metric = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=[1,2,3]).to(self.device)
        self.psnr_max = cfg.get("psnr_max", 50.0)  # 用于归一化的最大 PSNR 值
    def forward(self, preds, target):
        preds = self.prepare_data(preds)
        target = self.prepare_data(target)
        return self.metric(preds, target)
    
    def _normalize(self, x):
        # PSNR 理论上无上限，通常 30-50dB 之间较常见
        # 这里简单归一化到 [0,1]，假设最大值为 50dB
        return torch.clamp(x / self.psnr_max, 0.0, 1.0)


class SSIM(BaseMetric):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        # kernel_size=11 是 SSIM 论文的标准设置
        # 使用 reduction='none' 以返回每张图的 SSIM
        self.metric = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11, reduction='none').to(self.device)

    def forward(self, preds, target):
        preds = self.prepare_data(preds)
        target = self.prepare_data(target)
        return self.metric(preds, target)
    
    def _normalize(self, x):
        # SSIM 本身输出即在 [0,1] 范围
        return x


class FSIM(BaseMetric):
    """
    FSIM / FSIMc 实现

    实现说明：
    - 使用多尺度 Log-Gabor 谱带作为带通，结合 Riesz 变换（monogenic signal）来得到
      每尺度的偶/奇相量响应，从而计算相位一致性 (Phase Congruency, PC)，
      该方法与 Kovesi 的算法精神一致并适合用作科研复现。
    - 为保持数值语义一致性，输入在内部会被缩放回 [0,255]。
    - 支持 FSIM (亮度) 与 FSIMc (亮度+色度)。

    """
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.use_chromatic = self.cfg.get("chromatic", True)

        # FSIM 常数（保持与文献同一语义，适用于像素范围 0..255）
        self.T1 = 0.85
        self.T2 = 160.0
        self.T3 = 200.0
        self.T4 = 200.0
        self.lambda_val = 0.03

        # Phase congruency / log-gabor 参数（可通过 cfg 调整）
        self.nscale = int(self.cfg.get("nscale", 4))
        self.min_wavelength = float(self.cfg.get("min_wavelength", 3.0))
        self.mult = float(self.cfg.get("mult", 2.0))
        self.sigma_onf = float(self.cfg.get("sigma_onf", 0.55))
        self.eps = 1e-8

    def _similarity_map(self, map1, map2, C):
        return (2 * map1 * map2 + C) / (map1**2 + map2**2 + C)

    def _gradient_magnitude(self, img_gray):
        # 更稳健的 grads 访问，支持不同 kornia 版本和维度排列
        # Kornia 返回通常为 (B, C, 2, H, W)（deriv dim 在第三位），
        # 但某些版本/包装可能返回 (B, C, H, W, 2)。
        grads = k_filters.spatial_gradient(img_gray, order=1)

        if grads.ndim == 5 and grads.shape[2] == 2:
            # (B, C, 2, H, W)
            dx = grads[:, :, 0, :, :]
            dy = grads[:, :, 1, :, :]
        elif grads.ndim == 5 and grads.shape[-1] == 2:
            # (B, C, H, W, 2)
            dx = grads[..., 0]
            dy = grads[..., 1]
        elif grads.ndim == 4 and grads.shape[-1] == 2:
            # (B, C, H, 2) or (B, C, W, 2) - fallback
            dx = grads[..., 0]
            dy = grads[..., 1]
        else:
            # as a last resort, try indexing common patterns
            try:
                dx = grads[:, :, 0, ...]
                dy = grads[:, :, 1, ...]
            except Exception:
                raise RuntimeError(f"Unexpected spatial_gradient shape: {grads.shape}")

        gm = torch.sqrt(dx**2 + dy**2 + 1e-12)
        return gm

    def _phase_congruency(self, img_gray):
        """
        基于多尺度 Log-Gabor + Riesz (monogenic) 的相位一致性近似实现。
        输入: img_gray: (B,1,H,W), 值域应为 0..255
        返回: PC map (B,1,H,W)
        """
        B, C, H, W = img_gray.shape
        device = img_gray.device
        dtype = img_gray.dtype

        # 构建频域坐标 (以 cycles/image 为单位，中心化频谱)
        fy = torch.fft.fftfreq(H, d=1.0, device=device).reshape(-1, 1)  # (H,1)
        fx = torch.fft.fftfreq(W, d=1.0, device=device).reshape(1, -1)  # (1,W)
        # repeat to create full frequency grid of shape (H, W)
        u = fx.repeat(H, 1)  # (H,W)
        v = fy.repeat(1, W)  # (H,W)
        # ensure matching dtype with input to avoid unwanted upcasting
        u = u.to(dtype=dtype)
        v = v.to(dtype=dtype)
        radius = torch.sqrt(u**2 + v**2)
        radius[0,0] = 1.0  # 防止 log(0)

        # Fourier transform of the image
        f = torch.fft.fft2(img_gray.squeeze(1))  # (B,H,W), complex

        # Riesz transform frequency components
        KR = radius.clone()
        KR[0,0] = 1.0
        Kx = u
        Ky = v
        riesz_x = -1j * (Kx / KR)
        riesz_y = -1j * (Ky / KR)

        # accumulate across scales
        sum_amplitude = torch.zeros((B, H, W), device=device, dtype=dtype)
        sum_even = torch.zeros((B, H, W), device=device, dtype=torch.cfloat)
        sum_odd_x = torch.zeros((B, H, W), device=device, dtype=torch.cfloat)
        sum_odd_y = torch.zeros((B, H, W), device=device, dtype=torch.cfloat)

        for s in range(self.nscale):
            wavelength = self.min_wavelength * (self.mult ** s)
            fo = 1.0 / wavelength

            # log-Gabor radial component
            log_rad = torch.log(radius / fo)
            log_gabor = torch.exp((- (log_rad ** 2)) / (2 * (torch.log(torch.tensor(self.sigma_onf, device=device)) ** 2)))
            log_gabor[0,0] = 0.0  # no DC

            LG = log_gabor.unsqueeze(0).repeat(B, 1, 1)  # (B,H,W)
            resp_even = torch.fft.ifft2(f * LG)

            resp_odd_x = torch.fft.ifft2(f * LG * riesz_x)
            resp_odd_y = torch.fft.ifft2(f * LG * riesz_y)

            amp = torch.sqrt((resp_even.real ** 2) + (resp_odd_x.real ** 2) + (resp_odd_y.real ** 2) + self.eps)

            sum_amplitude = sum_amplitude + amp
            sum_even = sum_even + resp_even
            sum_odd_x = sum_odd_x + resp_odd_x
            sum_odd_y = sum_odd_y + resp_odd_y

        energy = torch.sqrt((sum_even.real ** 2) + (sum_odd_x.real ** 2) + (sum_odd_y.real ** 2) + self.eps)

        # Take per-batch maximum over spatial dims by flattening spatial dims.
        T = 0.0001 * torch.max(sum_amplitude.view(B, -1), dim=1).values.view(B, 1, 1)

        pc = (energy - T) / (sum_amplitude + self.eps)
        pc = torch.clamp(pc, min=0.0, max=1.0)

        return pc.unsqueeze(1)  # (B,1,H,W)

    def forward(self, preds, target):
        preds = self.prepare_data(preds)
        target = self.prepare_data(target)

        # FSIM 经验常数基于 0..255 量化，因此在内部将值域还原
        preds_255 = preds * 255.0
        target_255 = target * 255.0

        if preds_255.shape[1] != 3:
            raise ValueError("FSIM expects 3-channel RGB input")

        # RGB -> YIQ 
        pred_yiq = _rgb_to_yiq(preds_255)
        target_yiq = _rgb_to_yiq(target_255)
        pred_y, pred_i, pred_q = torch.split(pred_yiq, 1, dim=1)
        target_y, target_i, target_q = torch.split(target_yiq, 1, dim=1)

        # phase congruency 使用多尺度 Log-Gabor + Riesz
        pc_p = self._phase_congruency(pred_y)
        pc_t = self._phase_congruency(target_y)

        # gradient magnitude 仍作为 GM 特征
        gm_p = self._gradient_magnitude(pred_y)
        gm_t = self._gradient_magnitude(target_y)

        # 相似度分量
        S_pc = self._similarity_map(pc_p, pc_t, self.T1)
        S_gm = self._similarity_map(gm_p, gm_t, self.T2)
        S_L = S_pc * (S_gm ** self.lambda_val)

        PC_m = torch.max(pc_p, pc_t)

        if self.use_chromatic:
            S_i = self._similarity_map(pred_i, target_i, self.T3)
            S_q = self._similarity_map(pred_q, target_q, self.T4)
            S_final = S_L * S_i * S_q
        else:
            S_final = S_L

        numerator = torch.sum(S_final * PC_m, dim=[1,2,3])
        denominator = torch.sum(PC_m, dim=[1,2,3])
        score = numerator / (denominator + 1e-8)
        return score
    
    def _normalize(self, x):
        # FSIM 本身输出即在 [0,1] 范围
        return x