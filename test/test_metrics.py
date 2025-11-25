import torch
import sys
import os

print(torch.__version__)
print(torch.cuda.is_available())

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from metrics.classical import PSNR, SSIM, FSIM

print("--- Testing Classical Metrics ---")
# 模拟数据: Batch=2, 3通道, 256x256
sim_img = torch.rand(8, 3, 256, 256)
real_img = torch.rand(8, 3, 256, 256) # 随机噪声

# 1. 测试 PSNR
psnr_metric = PSNR()
score_psnr = psnr_metric(sim_img, real_img)
print(f"PSNR Score: {score_psnr}")

# 2. 测试 SSIM
ssim_metric = SSIM()
score_ssim = ssim_metric(sim_img, real_img)
print(f"SSIM Score: {score_ssim}")

# 3. 测试 FSIM
fsim_metric = FSIM(cfg={'chromatic': True})
score_fsim = fsim_metric(sim_img, real_img)
print(f"FSIM Score: {score_fsim}")

from metrics.deep import LPIPS, FID

print("\n--- Testing Deep Metrics ---")
# 模拟数据 B=4, C=3, H=128, W=128 (注意 FID 需要至少一定数量样本才能算得准，这里仅跑通流程)
sim_img = torch.rand(32, 3, 128, 128)
real_img = torch.rand(32, 3, 128, 128)

# 1. 测试 LPIPS
# 第一次运行会自动下载 VGG 权重
lpips_metric = LPIPS(cfg={'net': 'vgg', 'reduction': 'none', 'normalize': True})
score_lpips = lpips_metric(sim_img, real_img)
print(f"LPIPS Score: {score_lpips}")

# 2. 测试 FID
fid_metric = FID(cfg={'feature': 2048}) 
fid_metric(sim_img, real_img) # Update step

final_fid = fid_metric.compute()
print(f"FID Score: {final_fid.item():.4f}")