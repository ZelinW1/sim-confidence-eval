"""
该脚本用于计算 FID 的"最好"和"最差"基线，用于归一化 FID 分数到 [0, 1] 范围：
    S_fid = 1 对应"最好"（尽可能接近参考集）
    S_fid = 0 对应"最差"（在可接受定义下的最不像）

最好 / 最差 基线定义方法：
    FID_best（上界）：计算参考集随机切分为两半（多次随机切分），得到若干 FID(ref_A, ref_B)。
                     取这些值的中位数作为 FID_best
    FID_worst（下界）：计算参考集与一个完全不相关的图像集（CIFAR-10 或 ImageNet）之间的 FID，
                      作为 FID_worst
"""

import os
import sys
import yaml
import torch
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from PIL import Image
import torchvision.datasets as datasets

from src.metrics.deep import FID

# 配置日志
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# 路径和参数配置
config_path = "configs/basic_eval.yaml"
ref_dataset_path = "./data/global/real"
random_seed = 42
output_txt = "fid_baselines.txt"
image_net_path = "./data/image_net"

# FID_best 计算参数
num_splits = 10  # 随机切分次数

# FID_worst 计算参数：使用 CIFAR-10 作为独立数据集（更小更快），或可改为 ImageNet
use_cifar10 = False  # 如果 True 使用 CIFAR-10；如果 False 使用 ImageNet（需要更多时间）

# 设置随机种子
np.random.seed(random_seed)
torch.manual_seed(random_seed)

class SimpleImageDataset(Dataset):
    """加载指定目录中的所有图片"""
    def __init__(self, root_path, image_size=(256, 256)):
        self.root_path = Path(root_path)
        self.image_size = image_size
        
        # 支持的图片格式
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        self.image_paths = []
        
        # 递归遍历所有子目录
        for ext in extensions:
            self.image_paths.extend(self.root_path.rglob(f'*{ext}'))
            self.image_paths.extend(self.root_path.rglob(f'*{ext.upper()}'))
        
        self.image_paths = list(set(self.image_paths))  # 去重
        logger.info(f"Found {len(self.image_paths)} images in {root_path}")
        
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, Image.Resampling.LANCZOS),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img)
            return tensor
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            # 返回随机张量作为备选
            return torch.randn(3, *self.image_size)

def load_config(config_path):
    """加载 YAML 配置文件"""
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using defaults")
        return {'data': {'image_size': [256, 256]}}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def compute_fid(dataset1, dataset2, batch_size=64, device='cuda'):
    """计算两个数据集之间的 FID"""
    fid_metric = FID().to(device)
    
    loader1 = DataLoader(dataset1, batch_size=batch_size, shuffle=False, num_workers=0)
    loader2 = DataLoader(dataset2, batch_size=batch_size, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        # 数据集数量以较小的为准
        num_batches = min(len(loader1), len(loader2))
        for i, (data1, data2) in enumerate(zip(loader1, loader2)):
            if i >= num_batches:
                break
            data1 = data1.to(device)
            data2 = data2.to(device)
            fid_metric.update(data1, real=False)
            fid_metric.update(data2, real=True)
    
    fid_value = fid_metric.compute()
    fid_metric.reset()
    
    return fid_value.item()

def compute_fid_best(ref_dataset, num_splits=10, batch_size=64, device='cuda'):
    """
    计算参考集的 FID_best：
    随机切分参考集为两半多次，计算各切分的 FID，取中位数
    """
    fid_values = []
    dataset_size = len(ref_dataset)
    
    logger.info(f"Computing FID_best with {num_splits} random splits...")
    
    for split_idx in range(num_splits):
        # 随机切分
        indices = np.arange(dataset_size)
        np.random.shuffle(indices)
        
        split_point = dataset_size // 2
        indices_a = indices[:split_point]
        indices_b = indices[split_point:]
        
        # 创建子集
        subset_a = Subset(ref_dataset, indices_a)
        subset_b = Subset(ref_dataset, indices_b)
        
        # 计算 FID
        fid_val = compute_fid(subset_a, subset_b, batch_size=batch_size, device=device)
        fid_values.append(fid_val)
        logger.info(f"  Split {split_idx + 1}/{num_splits}: FID = {fid_val:.4f}")
    
    fid_best = np.median(fid_values)
    logger.info(f"FID_best (median of {num_splits} splits): {fid_best:.4f}")
    
    return fid_best, fid_values

def compute_fid_worst(ref_dataset, batch_size=64, device='cuda', use_cifar10=True):
    """
    计算参考集的 FID_worst：
    计算参考集与独立数据集（CIFAR-10 或 ImageNet）之间的 FID
    """
    logger.info("Computing FID_worst with independent dataset...")
    
    if use_cifar10:
        logger.info("Using CIFAR-10 as independent dataset...")
        # 下载 CIFAR-10（首次会下载）
        independent_dataset = datasets.CIFAR10(
            root='./data/cifar10',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.Resize((256, 256), Image.Resampling.LANCZOS),
                transforms.ToTensor(),
            ])
        )
    else:
        logger.info("Using ImageNet as independent dataset...")
        
        independent_path = image_net_path
        if not os.path.exists(independent_path):
            raise FileNotFoundError(
                f"ImageNet dataset not found at {independent_path}. "
                "Please download ImageNet manually or use CIFAR-10 instead."
            )
        independent_dataset = SimpleImageDataset(independent_path)
    
    fid_worst = compute_fid(ref_dataset, independent_dataset, batch_size=batch_size, device=device)
    logger.info(f"FID_worst (vs independent dataset): {fid_worst:.4f}")
    
    return fid_worst

def main():
    # 加载配置
    logger.info(f"Loading config from {config_path}...")
    cfg = load_config(config_path)
    image_size = cfg['global_data'].get('image_size', [256, 256])
    image_size_tuple = (image_size[0], image_size[1])
    
    # 检查参考数据集
    if not os.path.exists(ref_dataset_path):
        logger.error(f"Reference dataset path {ref_dataset_path} does not exist!")
        sys.exit(1)
    
    # 加载参考数据集
    logger.info(f"Loading reference dataset from {ref_dataset_path}...")
    ref_dataset = SimpleImageDataset(ref_dataset_path, image_size=image_size_tuple)
    
    if len(ref_dataset) == 0:
        logger.error(f"No images found in {ref_dataset_path}!")
        sys.exit(1)
    
    # 选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # 计算 FID_best
    try:
        fid_best, fid_splits = compute_fid_best(
            ref_dataset, 
            num_splits=num_splits, 
            batch_size=32, 
            device=device
        )
    except Exception as e:
        logger.error(f"Error computing FID_best: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 计算 FID_worst
    try:
        fid_worst = compute_fid_worst(
            ref_dataset,
            batch_size=32,
            device=device,
            use_cifar10=use_cifar10
        )
    except Exception as e:
        logger.error(f"Error computing FID_worst: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 打印结果
    logger.info("\n" + "="*60)
    logger.info("FID Baselines Calculation Results")
    logger.info("="*60)
    logger.info(f"Reference Dataset Path: {ref_dataset_path}")
    logger.info(f"Number of Reference Images: {len(ref_dataset)}")
    logger.info(f"Image Size: {image_size}")
    logger.info(f"Random Seed: {random_seed}")
    logger.info(f"Number of Splits for FID_best: {num_splits}")
    logger.info(f"Independent Dataset: {'CIFAR-10' if use_cifar10 else 'ImageNet'}")
    logger.info("-"*60)
    logger.info(f"FID_best (median FID within reference set): {fid_best:.4f}")
    logger.info(f"FID_worst (FID vs independent dataset):     {fid_worst:.4f}")
    logger.info(f"FID Range: [{fid_best:.4f}, {fid_worst:.4f}]")
    logger.info("="*60 + "\n")
    
    # 保存结果到 txt 文件
    output_file = output_txt
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("FID Baselines Calculation Results\n")
        f.write("="*60 + "\n\n")
        f.write("Configuration Parameters:\n")
        f.write(f"  Reference Dataset Path: {ref_dataset_path}\n")
        f.write(f"  Number of Reference Images: {len(ref_dataset)}\n")
        f.write(f"  Image Size: {image_size}\n")
        f.write(f"  Random Seed: {random_seed}\n")
        f.write(f"  Number of Splits for FID_best: {num_splits}\n")
        f.write(f"  Independent Dataset: {'CIFAR-10' if use_cifar10 else 'ImageNet'}\n")
        f.write(f"  Compute Device: {device}\n\n")
        
        f.write("Calculation Results:\n")
        f.write(f"  FID_best (median FID within reference set): {fid_best:.4f}\n")
        f.write(f"  FID_worst (FID vs independent dataset):     {fid_worst:.4f}\n")
        f.write(f"  FID Range: [{fid_best:.4f}, {fid_worst:.4f}]\n\n")
        
        f.write("FID Scores from Each Split:\n")
        for i, fid_val in enumerate(fid_splits):
            f.write(f"  Split {i + 1}: {fid_val:.4f}\n")
        
        f.write("\nNormalization Formula:\n")
        f.write("  S_fid = (FID_worst - FID) / (FID_worst - FID_best)\n")
        f.write("  where FID is the computed FID between Sim and Real datasets\n")
        f.write("  Range: S_fid in [0, 1], 1 is best, 0 is worst\n")
    
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
