import os
import logging
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from .utils import find_matching_file, parse_yolo_box, compute_iou
from .transforms import ImagePreprocessor

logger = logging.getLogger(__name__)

class SimRealPairedDataset(Dataset):
    def __init__(self, cfg, stage='test'):
        """
        :param cfg: 配置字典
        :param stage: 'train' / 'val' / 'test' (预留接口)
        """
        self.cfg = cfg
        self.sim_root = Path(cfg['sim_path'])
        self.real_root = Path(cfg['real_path'])
        self.use_iou = cfg.get('use_iou', False)
        self.iou_threshold = cfg.get('iou_threshold', 0.8) # 默认 IoU 阈值 0.8
        
        # 初始化预处理器
        self.preprocessor = ImagePreprocessor(cfg)
        
        self.samples = self._scan_and_pair()

    def _scan_and_pair(self):
        """
        核心逻辑：
        1. 遍历 real 目录
        2. 寻找对应的 sim 图像
        3. 寻找对应的 real_txt 和 sim_txt
        4. 校验 IoU
        """
        valid_samples = []
        skipped_count = 0
        
        logger.info(f"Scanning data from {self.real_root}...")

        # 遍历 Real 目录下的所有文件 (支持子目录递归)
        # rglob('*') 会列出所有文件，我们只筛选图片
        extensions = {'.jpg', '.png', '.jpeg', '.bmp'}
        
        for real_img_path in self.real_root.rglob('*'):
            if real_img_path.suffix.lower() not in extensions:
                continue
                
            # 1. 获取相对路径 (例如: ship/1_123.jpg -> ship/1_123)
            rel_path_obj = real_img_path.relative_to(self.real_root)
            rel_path_no_ext = rel_path_obj.with_suffix('')
            
            # 2. 寻找 Sim 图像 (允许后缀不同)
            sim_img_path_str = find_matching_file(self.sim_root, rel_path_no_ext)
            if not sim_img_path_str:
                # print(f"Missing Sim image for: {rel_path_no_ext}")
                continue

            # 3. 处理 YOLO 标注 (同名 txt)（可选）
            real_txt_path = real_img_path.with_suffix('.txt')
            sim_txt_path = Path(sim_img_path_str).with_suffix('.txt')
            
            # 获取图像尺寸用于 YOLO 坐标反归一化
            # 注意：这里需要打开图像头读取尺寸，为了性能优化，假设 Sim 和 Real 尺寸在原始状态下可能不同，分别读取
            w_real, h_real = self.preprocessor.get_image_size(real_img_path)
            w_sim, h_sim = self.preprocessor.get_image_size(sim_img_path_str)
            
            # 解析 Box
            box_real = parse_yolo_box(real_txt_path, w_real, h_real)
            box_sim = parse_yolo_box(sim_txt_path, w_sim, h_sim)
            
            # 4. IoU 校验逻辑（可选）
            # 当 self.use_iou 为 True 时，会严格要求存在 Real 标注且 Sim 标注与 Real 的 IoU 满足阈值
            # 当 self.use_iou 为 False 时，跳过 IoU/标注检查，使用整张图作为 ROI（即 ROI 与 full 图一致）
            if self.use_iou:
                # 如果没有 Real 标注，无法确定 ROI，跳过
                if box_real is None:
                    continue

                final_box = box_real # 默认使用 Real 的框作为裁剪基准

                if box_sim is not None:
                    # 将 Sim Box 缩放到 Real 尺寸下进行 IoU 计算
                    scale_x = w_real / w_sim
                    scale_y = h_real / h_sim
                    box_sim_mapped = (
                        int(box_sim[0] * scale_x), int(box_sim[1] * scale_y),
                        int(box_sim[2] * scale_x), int(box_sim[3] * scale_y)
                    )

                    iou = compute_iou(box_real, box_sim_mapped)

                    if iou < self.iou_threshold:
                        logger.warning(f"Skipping {rel_path_no_ext}: IoU {iou:.2f} < {self.iou_threshold}")
                        skipped_count += 1
                        continue
                else:
                    # Sim 无标注但 Real 有标注，此时是否跳过取决于策略；在 use_iou=True 下我们选择跳过
                    logger.warning(f"Skipping {rel_path_no_ext}: Sim missing annotation.")
                    skipped_count += 1
                    continue
            else:
                # 不使用 IoU/标注时，将 ROI 设为 None，表示后续返回的 ROI 将是整张图（与 full 图一致）
                final_box = None

            # 5. 加入有效列表
            # 记录相对路径作为 ID，方便后续输出报告
            valid_samples.append({
                'sim_path': sim_img_path_str,
                'real_path': str(real_img_path),
                'roi_box': final_box, # (x1, y1, x2, y2) on Real Image scale
                'id': str(rel_path_no_ext)
            })
            
        logger.info(f"Dataset loaded: {len(valid_samples)} valid pairs, {skipped_count} skipped.")
        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 改方法改为按 self.use_iou 只返回两个 tensor
        if not self.use_iou:
            roi_real_tensor = self.preprocessor.load_and_process(item['real_path'], box=None)
            roi_sim_tensor = self.preprocessor.load_and_process(item['sim_path'], box=None)
            return roi_sim_tensor, roi_real_tensor, item['id']

        # 加载并处理图像
        # ROI-cropped tensors (用于局部/配对度量)
        # 1) 读取原始 ROI（Real 图像的 box）
        real_box = item['roi_box']

        # 2) 读取图片尺寸（Real 与 Sim）以便映射坐标
        w_real, h_real = self.preprocessor.get_image_size(item['real_path'])
        w_sim, h_sim = self.preprocessor.get_image_size(item['sim_path'])

        # 3) 将 real_box 映射到 sim 图像的坐标系（一次计算）
        sim_box_mapped = (
            int(real_box[0] * (w_sim / w_real)),
            int(real_box[1] * (h_sim / h_real)),
            int(real_box[2] * (w_sim / w_real)),
            int(real_box[3] * (h_sim / h_real))
        )

        # 4) 使用映射后的框加载并预处理 ROI tensors
        roi_real_tensor = self.preprocessor.load_and_process(item['real_path'], box=real_box)
        roi_sim_tensor = self.preprocessor.load_and_process(item['sim_path'], box=sim_box_mapped)

        # 返回顺序：sim_roi, real_roi, id
        return roi_sim_tensor, roi_real_tensor, item['id']


class SingleDomainDataset(Dataset):
    def __init__(self, cfg, domain='sim', stage='test'):
        """
        :param cfg: 配置字典
        :param domain: 'sim' 或 'real'
        :param stage: 'train' / 'val' / 'test' (预留接口)
        """
        self.cfg = cfg
        self.domain = domain

        if domain == 'sim':
            self.root = Path(cfg['sim_path'])
        elif domain == 'real':
            self.root = Path(cfg['real_path'])
        else:
            raise ValueError("domain must be 'sim' or 'real'")
         
        # 初始化预处理器
        self.preprocessor = ImagePreprocessor(cfg)
        
        self.samples = self._scan_files()

    def _scan_files(self):
        """
        扫描目录下所有图片文件
        """
        valid_samples = []
        
        logger.info(f"Scanning data from {self.root}...")

        extensions = {'.jpg', '.png', '.jpeg', '.bmp'}
        
        for img_path in self.root.rglob('*'):
            if img_path.suffix.lower() not in extensions:
                continue
                
            rel_path_obj = img_path.relative_to(self.root)
            rel_path_no_ext = rel_path_obj.with_suffix('')
            
            valid_samples.append({
                'img_path': str(img_path),
                'id': str(rel_path_no_ext)
            })
            
        logger.info(f"Dataset-{self.domain} loaded: {len(valid_samples)} images.")
        return valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 加载并处理图像
        img_tensor = self.preprocessor.load_and_process(item['img_path'], box=None)

        return img_tensor, item['id']

def build_paired_dataloader(cfg):
    dataset = SimRealPairedDataset(cfg)
    
    return DataLoader(
        dataset,
        batch_size=cfg.get('batch_size', 16),
        shuffle=False, # 评估不需要 Shuffle
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True
    )

def build_global_dataloader(cfg):
    dataset_sim = SingleDomainDataset(cfg, domain='sim')
    dataset_real = SingleDomainDataset(cfg, domain='real')

    dataloader_sim = DataLoader(
        dataset_sim,
        batch_size=cfg.get('batch_size', 16),
        shuffle=False, # 评估不需要 Shuffle
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True
    )

    dataloader_real = DataLoader(
        dataset_real,
        batch_size=cfg.get('batch_size', 16),
        shuffle=False, # 评估不需要 Shuffle
        num_workers=cfg.get('num_workers', 4),
        pin_memory=True
    )

    return dataloader_sim, dataloader_real