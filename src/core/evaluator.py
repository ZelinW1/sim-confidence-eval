import torch
import tqdm
import os
from collections import defaultdict

from src.data.datasets import build_paired_dataloader, build_global_dataloader
from src.utils.logger import setup_logger
from src.utils.visualizer import Visualizer

from src.metrics.classical import PSNR, SSIM, FSIM, MS_SSIM
from src.metrics.deep import LPIPS, FID

class Evaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Setup Output & Logging
        self.output_dir = cfg.get('output_dir', './outputs')
        self.logger = setup_logger(self.output_dir)
        self.visualizer = Visualizer(self.output_dir)
        
        self.logger.info(f"Initializing Evaluator on {self.device}...")

        # 2. Build Metrics
        if 'metrics' not in cfg or not cfg['metrics']:
            self.logger.warning("No metrics specified in configuration.")
            self.metrics = []
        else:
            self.metrics = self._load_metrics(cfg['metrics'])
        
        # 3. Build DataLoader
        self.loader_paired = build_paired_dataloader(cfg['paired_data'])
        self.loader_global_sim, self.loader_global_real = build_global_dataloader(cfg['global_data'])
        
        # 4. Storage for results
        # 结构: self.results['SSIM'] = [{'id': 'x', 'score': 0.8, 'sim_path': '...', 'real_path': '...'}, ...]
        self.results = defaultdict(list) 
        self.global_results = {} # 存储 FID 等全局指标结果

    def _load_metrics(self, metric_cfg_list):
        """工厂模式加载指标"""
        available_metrics = {
            'PSNR': PSNR,
            'SSIM': SSIM,
            'MS_SSIM': MS_SSIM,
            'FSIM': FSIM,
            'LPIPS': LPIPS,
            'FID': FID
        }
        
        loaded = []
        for m_cfg in metric_cfg_list:
            name = m_cfg['name']
            params = m_cfg.get('params', {})
            
            if name in available_metrics:
                # 实例化并移至 GPU
                metric_instance = available_metrics[name](params).to(self.device)
                loaded.append(metric_instance)
                self.logger.info(f"Metric loaded: {name}")
            else:
                self.logger.warning(f"Metric {name} not found in registry, skipping.")
        
        return loaded

    def run(self):
        """主执行循环"""

        # metrics 为空的处理
        if not self.metrics:
            self.logger.error("No metrics loaded. Exiting evaluation.")
            return

        self.logger.info("Starting evaluation loop...")
        
        # 切换到 eval 模式 (针对 LPIPS/Inception 等网络)
        for m in self.metrics:
            m.eval()

        # 先处理 paired 数据集（局部指标）
        self.logger.info("Evaluating on paired dataset...")
        # 进度条
        pbar = tqdm.tqdm(self.loader_paired, desc="Processing Batches")
        
        with torch.no_grad(): # 全局禁用梯度，节省显存
            for batch_idx, batch in enumerate(pbar):
                # 支持两种 dataloader 输出格式：
                # (sim_roi, real_roi, ids) 或 (sim_roi, real_roi, ids, full_sim, full_real)
                if len(batch) == 3:
                    sim_imgs, real_imgs, ids = batch
                else:
                    # 不支持的返回结构，跳过
                    self.logger.error(f"Unexpected batch structure of length {len(batch)}. Skipping batch {batch_idx}.")
                    continue

                # 1. Move tensors to device
                sim_imgs = sim_imgs.to(self.device)
                real_imgs = real_imgs.to(self.device)

                # 2. Iterate metrics
                for metric in self.metrics:
                    if getattr(metric, 'is_global', False):
                        continue

                    scores = metric(sim_imgs, real_imgs) # Returns (B,) tensor

                    # 如果 scores 是 tuple (ori_result, normalized_result) 都保存
                    if isinstance(scores, tuple):
                        ori_scores, norm_scores = scores
                        scores_list_ori = ori_scores.cpu().tolist()
                        scores_list_norm = norm_scores.cpu().tolist()
                    else:
                        scores_list_ori = scores.cpu().tolist()
                        scores_list_norm = None

                    # 记录结果
                    for i, score in enumerate(scores_list_ori):
                        # 这里假设 loader 在 Dataset 中返回的顺序和 ids 是一致的
                        record = {
                            'id': ids[i],
                            'score_ori': scores_list_ori[i],
                            'score_norm': scores_list_norm[i] if scores_list_norm is not None else None,
                            # 如果需要，可以在 Dataset 中额外返回路径并记录
                        }
                        self.results[metric.name].append(record)

        # 3. 再处理 global 数据集（全局指标）
        self.logger.info("Evaluating on global dataset...")
        # 进度条   sim 和 real 分开单独迭代
        pbar_global_sim = tqdm.tqdm(self.loader_global_sim, desc="Processing Global Sim Batches", total=len(self.loader_global_sim))
        
        with torch.no_grad(): # 全局禁用梯度，节省显存
            for batch_idx, batch_sim in enumerate(pbar_global_sim):
                # 支持两种 dataloader 输出格式：
                # (sim_full, ids) 或 (sim_full, ids, roi_sim)
                if len(batch_sim) == 2:
                    full_sim_imgs, ids_sim = batch_sim
                elif len(batch_sim) == 3:
                    full_sim_imgs, ids_sim, _ = batch_sim
                else:
                    self.logger.error(f"Unexpected sim batch structure of length {len(batch_sim)}. Skipping batch {batch_idx}.")
                    continue

                # 1. Move full-image tensors to device
                full_sim_imgs = full_sim_imgs.to(self.device)

                # 2. Iterate metrics
                for metric in self.metrics:
                    if getattr(metric, 'is_global', False):
                        metric.update(full_sim_imgs, real=False)

        pbar_global_real = tqdm.tqdm(self.loader_global_real, desc="Processing Global Real Batches", total=len(self.loader_global_real))

        with torch.no_grad(): # 全局禁用梯度，节省显存
            for batch_idx, batch_real in enumerate(pbar_global_real):
                # 支持两种 dataloader 输出格式：
                # (real_full, ids) 或 (real_full, ids, roi_real)
                if len(batch_real) == 2:
                    full_real_imgs, ids_real = batch_real
                elif len(batch_real) == 3:
                    full_real_imgs, ids_real, _ = batch_real
                else:
                    self.logger.error(f"Unexpected real batch structure of length {len(batch_real)}. Skipping batch {batch_idx}.")
                    continue

                # 1. Move full-image tensors to device
                full_real_imgs = full_real_imgs.to(self.device)

                # 2. Iterate metrics
                for metric in self.metrics:
                    if getattr(metric, 'is_global', False):
                        metric.update(full_real_imgs, real=True)

        for metric in self.metrics:
            if getattr(metric, 'is_global', False):
                self.logger.info(f"Computing global metric: {metric.name}...")
                final_score = metric.compute()
                self.global_results[metric.name] = final_score.item()
                self.logger.info(f"Global Result - {metric.name}: {final_score.item():.4f}")

        # 4. Finalize
        self.finalize()

    def finalize(self):
        """后处理：保存、绘图、Bad Case 分析"""
        self.logger.info("Finalizing results...")
        
        # 1. 保存 Local Metrics 到 CSV
        if self.results:
            df = self.visualizer.save_csv(self.results)
            
            # 2. 打印均值统计
            stats_str = "\n" + "="*30 + " Evaluation Report " + "="*30 + "\n"
            for col in df.columns:
                if col == 'id': continue
                mean_val = df[col].mean()
                std_val = df[col].std()
                stats_str += f"{col:10s}: Mean = {mean_val:.4f} | Std = {std_val:.4f}\n"
            
            # 追加全局指标
            for name, val in self.global_results.items():
                stats_str += f"{name:10s}: {val:.4f} (Global)\n"
            
            stats_str += "="*79
            self.logger.info(stats_str)

            # 3. 绘图
            if self.cfg['visualization'].get('plot_hist', True):
                self.visualizer.plot_distributions(df)

            # # 4. Bad Cases 分析
            # # 假设我们想看 LPIPS 最差的 (LPIPS 越大越差)
            # # 或者 SSIM 最差的 (SSIM 越小越差)
            # if self.cfg['visualization'].get('save_bad_cases', False):
            #     # 简单示例：针对列表中的第一个 Metric 进行 Bad Case 导出
            #     target_metric = self.metrics[0].name
            #     # 判断方向 (通常 dataset 或者 metric 应该定义 higher_is_better)
            #     # 这里硬编码做个示例，实际可配置
            #     higher_is_better = True if target_metric in ['SSIM', 'PSNR', 'FSIM'] else False
                
            #     self.logger.info(f"Saving bad cases based on {target_metric}...")
            #     # 注意：Visualizer 需要 source path 才能复制文件。
            #     # 目前 results 里只有 id。如果需要复制文件，
            #     # 需要修改 Dataset __getitem__ 返回完整路径，并在 run loop 中记录。
        
        else:
            self.logger.warning("No local results computed.")