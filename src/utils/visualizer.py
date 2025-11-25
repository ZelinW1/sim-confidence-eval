import os
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, 'plots')
        self.bad_cases_dir = os.path.join(output_dir, 'bad_cases')
        
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # 设置绘图风格
        sns.set_theme(style="whitegrid")

    def save_csv(self, results_dict, filename="final_report.csv"):
        """
        将结果字典转换为 DataFrame 并保存
        results_dict structure: {'MetricName': [{'id': 'ship/1', 'score': 0.9}, ...]}
        """
        # 1) 构建一个扁平（flat）DataFrame 保持与旧行为兼容并作为返回值
        # 假设所有 metrics 的列表顺序一致
        if not results_dict:
            print("No results to save.")
            return pd.DataFrame()

        metric_names = list(results_dict.keys())
        first_metric = metric_names[0]
        ids = [item['id'] for item in results_dict[first_metric]]

        # 扁平 DataFrame：id (原始 id，如 'ship\\1_1743') + 各 metric 列
        flat_df = pd.DataFrame({'id': ids})
        for metric_name in metric_names:
            scores = [item['score'] for item in results_dict[metric_name]]
            flat_df[metric_name] = scores

        # 2) 生成按类别分组的数据结构（category, file_id, metrics...）
        # 处理 ID：支持 'cat\\file' 或 'cat/file' 或 仅 'file'
        rows = []
        for idx, full_id in enumerate(ids):
            # 处理分隔符，优先反斜杠（Windows 风格）
            if isinstance(full_id, str) and ('\\' in full_id or '/' in full_id):
                if '\\' in full_id:
                    parts = full_id.split('\\', 1)
                else:
                    parts = full_id.split('/', 1)
                category = parts[0]
                file_id = parts[1] if len(parts) > 1 else ''
            else:
                category = ''
                file_id = full_id

            row = {'category': category, 'id': file_id}
            for metric_name in metric_names:
                row[metric_name] = results_dict[metric_name][idx]['score']
            rows.append(row)

        grouped = {}
        for r in rows:
            grouped.setdefault(r['category'], []).append(r)

        # 3) 保存详细版 CSV：按 category 分块，块内为文件行，块尾附加 Mean/Std 行
        detailed_path = os.path.join(self.output_dir, filename.replace('.csv', '_detailed.csv'))
        detailed_rows = []
        for cat, items in grouped.items():
            # 可选：在分组前插入一行作为分组标题（在 CSV 中用空行和一行标题）
            # 这里我们插入一行 category header（category 放在 category 列，id 置空）
            detailed_rows.append({'category': cat, 'id': '' , **{m: '' for m in metric_names}})

            # 每个文件的记录
            for it in items:
                detailed_rows.append(it)

            # 统计均值与标准差
            df_block = pd.DataFrame(items)
            means = df_block[metric_names].mean()
            stds = df_block[metric_names].std()

            mean_row = {'category': cat, 'id': 'MEAN'}
            std_row = {'category': cat, 'id': 'STD'}
            for m in metric_names:
                mean_row[m] = means[m]
                std_row[m] = stds[m]

            detailed_rows.append(mean_row)
            detailed_rows.append(std_row)

            # 插入空行以作分组间隔
            detailed_rows.append({'category': '', 'id': '' , **{m: '' for m in metric_names}})

        detailed_df = pd.DataFrame(detailed_rows)
        os.makedirs(self.output_dir, exist_ok=True)
        detailed_df.to_csv(detailed_path, index=False)
        print(f"Detailed report saved to {detailed_path}")

        # 4) 保存简略版 CSV：每个 category 一行，仅包含每个指标的 mean/std
        summary_path = os.path.join(self.output_dir, filename.replace('.csv', '_summary.csv'))
        summary_rows = []
        for cat, items in grouped.items():
            df_block = pd.DataFrame(items)
            means = df_block[metric_names].mean()
            stds = df_block[metric_names].std()
            row = {'category': cat}
            for m in metric_names:
                row[f'{m}_mean'] = means[m]
                row[f'{m}_std'] = stds[m]
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary report saved to {summary_path}")

        # # 5) 同时保留原始兼容 CSV（平铺形式），以便 Evaluator 继续使用
        # flat_path = os.path.join(self.output_dir, filename)
        # flat_df.to_csv(flat_path, index=False)
        # print(f"Flat report saved to {flat_path}")

        return flat_df

    def plot_distributions(self, df):
        """绘制各指标的分数分布直方图"""
        metric_cols = [c for c in df.columns if c != 'id']
        
        for metric in metric_cols:
            plt.figure(figsize=(8, 6))
            sns.histplot(df[metric], kde=True, bins=20)
            plt.title(f'{metric} Score Distribution')
            plt.xlabel('Score')
            plt.ylabel('Count')
            
            plt.savefig(os.path.join(self.plots_dir, f'{metric}_dist.png'))
            plt.close()

    def save_bad_cases(self, df, data_root_sim, data_root_real, top_k=5, metric='LPIPS', higher_is_better=False):
        """
        保存表现最差的 Top-K 图片对以便人工分析
        :param metric: 依据哪个指标排序
        :param higher_is_better: True(如SSIM), False(如LPIPS)
        """
        if metric not in df.columns:
            return

        # 排序
        sorted_df = df.sort_values(by=metric, ascending=higher_is_better)
        bad_cases = sorted_df.head(top_k)
        
        save_dir = os.path.join(self.bad_cases_dir, metric)
        os.makedirs(save_dir, exist_ok=True)
        
        for idx, row in bad_cases.iterrows():
            img_id = row['id'] # e.g. "ship/1_123"
            score = row[metric]
            
            # 构建原始文件路径 (需要利用 utils.find_matching_file 的逻辑，这里简化处理)
            # 注意：这里我们假设 dataloader 里的 id 是相对路径（不含后缀）
            # 为了简单，我们重新搜索或由 Dataset 传递完整路径。
            # 由于 Dataset 没传完整路径给 Evaluator，这里为了通用性，
            # 建议只生成一个 HTML 或 Markdown 报告引用原图，或者简单的 copy 逻辑。
            
            # 这里演示简单的 Copy 逻辑 (假设后缀已知或尝试常见后缀)
            # *注：为了代码健壮性，实际工程中最好在 results 里存一下 full path*
            pass 
            # (此处留空，建议在 Evaluator 收集结果时直接把 path 存下来，
            # 下面的 Evaluator 代码会配合修改结构)